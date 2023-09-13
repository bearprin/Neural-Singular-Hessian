import os
import random
import io
import warnings

from tensorboardX import SummaryWriter
from PIL import Image
from scipy.spatial import KDTree as KDTree
from tqdm import tqdm
from skimage import measure
import trimesh

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad
import torch.backends.cudnn as cudnn

def get_aver(distances, face):
    return (distances[face[0]] + distances[face[1]] + distances[face[2]]) / 3.0


def remove_far(gt_pts, mesh, dis_trunc=0.1, is_use_prj=False):
    # gt_pts: trimesh
    # mesh: trimesh

    gt_kd_tree = KDTree(gt_pts)
    distances, vertex_ids = gt_kd_tree.query(mesh.vertices, p=2, distance_upper_bound=dis_trunc, workers=-1)
    faces_remaining = []
    faces = mesh.faces

    if is_use_prj:
        normals = gt_pts.vertex_normals
        closest_points = gt_pts.vertices[vertex_ids]
        closest_normals = normals[vertex_ids]
        direction_from_surface = mesh.vertices - closest_points
        distances = direction_from_surface * closest_normals
        distances = np.sum(distances, axis=1)

    for i in range(faces.shape[0]):
        if get_aver(distances, faces[i]) < dis_trunc:
            faces_remaining.append(faces[i])
    print(len(faces_remaining))
    mesh_cleaned = mesh.copy()
    mesh_cleaned.faces = faces_remaining
    mesh_cleaned.remove_unreferenced_vertices()

    return mesh_cleaned


def remove_outlier(gt_pts, q_pts, dis_trunc=0.003):
    # gt_pts: trimesh
    # mesh: trimesh
    gt_kd_tree = KDTree(gt_pts)
    distances, q_ids = gt_kd_tree.query(q_pts, p=2, distance_upper_bound=dis_trunc, workers=-1)
    q_pts = q_pts[distances < dis_trunc]
    return q_pts


def same_seed(seed):
    """

    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def normalize_mesh_export(mesh, file_out=None):
    # unit to [-0.5, 0.5]
    bounds = mesh.extents
    if bounds.min() == 0.0:
        return

    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation)

    # scale to unit cube
    scale = 1.0 / bounds.max()
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)
    if file_out is not None:
        mesh.export(file_out)
    return mesh


def eval_reconstruct_gt(rec_mesh: trimesh.Trimesh, gt_pts, sample_num=100000):
    def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
        ''' Computes minimal distances of each point in points_src to points_tgt.

        Args:
            points_src (numpy array): source points
            normals_src (numpy array): source normals
            points_tgt (numpy array): target points
            normals_tgt (numpy array): target normals
        '''
        kdtree = KDTree(points_tgt)
        dist, idx = kdtree.query(points_src, workers=-1)

        if normals_src is not None and normals_tgt is not None:
            normals_src = \
                normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
            normals_tgt = \
                normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

            #        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
            #        # Handle normals that point into wrong direction gracefully
            #        # (mostly due to mehtod not caring about this in generation)
            #        normals_dot_product = np.abs(normals_dot_product)

            normals_dot_product = np.abs(normals_tgt[idx] * normals_src)
            normals_dot_product = normals_dot_product.sum(axis=-1)
        else:
            normals_dot_product = np.array(
                [np.nan] * points_src.shape[0], dtype=np.float32)
        return dist, normals_dot_product

    def get_threshold_percentage(dist, thresholds):
        ''' Evaluates a point cloud.
        Args:
            dist (numpy array): calculated distance
            thresholds (numpy array): threshold values for the F-score calculation
        '''
        in_threshold = [
            (dist <= t).mean() for t in thresholds
        ]
        return in_threshold

    def eval_pointcloud(pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None, thresholds=[0.005]):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''
        # Return maximum losses if pointcloud is empty

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness ** 2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy ** 2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()
        # print(completeness,accuracy,completeness2,accuracy2)
        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (
                0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)

        # F-Score
        F = [
            2 * precision[i] * recall[i] / (precision[i] + recall[i])
            for i in range(len(precision))
        ]
        return normals_correctness, chamferL1, chamferL2, F[0]

    gt_pts = normalize_mesh_export(gt_pts)
    rec_mesh = normalize_mesh_export(rec_mesh)

    # sample point for rec
    pts_rec, idx = rec_mesh.sample(sample_num, return_index=True)
    normals_rec = rec_mesh.face_normals[idx]
    # sample point for gt
    pts_gt = None
    normals_gt = None
    if isinstance(gt_pts, trimesh.PointCloud):
        if gt_pts.shape[0] < sample_num:
            sample_num = gt_pts.shape[0]
        idx = np.random.choice(gt_pts.vertices.shape[0], sample_num, replace=False)
        pts_gt = gt_pts.vertices[idx]
        normals_gt = None
    elif isinstance(gt_pts, trimesh.Trimesh):
        pts_gt, idx = gt_pts.sample(sample_num, return_index=True)
        normals_gt = gt_pts.face_normals[idx]

    normals_correctness, chamferL1, chamferL2, f1_mu = eval_pointcloud(pts_rec, pts_gt, normals_rec, normals_gt)

    return normals_correctness, chamferL1, chamferL2, f1_mu


def center_and_scale(points, cp=None, scale=None):
    # center a point cloud and scale it to unite sphere.
    if cp is None:
        cp = points.mean(axis=1)
    points = points - cp[:, None, :]
    if scale is None:
        scale = np.linalg.norm(points, axis=-1).max(-1)
    points = points / scale[:, None, None]
    return points, cp, scale


def log_losses(writer, epoch, bach_idx, num_batches, loss_dict, batch_size):
    # helper function to log losses to tensorboardx writer
    fraction_done = (bach_idx + 1) / num_batches
    iteration = (epoch + fraction_done) * num_batches * batch_size
    for loss in loss_dict.keys():
        writer.add_scalar(loss, loss_dict[loss].item(), iteration)
    return iteration


def log_weight_hist(writer, epoch, bach_idx, num_batches, net_blocks, batch_size):
    # helper function to log losses to tensorboardx writer
    fraction_done = (bach_idx + 1) / num_batches
    iteration = (epoch + fraction_done) * num_batches * batch_size
    for i, block in enumerate(net_blocks):
        writer.add_histogram('layer_weights_' + str(i), block[0].weight, iteration)
        writer.add_histogram('layer_biases_' + str(i), block[0].bias
                             , iteration)
    return iteration


def log_images(writer, iteration, contour_img, curl_img, eikonal_img, div_image, z_diff_img, example_idx):
    # helper function to log images to tensorboardx writer
    writer.add_image('implicit_function/' + str(example_idx), contour_img.transpose(2, 0, 1), iteration)
    writer.add_image('curl/' + str(example_idx), curl_img.transpose(2, 0, 1), iteration)
    writer.add_image('eikonal_term/' + str(example_idx), eikonal_img.transpose(2, 0, 1), iteration)
    writer.add_image('divergence/' + str(example_idx), div_image.transpose(2, 0, 1), iteration)
    writer.add_image('z_diff/' + str(example_idx), z_diff_img.transpose(2, 0, 1), iteration)


def log_string(out_str, log_file):
    # helper function to log a string to file and print it
    log_file.write(out_str + '\n')
    log_file.flush()
    print(out_str)


def setup_logdir(logdir, args=None):
    # helper function to set up logging directory

    os.makedirs(logdir, exist_ok=True)
    log_writer_train = SummaryWriter(os.path.join(logdir, 'train'))
    log_writer_test = SummaryWriter(os.path.join(logdir, 'test'))
    log_filename = os.path.join(logdir, 'out.log')
    log_file = open(log_filename, 'w')
    model_outdir = os.path.join(logdir, 'trained_models')
    os.makedirs(model_outdir, exist_ok=True)

    if args is not None:
        params_filename = os.path.join(model_outdir, '%s_params.pth' % (args.model_name))
        torch.save(args, params_filename)  # save parameters
        log_string("input params: \n" + str(args), log_file)
    else:
        warnings.warn("Training options not provided. Not saving training options...")

    return log_file, log_writer_train, log_writer_test, model_outdir


def backup_code(logdir, dir_list=[], file_list=[]):
    # backup models code
    code_bkp_dir = os.path.join(logdir, 'code_bkp')
    os.makedirs(code_bkp_dir, exist_ok=True)
    for dir_name in dir_list:
        print("copying directory {} to {}".format(dir_name, code_bkp_dir))
        os.system('cp -r %s %s' % (dir_name, code_bkp_dir))  # backup the current model code
    for file_name in file_list:
        print("copying file {} to {}".format(file_name, code_bkp_dir))
        os.system('cp %s %s' % (file_name, code_bkp_dir))


def plotly_fig2array(fig):
    # convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf).convert('RGB')
    img = np.asarray(img)
    return img


def gradient(inputs, outputs, create_graph=True, retain_graph=True):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True)[0]  # [:, -3:]
    return points_grad


# def gradient(inputs, outputs):
#     outputs.sum().backward(retain_graph=True)
#     return inputs.grad
# sdf = net(q)
# sdf.sum().
# q_grad = q.grad.detach()  # (1, 500, 3)
# q_grad = F.normalize(q_grad, dim=2)


def get_cuda_ifavailable(torch_obj, device=None):
    # if cuda is available return a cuda obeject
    if (torch.cuda.is_available()):
        return torch_obj.cuda(device=device)
    else:
        return torch_obj


def compute_props(decoder, latent, z_gt, device):
    # compute derivative properties on a grid
    res = z_gt.shape[1]
    x, y, grid_points = get_2d_grid_uniform(resolution=res, range=1.2, device=device)
    grid_points.requires_grad_()
    if latent is not None:
        grid_points_latent = torch.cat([latent.expand(grid_points.shape[0], -1), grid_points], dim=1)
    else:
        grid_points_latent = grid_points
    z = decoder(grid_points_latent)
    z_np = z.detach().cpu().numpy().reshape(x.shape[0], x.shape[0])

    # plot z difference image
    z_diff = np.abs(np.abs(np.reshape(z_np, [res, res])) - np.abs(z_gt)).reshape(x.shape[0], x.shape[0])

    return x, y, z_np, z_diff


def compute_deriv_props(decoder, latent, z_gt, device):
    # compute derivative properties on a grid
    res = z_gt.shape[1]
    x, y, grid_points = get_2d_grid_uniform(resolution=res, range=1.2, device=device)
    grid_points.requires_grad_()
    if latent is not None:
        grid_points_latent = torch.cat([latent.expand(grid_points.shape[0], -1), grid_points], dim=1)
    else:
        grid_points_latent = grid_points
    z = decoder(grid_points_latent)
    z_np = z.detach().cpu().numpy().reshape(x.shape[0], x.shape[0])

    # compute derivatives
    grid_grad = gradient(grid_points, z)
    dx = gradient(grid_points, grid_grad[:, 0], create_graph=False, retain_graph=True)
    dy = gradient(grid_points, grid_grad[:, 1], create_graph=False, retain_graph=False)

    grid_curl = (dx[:, 1] - dy[:, 0]).cpu().detach().numpy().reshape(x.shape[0], x.shape[0])

    # compute eikonal term (gradient magnitude)
    eikonal_term = ((grid_grad.norm(2, dim=-1) - 1) ** 2).cpu().detach().numpy().reshape(x.shape[0], x.shape[0])

    # compute divergence
    grid_div = (dx[:, 0] + dy[:, 1]).detach().cpu().numpy().reshape(x.shape[0], x.shape[0])

    # compute det of hessian
    hessian = torch.stack([dx, dy], dim=1)
    eigs = torch.linalg.eigvalsh(hessian).detach().cpu().numpy().reshape(x.shape[0], x.shape[0], 2)
    hessian_det = torch.det(hessian).cpu().detach().numpy().reshape(x.shape[0], x.shape[0])

    # plot z difference image
    z_diff = np.abs(np.abs(np.reshape(z_np, [res, res])) - np.abs(z_gt)).reshape(x.shape[0], x.shape[0])

    return x, y, z_np, z_diff, eikonal_term, grid_div, grid_curl, hessian_det, eigs


def get_2d_grid_uniform(resolution=100, range=1.2, device=None):
    # generate points on a uniform grid within  a given range
    x = np.linspace(-range, range, resolution)
    y = x
    xx, yy = np.meshgrid(x, y)
    grid_points = get_cuda_ifavailable(torch.tensor(np.vstack([xx.ravel(), yy.ravel()]).T, dtype=torch.float),
                                       device=device)
    return x, y, grid_points


def get_3d_grid(resolution=100, bbox=1.2 * np.array([[-1, 1], [-1, 1], [-1, 1]]), device=None, eps=0.1,
                dtype=np.float16):
    # generate points on a uniform grid within  a given range
    # reimplemented from SAL : https://github.com/matanatz/SAL/blob/master/code/utils/plots.py
    # and IGR : https://github.com/amosgropp/IGR/blob/master/code/utils/plots.py

    shortest_axis = np.argmin(bbox[:, 1] - bbox[:, 0])
    if (shortest_axis == 0):
        x = np.linspace(bbox[0, 0] - eps, bbox[0, 1] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(bbox[1, 0] - eps, bbox[1, 1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(bbox[2, 0] - eps, bbox[2, 1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(bbox[1, 0] - eps, bbox[1, 1] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(bbox[0, 0] - eps, bbox[0, 1] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(bbox[2, 0] - eps, bbox[2, 1] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(bbox[2, 0] - eps, bbox[2, 1] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(bbox[0, 0] - eps, bbox[0, 1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(bbox[1, 0] - eps, bbox[1, 1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x.astype(dtype), y.astype(dtype), z.astype(dtype))  #
    # grid_points = get_cuda_ifavailable(torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float16),
    #                                          device=device)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float32)
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}


def scale_pc_to_unit_sphere(points, cp=None, s=None):
    if cp is None:
        cp = points.mean(axis=0)
    points = points - cp[None, :]
    if s is None:
        s = np.linalg.norm(points, axis=-1).max(-1)
    points = points / s
    return points, cp, s


def recon_metrics(pc1, pc2, one_sided=False, alphas=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
                  percentiles=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95], k=[10, 25, 50], return_all=False):
    # Compute reconstruction benchmarc evaluation metrics :
    # chamfer and hausdorff distance metrics between two point clouds pc1 and pc2 [nx3]
    # percentage of distance points metric (not used in the paper)
    # and the meal local chamfer variance
    # pc1 is the reconstruction and pc2 is the gt data

    scale = np.abs(pc2).max()

    # compute one side
    pc1_kd_tree = KDTree(pc1)
    one_distances, one_vertex_ids = pc1_kd_tree.query(pc2, n_jobs=4)
    cd12 = np.mean(one_distances)
    hd12 = np.max(one_distances)
    cdmed12 = np.median(one_distances)
    cd21 = None
    hd21 = None
    cdmed21 = None
    pods2 = None
    cdp2 = None
    chamfer_distance = cd12
    hausdorff_distance = hd12

    # compute chamfer distance percentiles cdp
    cdp1 = np.percentile(one_distances, percentiles, interpolation='lower')

    # compute PoD
    pod1 = []
    for alpha in alphas:
        pod1.append((one_distances < alpha * scale).sum() / one_distances.shape[0])

    if not one_sided:
        # compute second side
        pc2_kd_tree = KDTree(pc2)
        two_distances, two_vertex_ids = pc2_kd_tree.query(pc1, n_jobs=4)
        cd21 = np.mean(two_distances)
        hd21 = np.max(two_distances)
        cdmed21 = np.median(two_distances)
        chamfer_distance = 0.5 * (cd12 + cd21)
        hausdorff_distance = np.max((hd12, hd21))
        # compute chamfer distance percentiles cdp
        cdp2 = np.percentile(two_distances, percentiles)
        # compute PoD
        pod2 = []
        for alpha in alphas:
            pod2.append((two_distances < alpha * scale).sum() / two_distances.shape[0])

    # compute double sided pod
    pod12 = []
    for alpha in alphas:
        pod12.append(((one_distances < alpha * scale).sum() + (two_distances < alpha * scale).sum()) /
                     (one_distances.shape[0] + two_distances.shape[0]))
    cdp12 = np.percentile(np.concatenate([one_distances, two_distances]),
                          percentiles)  # compute chamfer distance percentiles cdp

    nn1_dist, local_idx2 = pc1_kd_tree.query(pc1, max(k), n_jobs=-1)
    nn1_dist_2pc2 = two_distances[local_idx2]
    malcv = [(nn1_dist_2pc2[:, :k0] / nn1_dist.mean(axis=1, keepdims=True)).var(axis=1).mean() for k0 in k]

    if return_all:
        return chamfer_distance, hausdorff_distance, (cd12, cd21, cdmed12, cdmed21, hd12, hd21), (pod1, pod2, pod12), \
            (cdp1.tolist(), cdp2.tolist(), cdp12.tolist()), malcv, (one_distances, two_distances)

    return chamfer_distance, hausdorff_distance, (cd12, cd21, cdmed12, cdmed21, hd12, hd21), (pod1, pod2, pod12), \
        (cdp1.tolist(), cdp2.tolist(), cdp12.tolist()), malcv


def load_reconstruction_data(file_path, n_points=30000, sample_type='vertices'):
    extension = file_path.split('.')[-1]
    if extension == 'xyz':
        points = np.loadtxt(file_path)
    elif extension == 'ply':
        mesh = trimesh.load_mesh(file_path)

        if hasattr(mesh, 'faces') and not sample_type == 'vertices':
            # sample points if its a triangle mesh
            points = trimesh.sample.sample_surface(mesh, n_points)[0]
        else:
            # use the vertices if its a point cloud
            points = mesh.vertices
    # Center and scale points
    # cp = points.mean(axis=0)
    # points = points - cp[None, :]
    # scale = np.abs(points).max()
    # points = points / scale
    return np.array(points).astype('float32')


def implicit2mesh(decoder, mods, grid_res, translate=[0., 0., 0.], scale=1, get_mesh=True, device=None,
                  bbox=np.array([[-1, 1], [-1, 1], [-1, 1]]), feat=None, hash_tree=None):
    # compute a mesh from the implicit representation in the decoder.
    # Uses marching cubes.
    # reimplemented from SAL get surface trace function : https://github.com/matanatz/SAL/blob/master/code/utils/plots.py
    print('in implicit2mesh')
    print(grid_res, translate, scale, bbox)
    mesh = None
    grid_dict = get_3d_grid(resolution=grid_res, bbox=bbox, device=device)
    print('Finished getting grid_dict')
    cell_width = grid_dict['xyz'][0][2] - grid_dict['xyz'][0][1]
    pnts = grid_dict["grid_points"]

    z = []
    for point in tqdm(torch.split(pnts, 100000, dim=0)):
        # point: (100000, 3)
        point = get_cuda_ifavailable(point, device=device)
        if feat is not None:
            if point.dim() == 2:
                point = point.unsqueeze(0)
            query_feat = decoder.encoder.query_feature(feat, point)
            z.append(decoder.decoder(point, query_feat).detach().squeeze(0).cpu().numpy())
        elif hash_tree is not None:
            if point.dim() == 2:
                point = point.unsqueeze(0)
            # query_feat = decoder.encoder.query_feature(feat, point)
            query_feat = decoder.encoder.query_feature(hash_tree, feat, point)
            z.append(decoder.decoder(point, query_feat).detach().squeeze(0).cpu().numpy())
        else:
            z.append(decoder(point.type(torch.float32), mods).detach().squeeze(0).cpu().numpy())
    z = np.concatenate(z, axis=0).reshape(grid_dict['xyz'][1].shape[0], grid_dict['xyz'][0].shape[0],
                                          grid_dict['xyz'][2].shape[0]).transpose([1, 0, 2]).astype(np.float64)

    print(z.min(), z.max())

    threshs = [0.00]
    mesh_list = list()
    for thresh in threshs:
        if (np.sum(z > 0.0) < np.sum(z < 0.0)):
            thresh = -thresh
        verts, faces, normals, values = measure.marching_cubes(volume=z, level=thresh,
                                                               spacing=(cell_width, cell_width, cell_width))

        verts = verts + np.array([grid_dict['xyz'][0][0], grid_dict['xyz'][1][0], grid_dict['xyz'][2][0]])
        verts = verts * (1 / scale) - translate

        if get_mesh:
            mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals, vertex_colors=values, validate=True)
        mesh_list.append(mesh)
    return mesh_list[0]


# @nb.jit()
def surface_extraction_single(ndf, grad, b_max, b_min, resolution):
    '''
    From CAP-UDF (https://github.com/junshengzhou/CAP-UDF)
    '''
    v_all = []
    t_all = []
    threshold = 0.005  # accelerate extraction
    v_num = 0
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            for k in range(resolution - 1):
                ndf_loc = ndf[i:i + 2]
                ndf_loc = ndf_loc[:, j:j + 2, :]
                ndf_loc = ndf_loc[:, :, k:k + 2]
                if np.min(ndf_loc) > threshold:
                    continue
                grad_loc = grad[i:i + 2]
                grad_loc = grad_loc[:, j:j + 2, :]
                grad_loc = grad_loc[:, :, k:k + 2]

                res = np.ones((2, 2, 2))
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            if np.dot(grad_loc[0][0][0], grad_loc[ii][jj][kk]) < 0:
                                res[ii][jj][kk] = -ndf_loc[ii][jj][kk]
                            else:
                                res[ii][jj][kk] = ndf_loc[ii][jj][kk]

                if res.min() < 0:
                    vertices, triangles, _, _ = measure.marching_cubes(
                        res, 0.0)
                    # print(vertices)
                    # vertices -= 1.5
                    # vertices /= 128
                    vertices[:, 0] += i  # / resolution
                    vertices[:, 1] += j  # / resolution
                    vertices[:, 2] += k  # / resolution
                    triangles += v_num
                    # vertices =
                    # vertices[:,1] /= 3  # TODO
                    v_all.append(vertices)
                    t_all.append(triangles)

                    v_num += vertices.shape[0]
                    # print(v_num)

    v_all = np.concatenate(v_all)
    t_all = np.concatenate(t_all)

    mesh = trimesh.Trimesh(v_all, t_all)
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.fill_holes()

    return mesh.vertices, mesh.faces


def inner_iteration(ndf_loc, grad_loc, i, j, k):
    ndf_loc = np.ascontiguousarray(ndf_loc)
    grad_loc = np.ascontiguousarray(grad_loc)
    res = np.ones((2, 2, 2))
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                if np.dot(grad_loc[0][0][0], grad_loc[ii][jj][kk]) < 0:
                    res[ii][jj][kk] = -ndf_loc[ii][jj][kk]
                else:
                    res[ii][jj][kk] = ndf_loc[ii][jj][kk]
    shape_dict = dict()
    if res.min() < 0:
        v, f, _, _ = measure.marching_cubes(res, 0.0)
        v[:, 0] = v[:, 0] + i
        v[:, 1] = v[:, 1] + j
        v[:, 2] = v[:, 2] + k
        shape_dict['v'] = v
        shape_dict['f'] = f
    else:
        shape_dict['v'] = None
        shape_dict['f'] = None

    return shape_dict


import numba

@numba.jit(nopython=True, fastmath=True)
def map(ndf, grad, resolution=128, threshold=0.01):
    call_params = list()
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            for k in range(resolution - 1):
                ndf_loc = ndf[i:i + 2]
                ndf_loc = ndf_loc[:, j:j + 2, :]
                ndf_loc = ndf_loc[:, :, k:k + 2]
                if np.min(ndf_loc) > threshold:
                    continue
                grad_loc = grad[i:i + 2]
                grad_loc = grad_loc[:, j:j + 2, :]
                grad_loc = grad_loc[:, :, k:k + 2]
                call_params.append((ndf_loc, grad_loc, i, j, k))
    return call_params


def reduce(res_dict_list):
    v_all = list()
    f_all = list()
    v_num = 0
    for res in res_dict_list:
        if res['v'] is not None:
            v_all.append(res['v'])
            f_all.append(res['f'] + v_num)
            v_num += res['v'].shape[0]
    v_all = np.concatenate(v_all)
    f_all = np.concatenate(f_all)
    mesh = trimesh.Trimesh(v_all, f_all)
    return mesh


num_processes = 8

from .utils_mp import start_process_pool


def surface_extraction_mp(ndf, grad, resolution):
    '''
    Modified From CAP-UDF (https://github.com/junshengzhou/CAP-UDF)
    '''
    call_params = map(ndf, grad, resolution)
    res_dict_list = start_process_pool(inner_iteration, call_params, num_processes)
    mesh = reduce(res_dict_list)

    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.fill_holes()

    return mesh.vertices, mesh.faces


def udf2mesh(decoder, latent, grid_res, translate=[0., 0., 0.], scale=1, get_mesh=True, device=None,
             bbox=np.array([[-1, 1], [-1, 1], [-1, 1]])):
    # using mc from MeshUDF
    print('in implicit2mesh')
    print(grid_res, translate, scale, bbox)
    mesh = None

    def get_query_point(device, bd=1.05, resolution=128):
        shape = (resolution, resolution, resolution)
        vxs = torch.arange(-bd, bd, bd * 2 / resolution)
        vys = torch.arange(-bd, bd, bd * 2 / resolution)
        vzs = torch.arange(-bd, bd, bd * 2 / resolution)
        pxs = vxs.view(-1, 1, 1).expand(*shape).contiguous().view(resolution ** 3)
        pys = vys.view(1, -1, 1).expand(*shape).contiguous().view(resolution ** 3)
        pzs = vzs.view(1, 1, -1).expand(*shape).contiguous().view(resolution ** 3)
        p = torch.stack([pxs, pys, pzs], dim=1).reshape(resolution ** 3, 3).to(device)
        return p

    pnts = get_query_point(resolution=grid_res, device=device)
    z = []
    n = []
    for point in tqdm(torch.split(pnts, 100000, dim=0)):
        # point: (100000, 3)
        if latent is not None:
            point = torch.cat([point, latent.unsqueeze(0).repeat(point.shape[0], 1), ], dim=1)
        point = get_cuda_ifavailable(point, device=device)
        point.requires_grad_()
        pred = decoder(point.type(torch.float32))
        normals = gradient(point, pred).detach()
        z.append(pred.detach().cpu().float().numpy())
        n.append(normals.detach().cpu().float().numpy())
    z = np.concatenate(z, axis=0).reshape(grid_res, grid_res, grid_res).astype(np.float32)
    n = np.concatenate(n, axis=0).reshape(grid_res, grid_res, grid_res, 3).astype(np.float32)
    print(z.min(), z.max())

    pnts = pnts.detach().cpu().float().numpy().reshape(grid_res, grid_res, grid_res, 3)
    # verts, faces = surface_extraction_single(z, n, resolution=grid_res, b_min=bbox.min(1), b_max=bbox.max(1))
    verts, faces = surface_extraction_mp(z, n, resolution=grid_res)
    # verts, faces = EMC(grids_coords=pnts, grids_udf=z, grids_udf_grad=n,
    #                    voxel_size=2. / grid_res, res=grid_res)

    # # verts = verts + np.array([grid_dict['xyz'][0][0], grid_dict['xyz'][1][0], grid_dict['xyz'][2][0]])
    # # verts = verts * (1 / scale) - translate
    #
    if get_mesh:
        mesh = trimesh.Trimesh(verts, faces, validate=True)
    res_dict = dict()
    res_dict['mesh'] = mesh
    res_dict['z'] = z
    res_dict['n'] = n
    return res_dict


def convert_xyz_to_ply_with_noise(file_path, noise=None):
    # convert ply file in xyznxnynz format to ply file
    points = np.loadtxt(file_path)
    if noise is None:
        mesh = trimesh.Trimesh(points[:, :3], [], vertex_normals=points[:, 3:])
        mesh.export(file_path.split('.')[0] + '.ply', vertex_normal=True)
    else:
        for std in noise:
            bbox_scale = np.abs(points).max()
            var = std * std
            cov_mat = bbox_scale * np.array([[var, 0., 0.], [0., var, 0.], [0., 0., var]])
            noise = np.random.multivariate_normal([0., 0., 0.], cov_mat, size=points.shape[0], check_valid='warn',
                                                  tol=1e-8)
            mesh = trimesh.Trimesh(points[:, :3] + noise, [], vertex_normals=points[:, 3:])
            mesh.export(file_path.split('.')[0] + '_' + str(std) + '.ply', vertex_normal=True)


def count_parameters(model):
    # count the number of parameters in a given model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    file_path = '/home/sitzikbs/Datasets/Reconstruction_IKEA_sample/interior_room.xyz'
    convert_xyz_to_ply_with_noise(file_path, noise=[0.01])
