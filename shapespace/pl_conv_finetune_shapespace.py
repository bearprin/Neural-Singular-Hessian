import os
import sys
import time

import numpy as np
import tqdm

# import pymesh

# import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from shapespace.dfaust_dataset import DFaustDataSet
from models import NewNetwork, MorseLoss
import utils.utils as utils
import shapespace.shapespace_dfaust_args as shapespace_dfaust_args
from shapespace.shapespace_utils import mkdir_ifnotexists

# get training parameters
args = shapespace_dfaust_args.get_args()
logdir = os.path.join(args.logdir)
os.makedirs(logdir, exist_ok=True)
log_file, log_writer_train, log_writer_test, model_outdir = utils.setup_logdir(logdir, args)

mkdir_ifnotexists(os.path.join(logdir, "trained_models"))
project_dir = os.path.join(logdir, "trained_models")

os.system('cp %s %s' % (__file__, logdir))  # backup the current training file
os.system('cp %s %s' % ('../models/DiGS.py', logdir))  # backup the models files
os.system('cp %s %s' % ('../models/losses.py', logdir))  # backup the models files


class DFaustDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        self.train_set = DFaustDataSet(self.args.dataset_path, self.args.test_split_path,
                                       # gt_path=self.args.gt_path,
                                       # scan_path=self.args.scan_path, \
                                       with_normals=False, part_i=args.test_part_i)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.args.batch_size, num_workers=32, persistent_workers=True,
                          shuffle=False,
                          pin_memory=True)


class BaseTrainer(pl.LightningModule):
    def __init__(self, args):
        super(BaseTrainer, self).__init__()
        self.args = args
        self.learning_rate = args.test_lr
        self.net = NewNetwork(decoder_hidden_dim=args.decoder_hidden_dim,
                              decoder_n_hidden_layers=args.decoder_n_hidden_layers)
        self.criterion = MorseLoss(weights=args.test_loss_weights, loss_type=args.test_loss_type,
                                   div_decay=args.morse_decay,
                                   div_type=args.morse_type, bidirectional_morse=args.bidirectional_morse)

        self.args.morse_near = False
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        self.net.train()
        mnfld_points, _, nonmnfld_points, near_points, indices, name = batch['mnfld_points'], batch[
            'mnfld_n'], batch['nonmnfld_points'], batch['near_points'], batch['indices'], batch['name']
        # load pretrained model
        print('load pretrained model')
        self.load_state_dict(
            torch.load(os.path.join(project_dir, 'last.ckpt'), map_location=mnfld_points.device)['state_dict'])
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, amsgrad=True)
        optimizer.load_state_dict(torch.load(os.path.join(project_dir, 'last.ckpt'))['optimizer_states'][0])
        # save the pre_optim state
        self.generate_mesh(mnfld_points[:1], name, dir_name='pre_optim')
        mnfld_points.requires_grad_()
        nonmnfld_points.requires_grad_()
        near_points.requires_grad_()
        for i in tqdm.tqdm(range(args.test_iter)):
            # for i in tqdm.tqdm(range(50)):
            self.net.zero_grad()
            # decay lr
            if i == args.test_iter // 2:
                for g in optimizer.param_groups:
                    g['lr'] = self.learning_rate / 10
            output_pred = self.net(nonmnfld_points, mnfld_points,
                                   near_points=near_points if self.args.morse_near else None)

            loss_dict, _ = self.criterion(output_pred, mnfld_points, nonmnfld_points,
                                          near_points=near_points if args.morse_near else None)
            loss_dict["loss"].backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), args.grad_clip_norm)
            optimizer.step()
            # self.optimizers().step()
            # if self.local_rank == 0 and self.global_rank == 0 and i % 50 == 0:
            #     weights = self.criterion.weights
            #     utils.log_string("Weights: {}".format(weights), log_file)
            #     utils.log_string('Epoch: {}, Loss: {:.5f} = L_Mnfld: {:.5f} + '
            #                      'L_NonMnfld: {:.5f} + L_Nrml: {:.10f} + L_Eknl: {:.5f} + L_Div: {:.5f} + L_Morse: {:.5f} + L_Latent: {:.10f}'.format(
            #         self.current_epoch, loss_dict["loss"].item(), weights[0] * loss_dict["sdf_term"].item(),
            #                                                       weights[1] * loss_dict["inter_term"].item(),
            #                                                       weights[2] * loss_dict["normals_loss"].item(),
            #                                                       weights[3] * loss_dict["eikonal_term"].item(),
            #                                                       weights[4] * loss_dict["div_loss"].item(),
            #                                                       weights[5] * loss_dict['morse_term'].item(),
            #                                                       weights[6] * loss_dict['latent_reg_term'].item()),
            #         log_file)
            #     utils.log_string('Unweighted L_s : L_Mnfld: {:.5f},  '
            #                      'L_NonMnfld: {:.5f},  L_Nrml: {:.10f},  L_Eknl: {:.5f}, L_Morse: {:.5f}, L_Latent: {:.10f}'.format(
            #         loss_dict["sdf_term"].item(), loss_dict["inter_term"].item(),
            #         loss_dict["normals_loss"].item(), loss_dict["eikonal_term"].item(),
            #         loss_dict['morse_term'].item(), loss_dict['latent_reg_term'].item()),
            #         log_file)
            #     self.generate_mesh(mnfld_points[:1], name, dir_name='iter', i=i)
        return {'mnfld': mnfld_points[:1], 'name': name}

    def generate_mesh(self, mnfld, name, dir_name='', i=None):
        cat = '/'.join(name[0].split('/')[-3:-1])
        name = os.path.splitext(name[0].split('/')[-1])[0]
        self.net.eval()
        if self.global_rank == 0 and self.local_rank == 0:
            with torch.no_grad():
                t0 = time.time()
                out_dir = "{}/vis_results_test_{}/{}".format(args.logdir, dir_name, cat)
                out_dir_pts = "{}/vis_results_test_pts_{}/{}".format(args.logdir, dir_name, cat)
                os.makedirs(out_dir, exist_ok=True)
                os.makedirs(out_dir_pts, exist_ok=True)
                global_feat = self.net.encoder.encode(mnfld)
                try:
                    pred_mesh = utils.implicit2mesh(decoder=self.net, mods=None, feat=global_feat,
                                                    grid_res=512,
                                                    get_mesh=True, device=next(self.net.parameters()).device)
                    if i is None:
                        pred_mesh.export(os.path.join(out_dir, "{}.ply".format(name)))
                    else:
                        pred_mesh.export(os.path.join(out_dir, "{}_iter_{}.ply".format(name, i)))
                    np.savetxt('{}/{}.xyz'.format(out_dir_pts, name), mnfld[0].cpu().numpy())
                except Exception as e:
                    print('Can not plot')
                    print(e)
                print('Plot took {:.3f}s'.format(time.time() - t0))

    def training_step_end(self, step_output):
        mnfld = step_output['mnfld']
        name = step_output['name']
        self.generate_mesh(mnfld, name, dir_name='res')

    def configure_optimizers(self):
        # Setup Adam optimizers
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, amsgrad=True)
        return optimizer


lr_monitor = LearningRateMonitor(logging_interval='step')
pl.seed_everything(args.seed, workers=True)
trainer = pl.Trainer(
    max_epochs=180,
    # auto_select_gpus=True,
    accelerator='gpu',
    devices=1,
    callbacks=[TQDMProgressBar(refresh_rate=10), lr_monitor],
    benchmark=True,
    deterministic=False,
    logger=True,
)
base_trainer = BaseTrainer(args)
dm = DFaustDataModule(args)
trainer.fit(base_trainer, dm, ckpt_path=os.path.join(project_dir, 'last.ckpt') if os.path.exists(
    os.path.join(project_dir, 'last.ckpt')) else None)
