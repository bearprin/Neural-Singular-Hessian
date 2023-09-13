import os
import threading

if __name__ == '__main__':
    data_path = '../data/sdf/input'

    logdir = './log/sdf'

    # data process
    n_points = 15000  # points number
    n_samples = 10000  # iterations number
    batch_size = 1
    grid_res = 256  # mc resolution
    # network
    layers = 4
    decoder_hidden_dim = 256
    sphere_init_params = (1.6, 0.1)
    init_type = 'siren'  # 'siren' | 'mfgi'
    # loss
    loss_type = 'siren_wo_n_w_morse'
    loss_weights = (7e3, 6e2, 1e2, 5e1, 0, 3)
    morse_type = 'l1'
    morse_decay = 'linear'  # 'linear' | 'quintic' | 'step'
    decay_params = (3, 0.2, 3, 0.4, 0.001, 0.0001)
    # opt
    lr = 5e-5
    grad_clip = 10

    files = list()
    for f in sorted(os.listdir(data_path)):
        if os.path.splitext(f)[1] in ['.xyz', '.ply']:  # only accept .xyz and .ply files
            files.append(f)
    used = list()
    device_ID = [0]  # set used GPU
    i = 0
    while True:
        if len(files) == 0:
            break
        if len(device_ID) != 0 and (
                device_ID[i] not in used or device_ID.count(device_ID[i]) > used.count(device_ID[i])):
            f = files.pop(0)


            def sub_thread():
                id = device_ID[i]
                used.append(id)
                os.system(f'CUDA_VISIBLE_DEVICES={id} python train_surface_reconstruction.py \
                          --logdir {logdir} --data_path {os.path.join(data_path, f)} --n_samples {n_samples} --n_points {n_points} --grid_res {grid_res} \
                          --lr {lr} --grad_clip_norm {grad_clip} \
                          --init_type {init_type} --decoder_hidden_dim {decoder_hidden_dim} --decoder_n_hidden_layers {layers} \
                          --loss_type {loss_type} --loss_weights {loss_weights[0]} {loss_weights[1]} {loss_weights[2]} {loss_weights[3]} {loss_weights[4]} {loss_weights[5]} \
                          --decay_params {decay_params[0]} {decay_params[1]}  {decay_params[2]} {decay_params[3]} {decay_params[4]} {decay_params[5]} --morse_type {morse_type} --morse_decay {morse_decay} --morse_near --output_any'
                          )

                used.remove(id)


            t = threading.Thread(target=sub_thread)
            t.start()
        i = (i + 1) % len(device_ID)
