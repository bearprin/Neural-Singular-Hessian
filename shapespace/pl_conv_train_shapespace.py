import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy

from shapespace.dfaust_dataset import DFaustDataSet
from models import ShapeNetwork, MorseLoss
import utils.utils as utils
import shapespace.shapespace_dfaust_args as shapespace_dfaust_args

# get training parameters
args = shapespace_dfaust_args.get_args()
logdir = os.path.join(args.logdir)
os.makedirs(logdir, exist_ok=True)
log_file, log_writer_train, log_writer_test, model_outdir = utils.setup_logdir(logdir, args)

os.makedirs(os.path.join(logdir, "trained_models"), exist_ok=True)
project_dir = os.path.join(logdir, "trained_models")

os.system('cp %s %s' % (__file__, logdir))  # backup the current training file
os.system('cp %s %s' % ('../models/new_network.py', logdir))  # backup the models files
os.system('cp %s %s' % ('../models/convolutionalfeature.py', logdir))  # backup the models files
os.system('cp %s %s' % ('../models/losses.py', logdir))  # backup the models files


class DFaustDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        self.train_set = DFaustDataSet(self.args.dataset_path, self.args.train_split_path,
                                       with_normals=args.with_normal)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.args.batch_size, num_workers=32, persistent_workers=True,
                          pin_memory=True)


class BaseTrainer(pl.LightningModule):
    def __init__(self, args):
        super(BaseTrainer, self).__init__()
        self.args = args
        self.learning_rate = args.lr
        self.net = ShapeNetwork(decoder_hidden_dim=args.decoder_hidden_dim,
                              decoder_n_hidden_layers=args.decoder_n_layers)

        self.criterion = MorseLoss(weights=args.loss_weights, loss_type=args.loss_type, div_decay=args.morse_decay,
                                   div_type=args.morse_type, bidirectional_morse=args.bidirectional_morse)

    def training_step(self, batch, batch_idx):
        self.net.train()
        self.net.zero_grad(set_to_none=True)
        mnfld_points, mnfld_n_gt, nonmnfld_points, near_points, indices = batch['mnfld_points'], batch[
            'mnfld_n'], batch['nonmnfld_points'], batch['near_points'], batch['indices']
        mnfld_points.requires_grad_()
        nonmnfld_points.requires_grad_()
        near_points.requires_grad_()

        output_pred = self.net(nonmnfld_points, mnfld_points, near_points=near_points if self.args.morse_near else None)

        loss_dict, _ = self.criterion(output_pred, mnfld_points, nonmnfld_points,
                                      near_points=near_points if args.morse_near else None)
        for key, value in loss_dict.items():
            self.log(key, value, on_step=True, logger=True)
        self.log('total_loss', loss_dict['loss'], on_step=True, logger=True, prog_bar=True)
        if self.local_rank == 0 and self.global_rank == 0 and batch_idx % 30 == 0:
            weights = self.criterion.weights
            utils.log_string("Weights: {}".format(weights), log_file)
            utils.log_string('Epoch: {}, Loss: {:.5f} = L_Mnfld: {:.5f} + '
                             'L_NonMnfld: {:.5f} + L_Nrml: {:.5f} + L_Eknl: {:.5f} + L_Div: {:.5f} + L_Morse: {:.5f} + L_Latent: {:.10f}'.format(
                self.current_epoch, loss_dict["loss"].item(), weights[0] * loss_dict["sdf_term"].item(),
                                                              weights[1] * loss_dict["inter_term"].item(),
                                                              weights[2] * loss_dict["normals_loss"].item(),
                                                              weights[3] * loss_dict["eikonal_term"].item(),
                                                              weights[4] * loss_dict["div_loss"].item(),
                                                              weights[5] * loss_dict['morse_term'].item(),
                                                              weights[6] * loss_dict['latent_reg_term'].item()),
                log_file)
            utils.log_string('Unweighted L_s : L_Mnfld: {:.5f},  '
                             'L_NonMnfld: {:.5f},  L_Nrml: {:.5f},  L_Eknl: {:.5f}, L_Morse: {:.5f}, L_Latent: {:.10f}'.format(
                loss_dict["sdf_term"].item(), loss_dict["inter_term"].item(),
                loss_dict["normals_loss"].item(), loss_dict["eikonal_term"].item(),
                loss_dict['morse_term'].item(), loss_dict['latent_reg_term'].item()),
                log_file)
        return {'loss': loss_dict['loss'], 'mnfld': mnfld_points[:1]}

    def training_epoch_end(self, outputs):
        mnfld = outputs[0]['mnfld']
        self.net.eval()
        if self.global_rank == 0 and self.local_rank == 0:
            with torch.no_grad():
                t0 = time.time()
                out_dir = "{}/vis_results/".format(args.logdir)
                os.makedirs(out_dir, exist_ok=True)
                global_feat = self.net.encoder.encode(mnfld)
                try:
                    pred_mesh = utils.implicit2mesh(decoder=self.net, mods=None, feat=global_feat,
                                                    grid_res=128,
                                                    get_mesh=True, device=next(self.net.parameters()).device)
                    pred_mesh.export(os.path.join(out_dir, "pred_mesh_{}.ply".format(self.current_epoch)))
                except Exception as e:
                    print('Can not plot')
                    print(e)
                print('Plot took {:.3f}s'.format(time.time() - t0))
        # update weights
        curr_epoch = self.current_epoch
        self.criterion.update_morse_weight(curr_epoch, self.args.num_epochs, self.args.decay_params)

    def configure_optimizers(self):
        # Setup Adam optimizers
        optimizer = torch.optim.Adam(self.trainer.model.parameters(), lr=self.learning_rate, amsgrad=True)
        lr_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150 * 10, T_mult=2, eta_min=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_sch,
                "interval": "step",
                "frequency": 1,
            },
        }


check_callback = ModelCheckpoint(
    dirpath=project_dir,
    filename='model-{epoch:02d}',
    save_top_k=-1,
    save_last=True
)
lr_monitor = LearningRateMonitor(logging_interval='step')
pl.seed_everything(args.seed, workers=True)
trainer = pl.Trainer(gradient_clip_val=args.grad_clip_norm,
                     max_epochs=args.num_epochs,
                     auto_select_gpus=True,
                     accelerator='gpu',
                     strategy=DDPStrategy(find_unused_parameters=False),
                     devices=8,
                     callbacks=[check_callback, TQDMProgressBar(refresh_rate=10), lr_monitor],
                     accumulate_grad_batches=8,
                     benchmark=True,
                     deterministic=False,
                     logger=True,
                     )
base_trainer = BaseTrainer(args)
dm = DFaustDataModule(args)

trainer.fit(base_trainer, dm, ckpt_path=os.path.join(project_dir, 'last.ckpt') if os.path.exists(
    os.path.join(project_dir, 'last.ckpt')) else None)
