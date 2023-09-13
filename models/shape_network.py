import torch.nn as nn

from .filmsiren import FilmSiren
from .convolutionalfeature import ConvolutionalFeature


class ShapeNetwork(nn.Module):
    def __init__(self, decoder_hidden_dim=256, decoder_n_hidden_layers=5):
        super().__init__()
        self.encoder = ConvolutionalFeature()
        self.decoder = FilmSiren(hidden_size=decoder_hidden_dim, n_layers=decoder_n_hidden_layers)

    def forward(self, non_mnfld_pnts=None, mnfld_pnts=None, near_points=None):
        return self.forward_dense(non_mnfld_pnts, mnfld_pnts, near_points)

    def forward_dense(self, non_mnfld_pnts=None, mnfld_pnts=None, near_points=None):
        latent_reg = None
        # mnfld
        global_feat = self.encoder.encode(mnfld_pnts)
        mnfld_pnts_feat = self.encoder.query_feature(global_feat, mnfld_pnts)
        manifold_pnts_pred = self.decoder(mnfld_pnts, mnfld_pnts_feat)
        # nonfld
        nonmanifold_pnts_pred = None
        if non_mnfld_pnts is not None:
            non_mnfld_pnts_feat = self.encoder.query_feature(global_feat, non_mnfld_pnts)
            nonmanifold_pnts_pred = self.decoder(non_mnfld_pnts, non_mnfld_pnts_feat)

        near_points_pred = None
        if near_points is not None:
            near_points_feat = self.encoder.query_feature(global_feat, near_points)
            near_points_pred = self.decoder(near_points, near_points_feat)

        return {"manifold_pnts_pred": manifold_pnts_pred,
                "nonmanifold_pnts_pred": nonmanifold_pnts_pred,
                'near_points_pred': near_points_pred,
                "latent_reg": latent_reg,
                }
