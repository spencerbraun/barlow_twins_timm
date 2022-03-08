import torch
import torch.nn as nn

import timm


class Encoder(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=True):
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.fc = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        return x


class Projector(nn.Module):
    def __init__(self, in_features, out_features, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_features, out_features, bias=False),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
            ]
            * (n_layers - 1)
        )
        self.layers.append(nn.Linear(out_features, out_features, bias=False))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BarlowTwins(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.model, config.pretrained)
        self.projector = Projector(2048, config.proj_dim, n_layers=config.proj_layers)
        self.normalize = nn.BatchNorm1d(config.proj_dim, affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.encoder(y1))
        z2 = self.projector(self.encoder(y2))

        # empirical cross-correlation matrix
        c = self.normalize(z1).T @ self.normalize(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.config.batch_size).sum()
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

        diag_mask = (torch.ones(c.shape[0], c.shape[0]) - torch.eye(c.shape[0])).to(
            c.device
        )
        off_diag = c.mul_(diag_mask).pow_(2).sum()
        loss = on_diag + self.config.lambda_ * off_diag

        return loss
