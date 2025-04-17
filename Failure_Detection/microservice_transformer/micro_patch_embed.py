import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from microservice_transformer.micro_utils import trunc_normal_

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class EmbeddingStem(nn.Module):
    def __init__(
        self,
        window_length=3,
        feature_num=5,
        microservice_num=15,
        embedding_dim=75,
        position_embedding_dropout=None,
        cls_head=True,
    ):
        super(EmbeddingStem, self).__init__()
        embedding_dim = feature_num * microservice_num
        num_patches = window_length
        if cls_head:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            num_patches += 1

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embedding_dim)
        )
        self.pos_drop = nn.Dropout(p=position_embedding_dropout)
        self.cls_head = cls_head
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(2), -1)
        if self.cls_head:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        return self.pos_drop(x + self.pos_embed)