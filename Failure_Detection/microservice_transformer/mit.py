import torch.nn as nn

from microservice_transformer.micro_patch_embed import EmbeddingStem
from microservice_transformer.micro_transformer import Transformer
from microservice_transformer.micro_modules import OutputLayer


class MicroserviceTransformer(nn.Module):
    def __init__(
        self,
        window_length=3,
        feature_num=5,
        microservice_num=15,
        embedding_dim=75,
        num_layers=4,
        num_heads=5,
        qkv_bias=True,
        mlp_ratio=4.0,
        use_revised_ffn=False,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        cls_head=False,
    ):


        super(MicroserviceTransformer, self).__init__()

        # embedding layer
        self.embedding_layer = EmbeddingStem(
            window_length=window_length,
            feature_num=feature_num,
            microservice_num=microservice_num,
            embedding_dim=embedding_dim,
            position_embedding_dropout=dropout_rate,
            cls_head=cls_head,
        )

        # transformer
        self.transformer = Transformer(
            dim=embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout_rate,
            dropout=dropout_rate,
            qkv_bias=qkv_bias,
            revised=use_revised_ffn,
        )
        self.post_transformer_ln = nn.LayerNorm(embedding_dim)  #normalization layer

        # output layer
        self.cls_layer = OutputLayer(
            cls_head=cls_head,
        )

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformer(x)
        x = self.post_transformer_ln(x)
        x = self.cls_layer(x)
        return x