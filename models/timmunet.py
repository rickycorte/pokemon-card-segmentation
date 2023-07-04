import torch
import timm
from torchvision import transforms

from models.baseline import UConvGroup

class UnetTimm(torch.nn.Module):
    def __init__(self, out_depth:int, backbone_name="efficientnet_b0", pretrained=True, decoder_scale = 1):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,
            pretrained=pretrained
        )

        self.upconvs = []

        # get channels of backbone layers in inverted order (lower -> upper)
        bb_channels = self.backbone.feature_info.channels()[::-1]
        bb_channels.append(bb_channels[-1])

        for i in range(len(bb_channels)-1):
            if i == 0:
                layer = UConvGroup(bb_channels[i], decoder_scale * bb_channels[i+1])
            else:
               layer = UConvGroup((decoder_scale + 1) * bb_channels[i], decoder_scale * bb_channels[i+1])

            self.upconvs.append(layer)

        self.upconvs = torch.nn.ModuleList(self.upconvs)

        self.normalize = transforms.Normalize(
            mean=self.backbone.pretrained_cfg["mean"],
            std=self.backbone.pretrained_cfg["std"],
        )

        self.out_conv = torch.nn.Conv2d(decoder_scale * bb_channels[-1], out_depth, kernel_size=3, padding=1)



    def forward(self, x):
        #x = self.normalize(x)
        features = self.backbone(x)[::-1]

        for i, f in enumerate(features):
            if i == 0:
                void_shape = list(f.shape)
                void_shape[1] = 0
                p = self.upconvs[0](torch.empty(void_shape).to(x.device), f)
            else:
                p = self.upconvs[i](p, f)

            #print(f"{i}: {x.shape}")

        return self.out_conv(p)
