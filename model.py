# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
import os
from typing import Any, cast, Dict, List, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

__all__ = [
    "DiscriminatorForVGG", "SRResNet",
    "discriminator_for_vgg", "srresnet_x2", "srresnet_x4", "srresnet_x8",
]

feature_extractor_net_cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_layers(net_cfg_name: str, batch_norm: bool = False) -> nn.Sequential:
    net_cfg = feature_extractor_net_cfgs[net_cfg_name]
    layers: nn.Sequential[nn.Module] = nn.Sequential()
    in_channels = 3
    for v in net_cfg:
        if v == "M":
            layers.append(nn.MaxPool2d((2, 2), (2, 2)))
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1))
            # 已删除了BN层
            layers.append(conv2d)   
            layers.append(nn.ReLU(True))
            in_channels = v

    return layers

#----------------------------------------------------------------------------
# 深度可分离卷积 (Depthwise Separable Convolution)。这是“沙漏结构”的基础，用于减少参数量
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                                   padding, groups=in_channels, bias=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# 红外线性灰度变换 (LGT)。这个模块用于调整特征图的对比度和亮度。k 和 b 是可学习的参数。
import torch
import torch.nn as nn

class LinearGrayTransform(nn.Module):
    def __init__(self, in_channels):
        super(LinearGrayTransform, self).__init__()
        # 初始化可学习的对比度(k)和亮度(b)参数
        # 形状为 (1, C, 1, 1) 以便广播到 (B, C, H, W) 的特征图
        self.k = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        
    def forward(self, x):
        # 应用线性变换 y = k*x + b
        return self.k * x + self.b
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# 轻型注意力残差块 (LARB)。这是最重要的核心模块，它将取代原有的 ResidualBlock
import torch.nn as nn

class LightweightAttentionResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(LightweightAttentionResidualBlock, self).__init__()

        # 主路径：沙漏结构
        # 使用深度可分离卷积来降低参数
        self.main_path = nn.Sequential(
            DepthwiseSeparableConv(in_features, in_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(in_features, in_features, kernel_size=3, padding=1)
        )

        # 注意力路径 (在跳跃连接上)
        self.attention_path = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 论文描述比较模糊，这里提供一个最合理的实现：
        # 注意力图作用于主路径提取的特征，然后与输入进行残差连接
        main_out = self.main_path(x)
        attention_map = self.attention_path(x)
        
        # 残差连接
        # output = input + attention * features
        out = x + main_out * attention_map
        return out
#----------------------------------------------------------------------------



class _FeatureExtractor(nn.Module):
    def __init__(
            self,
            net_cfg_name: str = "vgg19",
            batch_norm: bool = False,
            num_classes: int = 1000) -> None:
        super(_FeatureExtractor, self).__init__()
        self.features = _make_layers(net_cfg_name, batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        # Initialize neural network weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            # 已删除BN层
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class SRResNet(nn.Module): # 生成器，相当于Generator
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            num_rcb: int = 14, # 源代码为16，按照论文改成14
            upscale: int = 4,
    ) -> None:
        super(SRResNet, self).__init__()
        # Low frequency information extraction layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # 修改点 3: 重建 self.trunk 序列
        # -----------------------------------------------------------------
        # 原来的代码:
        # trunk = []
        # for _ in range(num_rcb):
        # 	  trunk.append(_ResidualConvBlock(channels))
        # self.trunk = nn.Sequential(*trunk)
        #
        # 替换为下面的新逻辑:
        trunk = []
        for i in range(num_rcb):
            # 每次循环都添加一个LARB
            trunk.append(LightweightAttentionResidualBlock(in_features=channels))
            # 在第7个和第14个LARB后添加LGT模块
            if i + 1 == 7 or i + 1 == 14:
                trunk.append(LinearGrayTransform(in_channels=channels))
        self.trunk = nn.Sequential(*trunk)
        # -----------------------------------------------------------------

        # High-frequency information linear fusion layer 高频信息线性融合层
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=True), # bias设置成True,因为没有BN层了
            # 已删除BN层
        )

        # zoom block 上采样模块
        upsampling = []
        if upscale == 2 or upscale == 4 or upscale == 8:
            for _ in range(int(math.log(upscale, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        else:
            raise NotImplementedError(f"Upscale factor `{upscale}` is not support.")
        self.upsampling = nn.Sequential(*upsampling)

        # reconstruction block
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))

        # Initialize neural network weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            # 已删除BN层

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1) # 全局残差连接
        x = self.upsampling(x)
        x = self.conv3(x)

        x = torch.clamp_(x, 0.0, 1.0)

        return x


class DiscriminatorForVGG(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            channels: int = 64,
    ) -> None:
        super(DiscriminatorForVGG, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 96 x 96
            nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 48 x 48
            nn.Conv2d(channels, channels, (3, 3), (2, 2), (1, 1), bias=True), # bias改为true
            # 已删除BN层
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, int(2 * channels), (3, 3), (1, 1), (1, 1), bias=True),
            # 已删除BN层
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 24 x 24
            nn.Conv2d(int(2 * channels), int(2 * channels), (3, 3), (2, 2), (1, 1), bias=True),
            # 已删除BN层
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(2 * channels), int(4 * channels), (3, 3), (1, 1), (1, 1), bias=True),
            # 已删除BN层
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 12 x 12
            nn.Conv2d(int(4 * channels), int(4 * channels), (3, 3), (2, 2), (1, 1), bias=True),
            # 已删除BN层
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(4 * channels), int(8 * channels), (3, 3), (1, 1), (1, 1), bias=True),
            # 已删除BN层
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 6 x 6
            nn.Conv2d(int(8 * channels), int(8 * channels), (3, 3), (2, 2), (1, 1), bias=True),
            # 已删除BN层
            nn.LeakyReLU(0.2, True),
        )




# --------------------【修改点 1】--------------------
        # 计算分类器的输入维度： channels * H * W
        # 原始 96x96 -> 6x6,  现在 256x256 -> 16x16
        # 8 * channels = 512
        classifier_input_features = int(8 * channels) * 16 * 16 
        # ---------------------------------------------------
        
        
        self.classifier = nn.Sequential(
            # --------------------【修改点 2】--------------------
            nn.Linear(classifier_input_features, 1024), # 使用新的计算结果
            # ---------------------------------------------------
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        # --------------------【修改点 3】--------------------
        # Input image size must equal 256 (根据论文的 crop_size)
        assert x.size(2) == 256 and x.size(3) == 256, f"Input image size must be 256x256, but got {x.size()}"
        # ---------------------------------------------------
        
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=True),
            # 已删除BN层
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=True),
            # 已删除BN层
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = self.rcb(x)

        x = torch.add(x, identity)

        return x


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample_block(x)

        return x


class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(
            self,
            net_cfg_name: str,
            batch_norm: bool,
            num_classes: int,
            model_weights_path: str,
            feature_nodes: list,
            feature_normalize_mean: list,
            feature_normalize_std: list,
    ) -> None:
        super(ContentLoss, self).__init__()
        # Define the feature extraction model
        model = _FeatureExtractor(net_cfg_name, batch_norm, num_classes)
        # Load the pre-trained model
        if model_weights_path == "":
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        elif model_weights_path is not None and os.path.exists(model_weights_path):
            checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
            if "state_dict" in checkpoint.keys():
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError("Model weight file not found")
        # Extract the output of the feature extraction layer
        self.feature_extractor = create_feature_extractor(model, feature_nodes)
        # Select the specified layers as the feature extraction layer
        self.feature_extractor_nodes = feature_nodes
        # input normalization
        self.normalize = transforms.Normalize(feature_normalize_mean, feature_normalize_std)
        # Freeze model parameters without derivatives
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False
        self.feature_extractor.eval()

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> [Tensor]:
        assert sr_tensor.size() == gt_tensor.size(), "Two tensor must have the same size"
        device = sr_tensor.device

        losses = []
        # input normalization
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        # Get the output of the feature extraction layer
        sr_feature = self.feature_extractor(sr_tensor)
        gt_feature = self.feature_extractor(gt_tensor)

        # Compute feature loss
        for i in range(len(self.feature_extractor_nodes)):
            losses.append(F_torch.mse_loss(sr_feature[self.feature_extractor_nodes[i]],
                                           gt_feature[self.feature_extractor_nodes[i]]))

        losses = torch.Tensor([losses]).to(device)

        return losses


def srresnet_x2(**kwargs: Any) -> SRResNet:
    model = SRResNet(upscale=2, **kwargs)

    return model


def srresnet_x4(**kwargs: Any) -> SRResNet:
    model = SRResNet(upscale=4, **kwargs)

    return model


def srresnet_x8(**kwargs: Any) -> SRResNet:
    model = SRResNet(upscale=8, **kwargs)

    return model


def discriminator_for_vgg(**kwargs) -> DiscriminatorForVGG:
    model = DiscriminatorForVGG(**kwargs)

    return model
