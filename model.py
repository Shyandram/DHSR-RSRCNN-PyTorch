import math
import torch
from torch import nn
import torch.nn.functional as F
from guided_filter_pytorch.guided_filter import ConvGuidedFilter, GuidedFilter

class SRDHnet(nn.Module):
    def __init__(self, upscale_factor=2, nc=3, n_feature=16):
        super(SRDHnet, self).__init__()
        
        # SRCNN
        #upsampling layer
        upsample_block_num = int(math.log(upscale_factor, 2))
        block = [nn.Upsample(scale_factor=2, mode='bicubic') for _ in range(upsample_block_num)]
        # block = [UpsampleBLock(3, 2) for _ in range(upsample_block_num)]
        self.upsample = nn.Sequential(*block)

        # SR n_feature
        #origi = 64
        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(nc, n_feature, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )
        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(n_feature, n_feature//2, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )
        # Rebuild the layer.
        self.reconstruction = nn.Conv2d(n_feature//2, nc, (5, 5), (1, 1), (2, 2))

        # AOD-net
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=nc, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=nc*2, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=nc*2, out_channels=nc, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=nc*4, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.b = 1

    def forward(self, x):
        # SR
        up_x = self.upsample(x)
        sr_out = self._forward_impl(up_x)
        # DH
        DH_output = self._forward_dehaze(sr_out)
        return DH_output, sr_out 

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out
    def _forward_dehaze(self, x: torch.Tensor) -> torch.Tensor:
        # DH
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = F.relu(self.conv5(cat3))
        if k.size() != x.size():
            raise Exception("k, haze image are different size!")
        output = k * x - k + self.b
        DH_output = F.relu(output)

        return DH_output
    
    # def dehaze_all(self, x):
    #     x1 = F.relu(self.conv1(x))
    #     x2 = F.relu(self.conv2(x1))
    #     cat1 = torch.cat((x1, x2), 1)
    #     x3 = F.relu(self.conv3(cat1))
    #     cat2 = torch.cat((x2, x3), 1)
    #     x4 = F.relu(self.conv4(cat2))
    #     cat3 = torch.cat((x1, x2, x3, x4), 1)
    #     k = F.relu(self.conv5(cat3))

    #     if k.size() != x.size():
    #         raise Exception("k, haze image are different size!")

    #     output = k * x - k + self.b
    #     DH_output = F.relu(output)
    #     up_DH_output = self.upsample(DH_output)
    #     out = self._forward_impl(up_DH_output)
    #     return out, output
class DHSRnet(SRDHnet):
    def forward(self, x):
        # DH
        DH_output = self._forward_dehaze(x)
        # SR
        up_x = self.upsample(DH_output)
        sr_out = self._forward_impl(up_x)
        return DH_output, sr_out 

class DHSRnet_P(SRDHnet):
    def __init__(self, upscale_factor=2, nc=3, n_feature = 16):
        super(DHSRnet_P, self).__init__()
        # CGF
        self.cgf = ConvGuidedFilter()
        
        # SRCNN
        #upsampling layer
        upsample_block_num = int(math.log(upscale_factor, 2))
        block = [nn.Upsample(scale_factor=2, mode='bicubic') for _ in range(upsample_block_num)]
        # block = [UpsampleBLock(3, 2) for _ in range(upsample_block_num)]
        self.upsample = nn.Sequential(*block)

        # SR n_feature
        #origi = 64
        # Feature extraction layer.
        # self.features = nn.Sequential(
        #     nn.Conv2d(nc, n_feature, (9, 9), (1, 1), (4, 4)),
        #     nn.ReLU(True)
        # )
        # # Non-linear mapping layer.
        # self.map = nn.Sequential(
        #     nn.Conv2d(n_feature, n_feature//2, (5, 5), (1, 1), (2, 2)),
        #     nn.ReLU(True)
        # )
        # # Rebuild the layer.
        # self.reconstruction = nn.Conv2d(n_feature//2, nc, (5, 5), (1, 1), (2, 2))

        # AOD-net
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=nc, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=nc*2, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=nc*2, out_channels=nc, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=nc*4, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.b = 1


    def forward(self, x):
        # DH
        DH_output = self._forward_dehaze(x)
        # SR
        up_x = self.upsample(DH_output)
        # CGF
        sr_out = self.cgf(x, DH_output, up_x)
        # sr_out = self._forward_impl(sr_out)
        
        return DH_output, sr_out 
class DHSRnet_griddhnet(nn.Module):
    def __init__(self, upscale_factor=2, nc=3, n_feature=16):
        super(DHSRnet_griddhnet, self).__init__()
        
        # SRCNN
        #upsampling layer
        upsample_block_num = int(math.log(upscale_factor, 2))
        block = [nn.Upsample(scale_factor=2, mode='bicubic') for _ in range(upsample_block_num)]
        # block = [UpsampleBLock(3, 2) for _ in range(upsample_block_num)]
        self.upsample = nn.Sequential(*block)

        # SR n_feature
        #origi = 64
        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(nc, n_feature, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )
        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(n_feature, n_feature//2, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )
        # Rebuild the layer.
        self.reconstruction = nn.Conv2d(n_feature//2, nc, (5, 5), (1, 1), (2, 2))

        # dehaze
        self.dehaze = GridDehazeNet()

    def forward(self, x):
        # DH
        DH_output = self._forward_dehaze(x)
        # SR
        up_x = self.upsample(DH_output)
        sr_out = self._forward_impl(up_x)
        return DH_output, sr_out

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out
    def _forward_dehaze(self, x: torch.Tensor) -> torch.Tensor:
        # DH
        DH_output = self.dehaze(x)

        return DH_output
    
class AODnet(nn.Module):
    def __init__(self):
        super(AODnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.b = 1

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = F.relu(self.conv5(cat3))

        if k.size() != x.size():
            raise Exception("k, haze image are different size!")

        output = k * x - k + self.b
        return F.relu(output)

class SRCNN(nn.Module):
    def __init__(self) -> None:
        super(SRCNN, self).__init__()
        # no upsampling (bicubic)
        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )

        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )

        # Rebuild the layer.
        self.reconstruction = nn.Conv2d(32, 1, (5, 5), (1, 1), (2, 2))

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: model.py
about: model for GridDehazeNet
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --- Downsampling block in GridDehazeNet  --- #
class DownSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels, stride*in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out


# --- Upsampling block in GridDehazeNet  --- #
class UpSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride=stride, padding=1)
        self.conv = nn.Conv2d(in_channels, in_channels // stride, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x, output_size):
        out = F.relu(self.deconv(x, output_size=output_size))
        out = F.relu(self.conv(out))
        return out


# --- Main model  --- #
class GridDehazeNet(nn.Module):
    def __init__(self, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(GridDehazeNet, self).__init__()
        self.rdb_module = nn.ModuleDict()
        self.upsample_module = nn.ModuleDict()
        self.downsample_module = nn.ModuleDict()
        self.height = height
        self.width = width
        self.stride = stride
        self.depth_rate = depth_rate
        self.coefficient = nn.Parameter(torch.Tensor(np.ones((height, width, 2, depth_rate*stride**(height-1)))), requires_grad=attention)
        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.rdb_in = RDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb_out = RDB(depth_rate, num_dense_layer, growth_rate)

        rdb_in_channels = depth_rate
        for i in range(height):
            for j in range(width - 1):
                self.rdb_module.update({'{}_{}'.format(i, j): RDB(rdb_in_channels, num_dense_layer, growth_rate)})
            rdb_in_channels *= stride

        _in_channels = depth_rate
        for i in range(height - 1):
            for j in range(width // 2):
                self.downsample_module.update({'{}_{}'.format(i, j): DownSample(_in_channels)})
            _in_channels *= stride

        for i in range(height - 2, -1, -1):
            for j in range(width // 2, width):
                self.upsample_module.update({'{}_{}'.format(i, j): UpSample(_in_channels)})
            _in_channels //= stride

    def forward(self, x):
        inp = self.conv_in(x)

        x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
        i, j = 0, 0

        x_index[0][0] = self.rdb_in(inp)

        for j in range(1, self.width // 2):
            x_index[0][j] = self.rdb_module['{}_{}'.format(0, j-1)](x_index[0][j-1])

        for i in range(1, self.height):
            x_index[i][0] = self.downsample_module['{}_{}'.format(i-1, 0)](x_index[i-1][0])

        for i in range(1, self.height):
            for j in range(1, self.width // 2):
                channel_num = int(2**(i-1)*self.stride*self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.downsample_module['{}_{}'.format(i-1, j)](x_index[i-1][j])

        x_index[i][j+1] = self.rdb_module['{}_{}'.format(i, j)](x_index[i][j])
        k = j

        for j in range(self.width // 2 + 1, self.width):
            x_index[i][j] = self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1])

        for i in range(self.height - 2, -1, -1):
            channel_num = int(2 ** (i-1) * self.stride * self.depth_rate)
            x_index[i][k+1] = self.coefficient[i, k+1, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, k)](x_index[i][k]) + \
                              self.coefficient[i, k+1, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, k+1)](x_index[i+1][k+1], x_index[i][k].size())

        for i in range(self.height - 2, -1, -1):
            for j in range(self.width // 2 + 1, self.width):
                channel_num = int(2 ** (i - 1) * self.stride * self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, j)](x_index[i+1][j], x_index[i][j-1].size())

        out = self.rdb_out(x_index[i][j])
        out = F.relu(self.conv_out(out))

        return out
"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: residual_dense_block.py
about: build the Residual Dense Block
author: Xiaohong Liu
date: 01/08/19
"""
# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """
        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out