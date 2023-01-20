import math
import torch
from torch import nn
import torch.nn.functional as F
from guided_filter_pytorch.guided_filter import ConvGuidedFilter, GuidedFilter

class SRDHnet(nn.Module):
    def __init__(self, upscale_factor=2, nc=3):
        super(SRDHnet, self).__init__()
        
        # SRCNN
        #upsampling layer
        upsample_block_num = int(math.log(upscale_factor, 2))
        block = [nn.Upsample(scale_factor=2, mode='bicubic') for _ in range(upsample_block_num)]
        # block = [UpsampleBLock(3, 2) for _ in range(upsample_block_num)]
        self.upsample = nn.Sequential(*block)

        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )
        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )
        # Rebuild the layer.
        self.reconstruction = nn.Conv2d(32, 3, (5, 5), (1, 1), (2, 2))

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
        # DH
        DH_output = self._forward_dehaze(x)
        # SR
        up_x = self.upsample(DH_output)
        # CGF
        sr_out = self.cgf(x, DH_output, up_x)

        sr_out = self._forward_impl(sr_out)
        return DH_output, sr_out 
 
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