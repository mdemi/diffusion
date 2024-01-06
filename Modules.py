import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim, max_len=10000):
        super().__init__()
        self.dim = dim
        self.max_len = torch.Tensor([max_len])

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        exponent = -torch.log(self.max_len) / (half_dim - 1)
        angle = time[:, None] * torch.exp(torch.arange(half_dim, device=device) * exponent.to(device))[None, :]
        embedding = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
        return embedding
    
class TimeEmbedding(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            max_len=10000
            ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.sin_emb = SinusoidalEmbedding(dim=self.in_ch, max_len=max_len)
        self.dense1 = nn.Linear(self.in_ch, self.in_ch)
        self.dense2 = nn.Linear(self.in_ch, self.out_ch)

    def forward(self, x):
        y = self.sin_emb(x)
        y = self.dense1(y)
        y = F.layer_norm(y, y.shape)
        y = F.gelu(y)
        y = self.dense2(y)
        return y


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            mid_ch=None,
            dropout=0.0,
            num_groups=16,
            residual=False
    ):
        super().__init__()
        if mid_ch is None:
            self.mid_ch = out_ch
        else:
            self.mid_ch = mid_ch
        self.residual = residual
        self.conv1 = nn.Conv2d(in_ch, self.mid_ch, kernel_size=3, padding=1)
        self.groupnorm1 = nn.GroupNorm(num_groups, self.mid_ch)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.mid_ch, out_ch, kernel_size=3, padding=1)
        self.groupnorm2 = nn.GroupNorm(num_groups, out_ch)

    def forward(self, x):
        y = self.conv1(x)
        y = self.groupnorm1(y)
        y = F.gelu(y)
        y = self.dropout1(y)
        y = self.conv2(y)
        y = self.groupnorm2(y)
        y = F.gelu(y)
        if self.residual:
            return x + y
        else:
            return y

class DownSample(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            num_groups=16,
    ):
        super().__init__()
        self.convblock1 = ConvBlock(in_ch, in_ch//2, num_groups=num_groups, residual=False)
        self.convblock2 = ConvBlock(in_ch//2, out_ch, num_groups=num_groups, residual=False)
        self.time_emb = TimeEmbedding(out_ch, out_ch)
    
    def forward(self, x, t):
        y = F.max_pool2d(x, 2)
        y = self.convblock1(y)
        y = self.convblock2(y)

        emb = self.time_emb(t)[:, :, None, None].repeat(1, 1, y.shape[-2], y.shape[-1])
        return y + emb
    
class UpSample(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            num_groups=16,
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_ch, in_ch//2, kernel_size=3, padding="same")
        self.groupnorm1 = nn.GroupNorm(num_groups, in_ch//2)
        self.convblock1 = ConvBlock(in_ch, in_ch//2, num_groups=num_groups, residual=False)
        self.convblock2 = ConvBlock(in_ch//2, out_ch, num_groups=num_groups, residual=False)
        self.time_emb = TimeEmbedding(out_ch, out_ch)

    def forward(self, x, skip_x, t):
        y = self.up(x)
        y = self.conv1(y)
        y = self.groupnorm1(y)
        y = torch.cat([skip_x, y], dim=1)
        y = self.convblock1(y)
        y = self.convblock2(y)

        emb = self.time_emb(t)[:, :, None, None].repeat(1, 1, y.shape[-2], y.shape[-1])

        return y + emb        
    
class UNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_ch,
            out_ch,
            num_init_ch=64,
            num_downsamples=3,
            num_mid_convs=3,
            device='cuda'
            ):
        """
        Args:
            image_size (int): size of image
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            num_init_ch (int): number of channels in first layer
            num_downsamples (int): number of downsampling (and upsampling) layers
            num_mid_convs (int): number of convolutions in the middle layers
            device (str): device to run on
        """
        super().__init__()
        self.image_size = image_size
        self.device = device
        self.num_ch = num_init_ch
        num_upsamples = num_downsamples

        self.first_conv = ConvBlock(in_ch, self.num_ch,
                                     num_groups=self.num_ch//4, residual=False, dropout=0.0)

        self.downsample_layers = nn.ModuleList()
        for i in range(num_downsamples):
            num_ch = 2**i*self.num_ch
            self.downsample_layers.append(DownSample(num_ch, 2*num_ch,
                                                      num_groups=num_ch//2))
        
        self.mid_layers = nn.ModuleList()
        for i in range(num_mid_convs):
            num_ch = 2**(num_downsamples)*self.num_ch
            self.mid_layers.append(ConvBlock(num_ch,
                                              num_ch,
                                              num_groups=num_ch//4, residual=True))

        self.upsample_layers = nn.ModuleList()
        for i in range(num_upsamples):
            num_ch = 2**(num_downsamples-i)*self.num_ch
            self.upsample_layers.append(UpSample(num_ch,
                                                  num_ch//2,
                                                  num_groups=num_ch//8))
        
        self.last_conv = nn.Conv2d(self.num_ch, out_ch, kernel_size=1)
        
    
    def forward(self, x, t):
        y = self.first_conv(x)
        y_list = [y]
        for downsample in self.downsample_layers:
            y = downsample(y, t)
            y_list.append(y)
        for mid_layer in self.mid_layers:
            y = mid_layer(y)
        for i, upsample in enumerate(self.upsample_layers):
            y = upsample(y, y_list[-i-2], t)
        y = self.last_conv(y)
        return y