import os
import logging

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from utils.highDHA_utils import initialize_weights
#from networks.models.models_mae import MaskedAutoencoderViT
from .models.models_Swin_mae import MaskedAutoencoderViT, Swin_MAE_Segmenter
from einops import rearrange
from .models.SwinLSTM_B import SwinLSTM

BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

class FFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        
        return x
        
class IFFT(nn.Module):
    def __init__(self, dim, h=128, w=128):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(int((w+2)/2), h, dim, 2, dtype=torch.float32) * 0.02)
        self.W = w
        self.H = h

    def forward(self, x):

        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(self.W, self.H), dim=(1, 2), norm='ortho')
        return x 

class ComplexConv2d(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    def apply_complex(self, fr, fi, input, dtype = torch.complex64):
        return (fr(input.real)-fi(input.imag)).type(dtype) \
                + 1j*(fr(input.imag)+fi(input.real)).type(dtype)
            
    def forward(self, input):    
        return self.apply_complex(self.conv_r, self.conv_i, input)


##  动态大尺度卷积
class DLK(nn.Module):
    def __init__(self, dim, h=128, w=128):
        super().__init__()
        # self.att_conv1 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        # self.att_conv2 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        self.att_conv1 = ComplexConv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.att_conv2 = ComplexConv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)


        # self.spatial_se = nn.Sequential(
            # nn.Conv2d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            # nn.Sigmoid()
        # )
        self.spatial_se = nn.Sequential(
            ComplexConv2d(in_channels=2, out_channels=2, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.fft = FFT()
        self.ifft = IFFT(dim, h=h, w=w)
        
    def forward(self, x):
        
        x_fft = self.fft(x)
        
        att1 = self.att_conv1(x_fft)
        att2 = self.att_conv2(att1)
        
        att = torch.cat([att1, att2], dim=1)
        
        realmean = torch.mean(att.real, dim=1, keepdim=True)
        imagmean = torch.mean(att.imag, dim=1, keepdim=True)
        avg_att = torch.complex(realmean, imagmean)
        
        realmax,_ = torch.max(att.real, dim=1, keepdim=True)
        imagmax,_ = torch.max(att.imag, dim=1, keepdim=True)
        max_att = torch.complex(realmax, imagmax)
        
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        output = att1 * att[:,0,:,:].unsqueeze(1) + att2 * att[:,1,:,:].unsqueeze(1)
        
        output = rearrange(output, 'B C H W  -> B H W C')
        output = self.ifft(output)
        output = rearrange(output, 'B H W C  -> B C H W')
        
        output = output + x
        
        return output


class HMDADANet(nn.Module):

    def __init__(self, band, patchsize, num_classes, num_channels = 64):
        super(HMDADANet, self).__init__()
        
        self.patchsize = patchsize
        self.DLKModule = DLK(num_channels, h=patchsize, w=patchsize)
        # stem net for hsi
        self.conv1 = nn.Conv2d(band, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        # stem net for msi
        self.conv_msi = nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_msi = nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM)
        # stem net for sar
        self.conv_sar = nn.Conv2d(2, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_sar = nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM)
        
        self.MAE_dim = 96
        window_size = 4
        self.patch_size = 2
        ## MIM的输入均是降维64x64x64
        self.Swin_MAE = MaskedAutoencoderViT(img_size=(patchsize, patchsize), 
                        in_chans=num_channels, patch_size=self.patch_size, 
                        embed_dim=self.MAE_dim, decoder_embed_dim=self.MAE_dim, 
                        mlp_ratio=4., memory_num = 3, 
                        window_size = window_size)
                        
        self.Swin_MAE_Segmenter = Swin_MAE_Segmenter(self.Swin_MAE.Encoder, 
                        dim = self.MAE_dim, window_size = window_size)
        
        out_channel = 64 #self.MAE_dim
        self.convx1 = nn.Conv2d(self.MAE_dim, out_channel, 1, 1)              
        self.convx2 = nn.Conv2d(self.MAE_dim*2, out_channel, 1, 1)
        self.convx3 = nn.Conv2d(self.MAE_dim*4, out_channel, 1, 1)
        self.convy1 = nn.Conv2d(self.MAE_dim, out_channel, 1, 1)
        self.convy2 = nn.Conv2d(self.MAE_dim*2, out_channel, 1, 1)
        self.convy3 = nn.Conv2d(self.MAE_dim*4, out_channel, 1, 1)
        self.convz1 = nn.Conv2d(self.MAE_dim, out_channel, 1, 1)
        self.convz2 = nn.Conv2d(self.MAE_dim*2,out_channel, 1, 1)
        self.convz3 = nn.Conv2d(self.MAE_dim*4, out_channel, 1, 1)
   
 
        self.swinLSTM = SwinLSTM(img_size=(patchsize//self.patch_size, patchsize//self.patch_size), patch_size=1,
                         in_chans=out_channel, embed_dim=out_channel,
                         depths=(1, 1), num_heads=(1, 1),
                         window_size=window_size, drop_rate = 0.,
                         attn_drop_rate = 0., drop_path_rate=0.1)
   
        self.fuse_out = (out_channel)*6
        
        self.transconv = nn.Sequential(
            # nn.Conv2d(self.fuse_out, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
            # nn.ConvTranspose2d(self.fuse_out, 512, kernel_size=2, stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
            nn.ConvTranspose2d(self.fuse_out, 128, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            # nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
            # nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
        )
        self.final_conv = nn.Conv2d(128, num_classes, 1, 1)
        
        self.tanh = nn.Tanh()
    
    def forward(self, x, y, D, domain):
        _, _, height, width = x.shape #? 10 128 128

        ## 维度对齐
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        y = self.relu(self.bn_msi(self.conv_msi(y)))

        x = self.DLKModule(x)
        y = self.DLKModule(y)
        
        ## 重构与特征提取
        
        loss = self.Swin_MAE(x, y, mask_ratio=0.75)
        x_list, y_list = self.Swin_MAE_Segmenter(x, y)# ? 8 8 384*3
        
        x0_h = self.patchsize//self.patch_size
        x0_w = self.patchsize//self.patch_size
        
        #x1 = F.interpolate(x_list[0], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x1 = x_list[0]
        x1 = self.convx1(x1)
        x2 = F.interpolate(x_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = self.convx2(x2)
        x3 = F.interpolate(x_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = self.convx3(x3)
        #y1 = F.interpolate(y_list[0], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        y1 = y_list[0]
        y1 = self.convy1(y1)
        y2 = F.interpolate(y_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        y2 = self.convy2(y2)
        y3 = F.interpolate(y_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        y3 = self.convy3(y3)
        
        outputs = []
        states = [None] * 2

        x1 = rearrange(x1, 'B C H W -> B H W C')
        x2 = rearrange(x2, 'B C H W -> B H W C')
        x3 = rearrange(x3, 'B C H W -> B H W C')
        y1 = rearrange(y1, 'B C H W -> B H W C')
        y2 = rearrange(y2, 'B C H W -> B H W C')
        y3 = rearrange(y3, 'B C H W -> B H W C')
        
        firstinput = (x1, y1, x2, y2, x3)
        last_input = y3
        
        for i in range(5):
            output, states = self.swinLSTM(firstinput[i], states)
            outputs.append(output)

        for i in range(1):
            output, states = self.swinLSTM(last_input, states)
            outputs.append(output)

        x = torch.cat(outputs, 3)## 4 2016 32 32
        x = rearrange(x, 'B H W C -> B C H W')#'''
        ## domain adaptation
        if domain == 'source':
            xt = x
        if domain == 'target':
            xt = x
            # aa = D[0](x) 
            # aa = self.tanh(aa)
            # aa = torch.abs(aa)
            # aa_big = aa.expand(x.size()) ## attention map，像素级置信分数用于重加权中间特征V，纠正不同域局部表示偏移
            # xt = aa_big * x + x          ## 这里的x指的是论文4.4中的V
            
        ### head
        out = self.transconv(xt)
        out = self.final_conv(out)
        
        #return x, refine_x, refine_y, refine_z, out, loss
        return x, x_list[2], y_list[2], out, loss

    # def init_weights(self, pretrained='', ):
        # logger.info('=> init weights from normal distribution')
        # for m in self.modules():
            # if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, std=0.001)
            # elif isinstance(m, nn.BatchNorm2d):
                # nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
        # if os.path.isfile(pretrained):
            # pretrained_dict = torch.load(pretrained)
            # logger.info('=> loading pretrained model {}'.format(pretrained))
            # model_dict = self.state_dict()
            # pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               # if k in model_dict.keys()}
            # for k, _ in pretrained_dict.items():
                # logger.info(
                    # '=> loading {} pretrained model {}'.format(k, pretrained))
            # model_dict.update(pretrained_dict)
            # self.load_state_dict(model_dict)
