# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import logging
import pdb

from timm.models.vision_transformer import Block

#from util.pos_embed import get_2d_sincos_pos_embed
from ..models.utils.pos_embed import get_2d_sincos_pos_embed
from ..models.corss_transformers import CMAttention

# copied from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        # img_size = to_2tuple(img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., memory_num = 3, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

       ## modality HSI
        self.patch_embed_x = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed_x.num_patches

        self.cls_token_x = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        ## modality MSI
        self.patch_embed_y = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        #num_patches = self.patch_embed.num_patches

        self.cls_token_y = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_y = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        ## modality SAR
        self.patch_embed_z = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        #num_patches = self.patch_embed.num_patches

        self.cls_token_z = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_z = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        

        self.blocks_x = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm_x = norm_layer(embed_dim)
        self.blocks_y = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm_y = norm_layer(embed_dim)
        self.blocks_z = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm_z = norm_layer(embed_dim)
        
        
        #self.share_encoder_embed = nn.Linear(embed_dim, embed_dim*2, bias=True)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_x = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_y = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_z = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred_x = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        self.decoder_pred_y = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        self.decoder_pred_z = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        
        # --------------------------------------------------------------------------
        self.CMA_x = CMAttention(decoder_embed_dim, heads = decoder_num_heads, 
                                dim_head = decoder_embed_dim, dropout = 0.1, memory_num = memory_num)
                                
        self.CMA_y = CMAttention(decoder_embed_dim, heads = decoder_num_heads, 
                                dim_head = decoder_embed_dim, dropout = 0.1, memory_num = memory_num)

        self.CMA_z = CMAttention(decoder_embed_dim, heads = decoder_num_heads, 
                                dim_head = decoder_embed_dim, dropout = 0.1, memory_num = memory_num)               

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        #pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)  # modified
        # print(pos_embed.shape)
        # print(self.pos_embed.shape)
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        ##
        pos_embed_x = get_2d_sincos_pos_embed(self.pos_embed_x.shape[-1], self.patch_embed_x.grid_size, cls_token=True)  # modified
        self.pos_embed_x.data.copy_(torch.from_numpy(pos_embed_x).float().unsqueeze(0))
        ##
        pos_embed_y = get_2d_sincos_pos_embed(self.pos_embed_y.shape[-1], self.patch_embed_y.grid_size, cls_token=True)  # modified
        self.pos_embed_y.data.copy_(torch.from_numpy(pos_embed_y).float().unsqueeze(0))
        ##        
        pos_embed_z = get_2d_sincos_pos_embed(self.pos_embed_z.shape[-1], self.patch_embed_z.grid_size, cls_token=True)  # modified
        self.pos_embed_z.data.copy_(torch.from_numpy(pos_embed_z).float().unsqueeze(0))
        

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed_x.grid_size, cls_token=True)  # modified
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        w = self.patch_embed_x.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        w = self.patch_embed_y.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        w = self.patch_embed_z.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token_x, std=.02)
        torch.nn.init.normal_(self.mask_token_x, std=.02)
        
        torch.nn.init.normal_(self.cls_token_y, std=.02)
        torch.nn.init.normal_(self.mask_token_y, std=.02)
        
        torch.nn.init.normal_(self.cls_token_z, std=.02)
        torch.nn.init.normal_(self.mask_token_z, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # h = w = imgs.shape[2] // p
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        #x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        #x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * imgs.shape[1]))
        H = imgs.shape[2]
        W = imgs.shape[3]
        self.patch_info = (H, W, p, h, w)
        
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        #x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        #imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    '''
    def forward_mask(self, x, mask_ratio):
        # embed patches
        #print(11111, x.shape) # 4 64 128 128
        x = self.patch_embed(x)
        #print("mae的positional是："+str(x.shape))

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        # for blk in self.blocks:
            # x = blk(x)
        # x = self.norm(x)

        return x, mask, ids_restore#'''

    def forward_tokenizer(self, x, y, z, mask_ratio):
        # embed patches
        #print(11111, x.shape) # 4 64 128 128
        x = self.patch_embed_x(x)
        #print("mae的positional是："+str(x.shape))

        # add pos embed w/o cls token
        x = x + self.pos_embed_x[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask_x, ids_restore_x = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token_x = self.cls_token_x + self.pos_embed_x[:, :1, :]
        cls_tokens_x = cls_token_x.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens_x, x), dim=1)

        y = self.patch_embed_y(y)
        #print("mae的positional是："+str(x.shape))

        # add pos embed w/o cls token
        y = y + self.pos_embed_y[:, 1:, :]###维度  4 256 720

        # print(1133, y.shape)  
        # masking: length -> length * mask_ratio
        y, mask_y, ids_restore_y = self.random_masking(y, mask_ratio)

        # append cls token
        cls_token_y = self.cls_token_y + self.pos_embed_y[:, :1, :]
        cls_tokens_y = cls_token_y.expand(y.shape[0], -1, -1)
        y = torch.cat((cls_tokens_y, y), dim=1)###维度 4 65 720
        # print(1122, y.shape)  

              
        z = self.patch_embed_z(z)
        #print("mae的positional是："+str(x.shape))

        # add pos embed w/o cls token
        z = z + self.pos_embed_z[:, 1:, :]

        # masking: length -> length * mask_ratio
        z, mask_z, ids_restore_z = self.random_masking(z, mask_ratio)

        # append cls token
        cls_token_z = self.cls_token_z + self.pos_embed_z[:, :1, :]
        cls_tokens_z = cls_token_z.expand(z.shape[0], -1, -1)
        z = torch.cat((cls_tokens_z, z), dim=1)
        
        return x, y, z, mask_x, mask_y, mask_z, ids_restore_x, ids_restore_y, ids_restore_z
        
    def forward_encoder_specific(self, x, y, z):
        # apply Transformer blocks
        for blk in self.blocks_x:
            x = blk(x)
        x = self.norm_x(x)
        
        for blk in self.blocks_y:
            y = blk(y)
        y = self.norm_y(y)
        
        for blk in self.blocks_z:
            x = blk(z)
        z = self.norm_x(z)   

        return x, y, z
        
    def forward_encoder_share(self, x, y, z):
       
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) 
        
        for blk in self.blocks:
            y = blk(y)
        y = self.norm(y) 
        
        
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z) 
        
        return x, y, z
    '''
    def forward_decoder_share(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        #x = self.decoder_pred(x)

        # remove cls token
        #x = x[:, 1:, :]

        return x#'''
        
    def forward_decoder_share(self, x, y, z, ids_restore_x, ids_restore_y, ids_restore_z):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens_x = self.mask_token_x.repeat(x.shape[0], ids_restore_x.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens_x], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore_x.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        #x = self.decoder_pred(x)

        # remove cls token
        #x = x[:, 1:, :]
        
        y = self.decoder_embed(y)

        # append mask tokens to sequence
        mask_tokens_y = self.mask_token_y.repeat(y.shape[0], ids_restore_y.shape[1] + 1 - y.shape[1], 1)
        y_ = torch.cat([y[:, 1:, :], mask_tokens_y], dim=1)  # no cls token
        y_ = torch.gather(y_, dim=1, index=ids_restore_y.unsqueeze(-1).repeat(1, 1, y.shape[2]))  # unshuffle
        y = torch.cat([y[:, :1, :], y_], dim=1)  # append cls token

        # add pos embed
        y = y + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            y = blk(y)
        y = self.decoder_norm(y)
        
        z = self.decoder_embed(z)

        # append mask tokens to sequence
        mask_tokens_z = self.mask_token_z.repeat(z.shape[0], ids_restore_z.shape[1] + 1 - z.shape[1], 1)
        z_ = torch.cat([z[:, 1:, :], mask_tokens_z], dim=1)  # no cls token
        z_ = torch.gather(z_, dim=1, index=ids_restore_z.unsqueeze(-1).repeat(1, 1, z.shape[2]))  # unshuffle
        z = torch.cat([z[:, :1, :], z_], dim=1)  # append cls token

        # add pos embed
        z = z + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            z = blk(z)
        z = self.decoder_norm(z)
        
        return x, y, z
        
    def forward_decoder_specific(self, x, y, z):

        # predictor projection
        inp_x = x
        inp_y = y
        inp_z = z
        
        x = self.CMA_x(x, inp_x, inp_y, inp_z)
        x = self.decoder_pred_x(x)
        
        y = self.CMA_y(y, inp_x, inp_y, inp_z)
        y = self.decoder_pred_y(y)
        
        z = self.CMA_z(z, inp_x, inp_y, inp_z)
        z = self.decoder_pred_z(z)
        
        # remove cls token
        x = x[:, 1:, :]
        y = y[:, 1:, :]
        z = z[:, 1:, :]
        
        return x, y, z
        
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgsx, imgsy, imgsz, mask_ratio=0.75):
    
        # latentx, mask_x, ids_restore_x = self.forward_mask(imgsx, mask_ratio)
        # latenty, mask_y, ids_restore_y = self.forward_mask(imgsy, mask_ratio)
        # latentz, mask_z, ids_restore_z = self.forward_mask(imgsz, mask_ratio)
        latentx, latenty, latentz, mask_x, mask_y, mask_z, ids_restore_x, ids_restore_y, ids_restore_z = self.forward_tokenizer(imgsx, imgsy, imgsz, mask_ratio)
        # print(111, latentx.shape, latenty.shape, latentz.shape)## 4 65 720
        # print(222, mask_x.shape, mask_y.shape, mask_z.shape) ## 4 256 
        # print(333, ids_restore_x.shape, ids_restore_y.shape, ids_restore_z.shape)## 4 256
        
        ## 应该保持不同模态的mask不同，因为从所有令牌中统一选择可见令牌来调整MAE掩码采样策略，将导致大多数模态以相似的程度表示
        ## 即使来自一个模态的标记很少可见，由于跨模态的相互作用，所产生的预测也相对稳定和可信。
        ## 论文 MultiMAE: Multi-modal Multi-task Masked Autoencoders
        
        latentx, latenty, latentz = self.forward_encoder_specific(latentx, latenty, latentz)

        # latentx = self.forward_encoder_share(latentx)
        # latenty = self.forward_encoder_share(latenty)
        # latentz = self.forward_encoder_share(latentz)
        latentx, latenty, latentz = self.forward_encoder_share(latentx, latenty, latentz)# ？ 65 720
        #print(11111111111111111111, latentx.shape)

        ## 这里的latent指的是mask后剩余patch对应的特征。
        #logging
        # fileName='log.txt'
        # pdb.set_trace()
        # with open(fileName,'w')as file:
        #     file.write('average')
        #pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]  ? 64 16*16*64

        #pred_x = self.forward_decoder_share(latentx, ids_restore_x)  # [N, L, p*p*3]  ? 64 16*16*64
        #pred_y = self.forward_decoder_share(latenty, ids_restore_y)  # [N, L, p*p*3]  ? 64 16*16*64
        #pred_z = self.forward_decoder_share(latentz, ids_restore_z)  # [N, L, p*p*3]  ? 64 16*16*64
        pred_x, pred_y, pred_z = self.forward_decoder_share(latentx, latenty, latentz, ids_restore_x, ids_restore_y, ids_restore_z)  # [N, L, p*p*3]  ? 64 16*16*64
        
        pred_x, pred_y, pred_z = self.forward_decoder_specific(pred_x, pred_y, pred_z)
        #pred_x, pred_y, pred_z = self.forward_decoder_joint(pred_x, pred_y, pred_z)

        #print(11111111, pred.shape)
        loss_x = self.forward_loss(imgsx, pred_x, mask_x)
        loss_y = self.forward_loss(imgsy, pred_y, mask_y)
        loss_z = self.forward_loss(imgsz, pred_z, mask_z)
        
        #latent = self.forward_encoder_seg(imgs)
       
        H, W, p, h, w = self.patch_info
        
        pred_x = pred_x.reshape(shape=(pred_x.shape[0], H, W, -1))
        pred_x = torch.einsum('nhwc->nchw', pred_x)
        
        pred_y = pred_y.reshape(shape=(pred_y.shape[0], H, W, -1))
        pred_y = torch.einsum('nhwc->nchw', pred_y)
        
        pred_z = pred_z.reshape(shape=(pred_z.shape[0], H, W, -1))
        pred_z = torch.einsum('nhwc->nchw', pred_z)
        
        loss = loss_x + loss_y + loss_z
        
        return loss, pred_x, pred_y, pred_z


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
