
import torch
import torch.nn as nn
from .transformer.patch_operations import PatchEmbed, PatchMerging, FinalPatchExpand_X4
from .transformer.basic_layer import BasicLayer
from .encoder import EncoderMiniBlock
from .decoder import DecoderMiniBlock
from .attention import CBAM



class LN_UXFormer(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters=32, 
                 depths=[2, 2, 2, 2], num_heads=[2, 4, 8, 16], window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 img_size=224, patch_size=4):
        super(LN_UXFormer, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.ape = ape
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=n_channels, embed_dim=n_filters,
            norm_layer=norm_layer if patch_norm else None
        )
        
        self.patch_embed_1 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=32, embed_dim=32,
            norm_layer=norm_layer if patch_norm else None
        )
        
        self.patch_embed_2 = PatchEmbed(
            img_size=112, patch_size=patch_size, in_chans=64, embed_dim=64,
            norm_layer=norm_layer if patch_norm else None
        )
        
        self.patch_embed_3 = PatchEmbed(
            img_size=56, patch_size=patch_size, in_chans=128, embed_dim=128,
            norm_layer=norm_layer if patch_norm else None
        )
        
        self.patch_embed_4 = PatchEmbed(
            img_size=28, patch_size=patch_size, in_chans=256, embed_dim=256,
            norm_layer=norm_layer if patch_norm else None
        )
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, n_filters))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)        
        
        self.cblock1 = EncoderMiniBlock(n_channels, n_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.cblock2 = EncoderMiniBlock(n_filters, n_filters*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.cblock3 = EncoderMiniBlock(n_filters*2, n_filters*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.cblock4 = EncoderMiniBlock(n_filters*4, n_filters*8)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.cblock5 = EncoderMiniBlock(n_filters*8, n_filters*16)

        self.swin1 = BasicLayer(
            dim=n_filters, input_resolution=(56, 56), depth=depths[0], num_heads=num_heads[0], 
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer
        )
        self.patch_merging1 = PatchMerging(input_resolution=(56, 56), dim=n_filters)
        
        self.swin2 = BasicLayer(
            dim=n_filters*2, input_resolution=(28, 28), depth=depths[1], num_heads=num_heads[1], 
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer
        )
        self.patch_merging2 = PatchMerging(input_resolution=(28, 28), dim=n_filters*2)
        
        self.swin3 = BasicLayer(
            dim=n_filters*4, input_resolution=(14, 14), depth=depths[2], num_heads=num_heads[2], 
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer
        )
        self.patch_merging3 = PatchMerging(input_resolution=(14, 14), dim=n_filters*4)
        
        self.swin4 = BasicLayer(
            dim=n_filters*8, input_resolution=(7, 7), depth=depths[3], num_heads=num_heads[3], 
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer
        )
        
        self.cbam1 = CBAM(n_filters * 2)
        self.cbam2 = CBAM(n_filters * 4)
        self.cbam3 = CBAM(n_filters * 8)
        self.cbam4 = CBAM(n_filters * 16)
        
        self.up4 = nn.ConvTranspose2d(n_filters*16, n_filters*8, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(n_filters*8, n_filters*4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(n_filters*4, n_filters*2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(n_filters*2, n_filters, kernel_size=2, stride=2)
        
        self.ublock6 = DecoderMiniBlock(n_filters*16, n_filters*8)
        self.ublock7 = DecoderMiniBlock(n_filters*8, n_filters*4)
        self.ublock8 = DecoderMiniBlock(n_filters*4, n_filters*2)
        self.ublock9 = DecoderMiniBlock(n_filters*2, n_filters)
        
        self.conv10 = nn.Conv2d(n_filters, n_classes, kernel_size=1)
        
        self.scale_up1 = FinalPatchExpand_X4(input_resolution=(56, 56), dim_scale=4, dim=32)
        self.scale_up2 = FinalPatchExpand_X4(input_resolution=(28, 28), dim_scale=4, dim=64)
        self.scale_up3 = FinalPatchExpand_X4(input_resolution=(14, 14), dim_scale=4, dim=128)
        self.scale_up4 = FinalPatchExpand_X4(input_resolution=(7, 7), dim_scale=4, dim=256)

    def forward(self, x):
        x_swin = self.patch_embed(x)
        if self.ape:
            x_swin = x_swin + self.absolute_pos_embed
        x_swin = self.pos_drop(x_swin)
        
        c1 = self.cblock1(x)
        c1_vit = self.patch_embed_1(c1)
        s1 = self.swin1(x_swin, c1_vit)
        s1_scale_up = self.scale_up1(s1)
        s1_scale_up = s1_scale_up.view(s1_scale_up.shape[0], 224, 224, 32)
        s1_scale_up = s1_scale_up.permute(0, 3, 1, 2)
        
        p1 = self.pool1(c1)
        c2 = self.cblock2(p1)
        c2_vit = self.patch_embed_2(c2)
        s2 = self.patch_merging1(s1)
        s2 = self.swin2(s2, c2_vit)
        s2_scale_up = self.scale_up2(s2)
        s2_scale_up = s2_scale_up.view(s2_scale_up.shape[0], 112, 112, 64)
        s2_scale_up = s2_scale_up.permute(0, 3, 1, 2)
        
        p2 = self.pool2(c2)
        c3 = self.cblock3(p2)
        c3_vit = self.patch_embed_3(c3)
        s3 = self.patch_merging2(s2)
        s3 = self.swin3(s3, c3_vit)
        s3_scale_up = self.scale_up3(s3)
        s3_scale_up = s3_scale_up.view(s3_scale_up.shape[0], 56, 56, 128)
        s3_scale_up = s3_scale_up.permute(0, 3, 1, 2)
        
        p3 = self.pool3(c3)
        c4 = self.cblock4(p3)
        c4_vit = self.patch_embed_4(c4)
        s4 = self.patch_merging3(s3)
        s4 = self.swin4(s4, c4_vit)
        s4_scale_up = self.scale_up4(s4)
        s4_scale_up = s4_scale_up.view(s4_scale_up.shape[0], 28, 28, 256)
        s4_scale_up = s4_scale_up.permute(0, 3, 1, 2)
        
        p4 = self.pool4(c4)
        c5 = self.cblock5(p4)
        
        u4 = self.up4(c5)
        a4 = self.cbam4(torch.cat([s4_scale_up, u4], dim=1))
        d4 = self.ublock6(a4)
        
        u3 = self.up3(d4)
        a3 = self.cbam3(torch.cat([s3_scale_up, u3], dim=1))
        d3 = self.ublock7(a3)
        
        u2 = self.up2(d3)
        a2 = self.cbam2(torch.cat([s2_scale_up, u2], dim=1))
        d2 = self.ublock8(a2)
        
        u1 = self.up1(d2)
        a1 = self.cbam1(torch.cat([s1_scale_up, u1], dim=1))
        d1 = self.ublock9(a1)
        
        output = self.conv10(d1)
        
        return output