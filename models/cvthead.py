"""
1) DECA for face + Rome hair Deformation 
2) Src Image + Transformer --> Vertex descriptor
3) Depth Projector with Z-buffer
3) Drving Depth Image with deformation
normalized -1,1
"""
import os
import math
import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torchvision

from models.transformer import Transformer, StemNet, PatchEmbed
from models.autoencoder import Autoencoder
from models.unet import UNet
from models.face_parsing import FaceParsing
from models.spiralnet import get_coarse_mesh_decoder, downsample_vertices

from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.util import batch_orth_proj, vertex_normals


EMB_DIM = 128


def transform_points(points, tform, points_scale=None, out_scale=None):
    # points, (B, V, 3)
    points_2d = points[:,:,:2]
    #'input points must use original range'
    if points_scale:
        assert points_scale[0]==points_scale[1]
        points_2d = (points_2d*0.5 + 0.5)*points_scale[0]
    batch_size, n_points, _ = points.shape
    trans_points_2d = torch.bmm(
                    torch.cat([points_2d, torch.ones([batch_size, n_points, 1], device=points.device, dtype=points.dtype)], dim=-1), 
                    tform
                    ) 
    if out_scale: # h,w of output image size
        trans_points_2d[:,:,0] = trans_points_2d[:,:,0]/out_scale[1]*2 - 1
        trans_points_2d[:,:,1] = trans_points_2d[:,:,1]/out_scale[0]*2 - 1
    trans_points = torch.cat([trans_points_2d[:,:,:2], points[:,:,2:]], dim=-1)
    return trans_points



class my_round_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.round(input).long()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


def batched_index_select(input, dim, index):
    # input:(B, C, HW). index(B, N)
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index) # (B,C, N)


def verts_to_proj_inds(verts, h, w):
    # verts, (B, V, 3)
    b, num_verts, _ = verts.shape

    depth_val = verts[:, :, 2] # (B, V)
    _, ind_sort = depth_val.sort(dim=-1, descending=True)  # sort by depth (far to near), thus near depth will over-write
    # ind_sort, (B,V)

    u = torch.gather(verts[:, :, 0], -1, ind_sort)              # (B, V)  [-1, 1]
    v = torch.gather(verts[:, :, 1], -1, ind_sort)              # (B, V)  [-1, 1]
    u = u * w / 2 + w / 2                       # (B, V) in image plane  float
    v = v * h / 2 + h / 2                       # (B, V) in image plane  float
    u, v = u.clamp(0, w - 1),  v.clamp(0, h - 1)       # (B, V) in image plane  float
    uv_idx = my_round_func.apply(v) * h + my_round_func.apply(u)        # (B, V), coordinate index on image plane

    # verts index 
    verts_ind = torch.arange(1, num_verts + 1).to(verts.device)     # (1, V) val:from 1 to V, float
    verts_ind = verts_ind[ind_sort]                                 # (B, V)

    verts_ind_img = torch.zeros(b, h, w).long().to(verts.device)           # (B, H, W), val:0
    verts_ind_img.flatten(1, 2).scatter_(1, uv_idx, verts_ind)      # (B, H, W), float
    return verts_ind_img, uv_idx


def verts_feature_assign(verts_ind_img, verts_feat, pad_val=0):
    # verts_ind_img, (B, H, W), float, val:1-V
    # verts_feat, (B, V, C), float
    c = verts_feat.shape[2]
    b, h, w = verts_ind_img.shape

    # padding
    pad_val = pad_val * torch.ones(b, 1, c).to(verts_feat.device)
    verts_feat = torch.cat([pad_val, verts_feat], dim=1)    # (B, V+1, C)
    verts_feat = verts_feat.permute(0, 2, 1)        # (B, C, V+1)

    inds_f = verts_ind_img.flatten(1, 2).long()                 # (B, HW), val:1-V
    sample = batched_index_select(verts_feat, 2, inds_f)        # (B, C, HW)
    sample = sample.reshape(b, c, h, w)                         # (B, C, H, W)
    return sample


class CVTHead(nn.Module):
    def __init__(self, emb_d=32, deca_finetune=False):
        super().__init__()

        print(" ************ Load pre-traiend Face Parsing Model ************")
        self.face_parsing = FaceParsing(ckpt_path="data/79999_iter.pth")
        self.face_parsing.eval()
        for param in self.face_parsing.parameters():
                param.requires_grad = False

        print("************ Load pre-traiend Hair+Shoulder Deformation Model ************")
        hair_model_pth = "data/linear_hair.pth"
        mean_deform_pth = "data/mean_deform.pt"
        u_full_pth = "data/u_full.pt"

        hair_model = torchvision.models.resnet50()
        hair_model.fc = nn.Linear(in_features=2048, out_features=60, bias=True)     # (50 + 10)
        
        hair_model.load_state_dict(torch.load(hair_model_pth, map_location='cpu'))
        self.hair_model = hair_model
        self.hair_model.eval()
        if not deca_finetune:
            for param in self.hair_model.parameters():
                param.requires_grad = False
        
        mean_deform = torch.load(mean_deform_pth, map_location='cpu')       # (5023, 3) augment hair and shoulder region 
        u_full = torch.load(u_full_pth, map_location='cpu')                 # (15069, 60)
        u_full = u_full.reshape(5023, 3, 60)
        self.register_buffer("deform_mean", mean_deform)
        self.register_buffer("deform_basis", u_full)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DECA model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        print("\n\n ************ Load pre-traiend DECA ************")
        self.deca = DECA(config=deca_cfg)
        self.deca.eval()
        self.deca_size = [224, 224]  
        if not deca_finetune:
            for param in self.deca.parameters():
                param.requires_grad = False

        print("\n\n ************ Create mesh Render Transformer Model ************")


        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Vertex Feature Transformer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.autoencoder = Autoencoder(64, 512)

        if os.path.exists('data/rome.pth'):
            print("Load Pre-trained CNN Encoder weights of ROME ...")
            rome_dict = torch.load("data/rome.pth", map_location="cpu")
            rome_dict = {k: v for k, v in rome_dict.items() if 'autoencoder' in k}              # only keep autoencoder weights
            rome_dict = {k.replace('autoencoder.', ''): v for k, v in rome_dict.items()}        # remove "autoencoder." prefix
            ae_keys = self.autoencoder.state_dict().keys()
            rome_encoder_dict = {}
            for k in ae_keys:
                if k in rome_dict.keys():
                    # print("--- Load ROME Encoder weights: ", k, rome_dict[k].shape)
                    rome_encoder_dict[k] = rome_dict[k]
            self.autoencoder.load_state_dict(rome_encoder_dict)
        else:
            print("CNN encoder from scratch!!")

        # Linear Projection
        self.ae_proj = nn.Conv2d(2048, 256, 1, 1, 0)
        patch_dim = 256 * 2 * 2 
        self.patch_to_embedding = nn.Linear(patch_dim, EMB_DIM)

        self.pos_embedding = nn.Parameter(self._make_sine_position_embedding(EMB_DIM), requires_grad=False)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> mesh decoder >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        vtx_upsampler, down_transform_list = get_coarse_mesh_decoder(emb_dim=EMB_DIM, transform_fp = "data/transform.pkl", down_degree=2)
        self.vtx_upsampler = vtx_upsampler
        self.num_v_coarse = down_transform_list[-1].shape[0]        # 314
        self.down_transform_list = down_transform_list

        self.vtx_query = nn.Parameter(torch.zeros(1, self.num_v_coarse, EMB_DIM))       # (1, Vc=314, C-3)
        self.transformer = Transformer(dim=EMB_DIM, depth=6, heads=4, dim_head=EMB_DIM // 4, mlp_dim=EMB_DIM * 2)

        # output head
        self.vtx_descriptor_head = nn.Sequential(
            nn.Linear(EMB_DIM, emb_d)
        )

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> U-Net Rendering >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.unet_generator = UNet(in_channels=1 + emb_d,       # depth + vtx feature
            out_channels=3 + 1,     # RGB + seg
            start_filts=32, 
            n_blocks=5)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = 16, 16

        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        print("==> Add Sine PositionEmbedding~")
        return pos


    def forward_vtx_deform(self, img):
        # ROME pre-trained linear hair/shoulder deformation model
        self.hair_model.eval()      # !!!
        beta_hair = self.hair_model(img)        # (B, 60)
        blend_shape = torch.einsum('bl,mkl->bmk', [beta_hair, self.deform_basis])       # (B, 60) * (5023, 3, 60) --> (B, 5023, 3)
        vtx_deform = blend_shape + self.deform_mean        # (B, 5023, 3)
        return vtx_deform

    def uv_sin_pos_encoding(self, uv, temperature=10000, scale=2 * math.pi):
        # uv, (B, N, 2). val:[0,1]
        uv = uv * scale 
        u, v = uv[:, :, 0:1], uv[:, :, 1:2]
        one_direction_feats = EMB_DIM // 2
        dim_t = torch.arange(one_direction_feats, dtype=torch.float32).to(uv.device)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)
        pos_u, pos_v = u / dim_t, v / dim_t
        pos_u = torch.stack([pos_u[:, :, 0::2].sin(), pos_u[:, :, 1::2].cos() ], dim=3).flatten(2)
        pos_v = torch.stack([pos_v[:, :, 0::2].sin(), pos_v[:, :, 1::2].cos()], dim=3).flatten(2)
        pos = torch.cat((pos_v, pos_u), dim=2)      # (B, N, C)
        return pos

    def depth_sin_pos_encoding(self, depth, temperature=10000, scale = 2 * math.pi):
        # depth, (B, N, 1)     val:[0,1]
        depth = depth * scale 
        dim_t = torch.arange(EMB_DIM, dtype=torch.float32).to(depth.device)
        dim_t = temperature ** (2 * (dim_t // 2) / EMB_DIM)
        pos_depth  = depth / dim_t
        pos_depth = torch.stack([pos_depth[:, :, 0::2].sin(), pos_depth[:, :, 1::2].cos() ], dim=3).flatten(2)  # (B, N, C)
        return pos_depth

    def foward_vtx_feat(self, img, mask, vertices):
        # **** vertex feature transformer **** 
        # img, (B, 3, 256, 256)
        # mask, (B, 1, 256, 256)
        # vertices, (B, V=5023, 3), in image coordinate, [-1, 1], (u,v,d)
        b = img.shape[0]

        # CNN Image feature Extractor
        x = torch.cat([img, mask], dim=1)       # (B, 4, 256, 256)
        x = self.autoencoder(x)                 # (B, 2048, 32, 32)
        x = self.ae_proj(x)                     # (B, 256, 32, 32)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 2, p2 = 2)        # (B, 16 * 16, P1P2C)
        x = self.patch_to_embedding(x)             # (B, 16 * 16, EMB_DIM)
        img_feat = x + self.pos_embedding          # (B, hw, C) + (1, hw, C)

        # learnable vertex queries
        vtx_query = repeat(self.vtx_query, '() n d -> b n d', b = b)    # (1, Vc, C) --> (B, Vc=314, C)

        #  2D projection of source vertices to pos encoding 
        coarse_vertices = downsample_vertices(vertices, self.down_transform_list)     # (B, Vc=314, 3), [-1, 1]
        uv_pos_emb = self.uv_sin_pos_encoding(0.5 * coarse_vertices[:, :, :2] + 0.5)  # (B, Vc=314, C)
        depth_emb = self.depth_sin_pos_encoding(0.5 * coarse_vertices[:, :, 2:] + 0.5)   # (B, Vc=314, C)

        vtx_query = vtx_query + uv_pos_emb + depth_emb        # (B, Vc, C)

        # Transformer Encoder Layers
        tokens = torch.cat([vtx_query, img_feat], dim=1)      # (B, Vc + hw, C)
        tokens = self.transformer(tokens)                     # (B, Vc + hw, C)

        # feature head 
        vtx_feat = self.vtx_upsampler(tokens[:, :self.num_v_coarse])      # (B, V=5023, C)
        vtx_descriptor = self.vtx_descriptor_head(vtx_feat)          # (B, V=5023, C)
        return vtx_descriptor
    
    def verts_aug_transfer(self, verts_aug, cam,  tform, h, w):
        # Follow DECA https://github.com/yfeng95/DECA/blob/master/decalib/deca.py#L176
        trans_verts_aug = batch_orth_proj(verts_aug, cam)   # (B, V=5023, 3) in cropped driving image coordinate in [-1, 1]
        trans_verts_aug[:,:,1:] = -trans_verts_aug[:,:,1:]
        tform = torch.inverse(tform).transpose(1, 2)                        # (B, 3, 3)
        trans_verts_aug = transform_points(trans_verts_aug, tform, self.deca_size, [h, w])  
        return trans_verts_aug

    def forward(self, crop_src_img, crop_drv_img, src_img, drv_img, 
                src_tform, drv_tform, is_train=True, hair_deform=True, is_cross_id=False):
        # src/drv crop_images, (B, 3, 224, 224), val: 0-1       (face only, for DECA encoder)
        # src/drv ori_images, (B, 3, 256, 256), val: [-1, 1]    (include hair + shoulder areas)
        # src/drv tforms, (B, 3, 3) from original image to deca cropped image

        b, _, H, W = src_img.shape       # (256, 256)

        outputs = {}
        with torch.no_grad():
            # 0) pre-trained face segmentation (include hair + shoulder)
            self.face_parsing.eval()        # !!!!
            # driven face mask
            drv_mask_gt, _  = self.face_parsing(drv_img)   # (B, C=19, H, W)  detach
            drv_mask = drv_mask_gt.argmax(1)               # (B, H, W)
            drv_mask = (drv_mask > 0) + 0
            drv_mask = drv_mask.unsqueeze(1).float()       # (B, 1, H, W)
            # source face mask
            src_mask_gt, _ = self.face_parsing(src_img)    # (B, C=19, H, W) detach
            src_mask = src_mask_gt.argmax(1)               # (B, H, W)
            src_mask = (src_mask > 0 ) + 0          
            src_mask = src_mask.unsqueeze(1).float()       # (B, 1, H, W)
            outputs["drv_mask_gt"] = drv_mask
            outputs["src_mask_gt"] = src_mask

            # 1) DECA Encoder forward 
            # src image
            self.deca.eval()        # !!!!
            src_codedict = self.deca.encode(crop_src_img, use_detail=False)   
            src_outputs = self.deca.decode(src_codedict, rendering=False, vis_lmk=False, return_vis=False, use_detail=False)
            src_verts = src_outputs["verts"]   # (B, V, 3) in canonical space
            # drv image
            drv_codedict = self.deca.encode(crop_drv_img, use_detail=False)   
            drv_codedict["shape"] = src_codedict["shape"]       # exchange shape !!! 
            drv_outputs = self.deca.decode(drv_codedict, rendering=False, vis_lmk=False, return_vis=False, use_detail=False)
            drv_verts = drv_outputs["verts"]   # (B, V, 3) in canonical space

            # 2) Add deformation to driving image
            # linear model
            if hair_deform:
                vtx_deform = self.forward_vtx_deform(src_img)       # (B, V, 3)
                src_verts_aug = src_verts + vtx_deform              # (B, V, 3) in canonical space, with hair and shoulder augmentation
                drv_verts_aug = drv_verts + vtx_deform              # (B, V, 3) in canonical space, with hair and shoulder augmentation
            else:
                src_verts_aug = src_verts              # (B, V, 3) in canonical space without hair and shoulder augmentation
                drv_verts_aug = drv_verts              # (B, V, 3) in canonical space without hair and shoulder augmentation 
            
            # !!!!!! when performs cross-identity generation, replace drv tform with src tform, 
            # since differnt face shape result in different transformation
            if is_cross_id:
                drv_tform = src_tform # 

            # project vertices in canonical space to 2D image space
            src_trans_verts_aug = self.verts_aug_transfer(src_verts_aug, src_codedict["cam"], src_tform, H, W)  # (B, V, 3), val: (u,v,d),[-1,1]
            drv_trans_verts_aug = self.verts_aug_transfer(drv_verts_aug, drv_codedict["cam"], drv_tform, H, W)  # (B, V, 3)

        # 3) Vertices Feature Transformer 
        vtx_descriptor = self.foward_vtx_feat(src_img, src_mask, src_trans_verts_aug)  # (B, V=5023, C)

        # 4) Projection for Driven Image 
        drv_ind_img, _ = verts_to_proj_inds(drv_trans_verts_aug, H, W)      # (B, H, W), val: {0,1,..,V}
        # Depth Image of driving
        drv_depth_img = verts_feature_assign(drv_ind_img, drv_trans_verts_aug[:, :, 2:], pad_val=-2) # (B, 1, H, W)
        # Vtx feature of driving
        drv_vtx_feat_image = verts_feature_assign(drv_ind_img, vtx_descriptor, pad_val=0)            # (B, C, H, W)

        # 5) U-Net rendering
        x = torch.cat([drv_depth_img, drv_vtx_feat_image], dim=1)        # (B, 1+C, H, W)
        pred_drv = self.unet_generator(x)
        pred_drv_img = torch.tanh(pred_drv[:, :3, :, :])             # (B, 3, H=256, W=256), val: [-1, 1]
        pred_drv_mask = torch.sigmoid(pred_drv[:, 3:, :, :])            # (B, 1, H=256, W=256), val: (0, 1)

        outputs["pred_drv_img"] = pred_drv_img          # (B, 3, H, W), [-1,1]
        outputs["pred_drv_mask"] = pred_drv_mask        # (B, 1, H, W), [0,1]
        outputs["drv_depth_img"] = drv_depth_img        # visualization only

        # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> training only (self reconstruction) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # if is_train:
        #     src_ind_img, _ = verts_to_proj_inds(src_trans_verts_aug, h, w)      # (B, H, W), val{0, 1, ..., V}
        #     # Depth Image of source (all vertices)
        #     src_depth_img = verts_feature_assign(src_ind_img, src_trans_verts_aug[:, :, 2:], pad_val=-2)          # (B, 1, H, W)
        #     # Vtx feature of driving (all vertices)
        #     src_vtx_feat_image = verts_feature_assign(src_ind_img, vtx_descriptor, pad_val=0)          # (B, C, H, W)

        #     # U-Net rendering 
        #     pred_src = self.unet_generator(torch.cat([src_depth_img, src_vtx_feat_image], dim=1))   # (B, 1+C, H, W) -> (B, 3, H, W)
        #     pred_src_img = torch.tanh(pred_src[:, :3, :, :])             # (B, 3, H=256, W=256), val:[-1, 1]
        #     pred_src_mask = torch.sigmoid(pred_src[:, 3:, :, :])            # (B, 1, H=256, W=256), val: (0, 1)
        #     outputs["pred_src_img"] = pred_src_img
        #     outputs["pred_src_mask"] = pred_src_mask

        return outputs
    

    def flame_coef_generation(self, crop_src_img, src_img, src_tform, hair_deform=True, shape=None, exp=None,):
        # 
        b, _, H, W = src_img.shape       # (256, 256)

        outputs = {}
        with torch.no_grad():
            # 0) pre-trained face segmentation (include hair + shoulder)
            self.face_parsing.eval()        # !!!!

            # source face mask
            src_mask_gt, _ = self.face_parsing(src_img)    # (B, C=19, H, W) detach
            src_mask = src_mask_gt.argmax(1)               # (B, H, W)
            src_mask = (src_mask > 0 ) + 0          
            src_mask = src_mask.unsqueeze(1).float()       # (B, 1, H, W)
            outputs["src_mask_gt"] = src_mask

            # 1) DECA Encoder forward 
            # src image
            self.deca.eval()        # !!!!
            src_codedict = self.deca.encode(crop_src_img, use_detail=False)   
            src_outputs = self.deca.decode(src_codedict, rendering=False, vis_lmk=False, return_vis=False, use_detail=False)
            src_verts = src_outputs["verts"]   # (B, V, 3) in canonical space

            # get drv codedict from 
            # drv codedict (shape, exp, )
            drv_codedict = copy.deepcopy(src_codedict)
            drv_codedict["shape"] = ...
            drv_outputs = self.deca.decode(drv_codedict, rendering=False, vis_lmk=False, return_vis=False, use_detail=False)
            drv_verts = drv_outputs["verts"]   # (B, V, 3) in canonical space

            drv_verts = src_verts
            # 2) Add deformation to driving image
            if hair_deform:
                vtx_deform = self.forward_vtx_deform(src_img)       # (B, V, 3)
                src_verts_aug = src_verts + vtx_deform              # (B, V, 3) in canonical space, with hair and shoulder augmentation
                drv_verts_aug = drv_verts + vtx_deform              # (B, V, 3) in canonical space, with hair and shoulder augmentation
            else:
                src_verts_aug = src_verts              # (B, V, 3) in canonical space without hair and shoulder augmentation
                drv_verts_aug = drv_verts              # (B, V, 3) in canonical space without hair and shoulder augmentation 

            # project vertices in canonical space to 2D image space
            src_trans_verts_aug = self.verts_aug_transfer(src_verts_aug, src_codedict["cam"], src_tform, H, W)  # (B, V, 3), val: (u,v,d),[-1,1]
            drv_trans_verts_aug = self.verts_aug_transfer(drv_verts_aug, drv_codedict["cam"], drv_tform, H, W)  # (B, V, 3)

        # 3) Vertices Feature Transformer 
        vtx_descriptor = self.foward_vtx_feat(src_img, src_mask, src_trans_verts_aug)  # (B, V=5023, C)

        # 4) Projection for Driven Image 
        drv_ind_img, _ = verts_to_proj_inds(drv_trans_verts_aug, H, W)      # (B, H, W), val: {0,1,..,V}
        # Depth Image of driving
        drv_depth_img = verts_feature_assign(drv_ind_img, drv_trans_verts_aug[:, :, 2:], pad_val=-2) # (B, 1, H, W)
        # Vtx feature of driving
        drv_vtx_feat_image = verts_feature_assign(drv_ind_img, vtx_descriptor, pad_val=0)            # (B, C, H, W)

        # 5) U-Net rendering
        x = torch.cat([drv_depth_img, drv_vtx_feat_image], dim=1)        # (B, 1+C, H, W)
        pred_drv = self.unet_generator(x)
        pred_drv_img = torch.tanh(pred_drv[:, :3, :, :])             # (B, 3, H=256, W=256), val: [-1, 1]
        pred_drv_mask = torch.sigmoid(pred_drv[:, 3:, :, :])            # (B, 1, H=256, W=256), val: (0, 1)

        outputs["pred_drv_img"] = pred_drv_img          # (B, 3, H, W), [-1,1]
        outputs["pred_drv_mask"] = pred_drv_mask        # (B, 1, H, W), [0,1]
        outputs["drv_depth_img"] = drv_depth_img        # visualization only
        return outputs
