from dataset.video_data import get_deca_tform
from PIL import Image
from models.cvthead import CVTHead

import face_alignment
from skimage.transform import estimate_transform, warp, resize, rescale
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

import os
import argparse

import yaml
import pickle 
import logging 


def preprocess_image(img_pth, fa, device="cuda"):
    img = Image.open(img_pth)
    img = img.resize((256, 256))

    img_npy = np.array(img)     # (H=256,W=256,3), val:0-255
    landmark = fa.get_landmarks(img_npy)[0]
    tform = get_deca_tform(landmark)    # (3,3)
          
    img_npy = img_npy / 255.       # (H,W,3), val: 0-1
    crop_image = warp(img_npy, tform.inverse, output_shape=(224, 224))   # (224, 224, 3), val:[0, 1]
 
    img_tensor = torch.from_numpy(img_npy).float()   # tensor, (H, W, 3), val: [0, 1]
    img_tensor = img_tensor.permute(2, 0, 1)    # (H, W, 3) --> (3, H, W)
    img_tensor = (img_tensor - 0.5) / 0.5      # (3,H,W), [-1, 1]

    crop_image = torch.tensor(np.asarray(crop_image)).float() # (224, 224, 3), val:0-1
    crop_image = crop_image.permute(2, 0, 1)             # (3, 224, 224), val:0-1
    tform = torch.tensor(np.asarray(tform)).float()     # (3,3)

    img_tensor = img_tensor.unsqueeze(0).to(device)    # (1,3,256,256)
    crop_image = crop_image.unsqueeze(0).to(device)    # (1,3,224,224)
    tform = tform.unsqueeze(0).to(device)              # (1,3,3)      
    return img_tensor, crop_image, tform


def driven_by_face(model, src_pth, drv_pth, out_pth, device, softmask=True):
    # face landmark detector
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)

    src_img, src_img_crop, src_tform = preprocess_image(src_pth, fa, device)
    drv_img, drv_img_crop, drv_tform = preprocess_image(drv_pth, fa, device)
    
    with torch.no_grad():
        outputs = model(src_img_crop, drv_img_crop, src_img, drv_img, src_tform, drv_tform, is_train=False, is_cross_id=True)
        predict_img = outputs["pred_drv_img"]   # (1,3,256,256), tensor, val:[-1,1]
        predict_mask = outputs["pred_drv_mask"] # (1,256,256), tensor, val:[0, 1], soft mask

        # visualize
        predict_img = 0.5 * (predict_img + 1)
        predict_img = predict_img[0].permute(1,2,0).cpu().numpy()   # (256,256,3), npy  
        predict_mask = predict_mask[0].permute(1,2,0).cpu().numpy()   # (256,256,1), npy
        if not softmask:
            predict_mask = (predict_mask > 0.6).float()     # (256,256,1), npy

        # apply mask
        predict_img = predict_img * predict_mask + (1 - predict_mask)  # apply mask to predicted image, val:[0, 1], npy
        predict_img = (predict_img * 255).astype(np.uint8)
        predict_img = Image.fromarray(predict_img)
        predict_img.save(out_pth)


def main(args):
    device = "cuda"

    # >>>>>>>>>>>>>>>>> Model >>>>>>>>>>>>>>>>>
    model = CVTHead()                                        # cpu model 
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)        # gpu model

    # load pre-trained weights
    ckpt = torch.load(args.ckpt_pth, map_location="cpu")["model"]
    model.load_state_dict(ckpt, strict=False)
    print(f'-- Number of parameters (G): {sum(p.numel() for p in model.parameters())/1e6} M\n')

    driven_by_face(model, args.src_pth, args.drv_pth, args.out_pth, device, softmask=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CVTHead Inference')
    parser.add_argument('--src_pth', type=str, default="examples/1.png")
    parser.add_argument('--drv_pth', type=str, default="examples/2.png")
    parser.add_argument('--out_pth', type=str, default="examples/output.png")
    parser.add_argument('--ckpt_pth', type=str, default="data/cvthead.pt")
    args = parser.parse_args()

    main(args)
