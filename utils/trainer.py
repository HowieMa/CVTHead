import torch
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import os
import time, datetime
import logging

from utils.loss import MaskedCriterion
from utils.metrics import LPIPS, PSNR, SSIM, MS_SSIM
from utils.common import  get_rank
from utils.visualize import draw_visualization_grid
from utils.meter import MetricLogger


from losses import VGGFace2Loss, PerceptualLossWrapper, SegmentationLoss

from collections import OrderedDict


class Trainer:
    def __init__(self, generator, discriminator, optimizer_G, optimizer_D, cfg, device, out_dir, stage):
        logging.info("\n ---- Set Up Trainer ----\n")
        self.net_G = generator
        self.net_D = discriminator

        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D

        self.config = cfg
        self.device = device
        self.out_dir = out_dir
        self.stage = stage
        logging.info("\n*********** Trainer Stage ***********{}".format(stage))

        # >>>>>>>>>>>>>>>>>>> Training Loss >>>>>>>>>>>>>>>>
        self._masked_L1 = MaskedCriterion(torch.nn.L1Loss(reduction="none"))

        self.vgg19_loss = PerceptualLossWrapper(num_scales=1, use_gpu=True, use_fp16=False)

        self.id_loss = VGGFace2Loss(pretrained_model="data/resnet50_scratch_weight.pth")

        self.seg_loss = SegmentationLoss(loss_type="dice")
        self._mse_loss = nn.MSELoss(reduction="mean")

        self.weight = {
            "l1": 1.0,
            "vgg19": 1.0,
            "id": 0.1, 
            "gan_score": 0.01,  # GAN loss
            "gan_match": 0.1,   # GAN loss
            "mask": 1.0, 
        }

        # >>>>>>>>>>>>>>>>>>>>>>>> Eval Metric >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self._metrics = ["L1", "PSNR", "MS_SSIM", "LPIPS", "SSIM"]
        self.ssim = SSIM(data_range=1, size_average=True, channel=3)
        self.ms_ssim = MS_SSIM(data_range=1, size_average=True, channel=3)
        self.psnr = PSNR()
        self.lpips = LPIPS()
        self.l1_mean = nn.L1Loss(reduction="mean")
        logging.info("\n ---- Finish Trainer Set Up ...\n\n")

    """
    Train Function
    """

    def train_one_epoch(self, data_loader, epoch, lr, print_every, time_elapsed):
        self.net_G.train()
        avg_loss_G, avg_loss_D = [], []
        for step, batch in enumerate(data_loader):

            t0 = time.perf_counter()
            log_dict, lossG, lossD = self.train_step(batch)              # one iteration
            time_elapsed += time.perf_counter() - t0
            time_elapsed_str = str(datetime.timedelta(seconds=time_elapsed))
            log_dict["lr-G"] = lr

            # >>>>>>>>>>>>>> Training Log >>>>>>>>>>>>>>>>>>
            if print_every > 0 and (step % print_every) == 0 and get_rank() == 0:
                log_str = ['{}={:f}'.format(k, v) for k, v in log_dict.items()]
                log_str = ' | '.join(log_str)
                log_str = 't=%s [Epoch %02d] step=%03d/%3d | '% (time_elapsed_str, epoch, step, len(data_loader)) + log_str
                logging.info(log_str)
            
            avg_loss_G.append(lossG)
            avg_loss_D.append(lossD)
        
        return sum(avg_loss_G)/len(avg_loss_G), sum(avg_loss_D)/len(avg_loss_D), time_elapsed

    def train_step(self, data):
        # >>>>>>>>>>>>>>>>>>>>>>> Update G >>>>>>>>>>>>>>>>>>>>>>>>>>
        self.optimizer_G.zero_grad()
        loss_G, loss_terms_G, outputs, src_img, drv_img = self.forward_generator(data)
        loss_G.backward()
        self.optimizer_G.step()
        
        loss_terms = {k: v.item() for k, v in loss_terms_G.items()}

        # >>>>>>>>>>>>>>>>>>>>>>> Update D >>>>>>>>>>>>>>>>>>>>>>>>>>
        self.optimizer_D.zero_grad()
        loss_D = self.discriminator_loss(outputs["pred_drv_img"].detach().clone(), 
                                        drv_img, 
                                        outputs["drv_mask_gt"].detach())
        loss_D.backward()
        self.optimizer_D.step()
        loss_terms["Dis"] = loss_D.item()
        
        # >>>>>>>>>>>>>>>>>>>>>>> Total Loss >>>>>>>>>>>>>>>>>>>>>>>>>
        return loss_terms, loss_G.item(), loss_D

    def image_loss(self, pred_img, target_img, pred_mask, target_mask, tt="_src"):
        loss_terms = {}

        if self.stage == 1:
            mask = pred_mask.clone().detach()       # use predicted mask !!! softmask
        else:
            mask = target_mask

        # mask = target_mask      # !!!!!
        random_bg_color = torch.rand(target_mask.shape[0], 3, 1, 1, dtype=target_mask.dtype, device=target_mask.device)
        
        pred_img_n = (pred_img + 1) * 0.5           # [-1,1] -> [0, 1]
        pred_img_n = pred_img_n * mask + random_bg_color * (1 - mask) # [0, 1] add random bg
        pred_img = 2 * pred_img_n - 1               # [0, 1] -> [-1, 1] with random bg 

        target_img_n = (target_img + 1) * 0.5       # [-1,1] -> [0, 1]
        target_img_n = target_img_n * mask + random_bg_color * (1 - mask)     # [0, 1] add random bg
        target_img = 2 * target_img_n - 1           # [0, 1] -> [-1, 1] with random bg

        # >>>>>>>>>>>>>>>>>>>>>>>> Loss: Photo >>>>>>>>>>>>>>>>>>>>>>>>
        # Photo loss (1.0) --> 0.08~0.10
        loss_terms["l1" + tt] = self._masked_L1(pred_img, target_img, mask) * self.weight["l1"]  # [-1, 1]
        # VGG19 perceptual loss (1.0)   --> 0.6
        loss_terms["vgg19" + tt] = self.vgg19_loss.forward(pred_img_n, target_img_n) * self.weight["vgg19"]
        # VGGface identity loss (0.1)  
        loss_terms["id" + tt] = self.id_loss(pred_img_n, target_img_n) * self.weight["id"]
        # Mask Dice loss (1.0)
        loss_terms["mask" + tt] = self.seg_loss(pred_mask, target_mask) * self.weight["mask"]

        # >>>>>>>>>>>>>>>>>>>>>>>>>> GAN Loss >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if "drv" in tt and self.stage == 2:
            pred_D_feat, pred_D_scores = self.net_D(pred_img)   # List [ (B, C, H, W), .., (B, C, H, W) ] , [B,1, h, w]
            with torch.no_grad():
                real_D_feat, _ = self.net_D(target_img)                   # [ (B, C, H, W), .., (B, C, H, W) ] , [B,1, h, w]

            # GAN score loss (0.1)
            loss_gan = 0
            for fake_scores_net in pred_D_scores:
                loss_gan -= fake_scores_net.mean()      # should be close to 1 (the large, the better)
            loss_gan /= len(pred_D_scores)
            loss_terms["gan_score" + tt] = loss_gan * self.weight["gan_score"]

            # GAN feature matching loss (1.0)
            loss_gan_match = 0
            for real_feat_net, pred_feat_net in zip(real_D_feat, pred_D_feat):      # only 1 
                loss_net = 0
                for real_feat_net_i, pred_feat_net_i in zip(real_feat_net, pred_feat_net):
                    loss_net += F.l1_loss(pred_feat_net_i, real_feat_net_i)
                loss_net = loss_net / len(real_feat_net)
                loss_gan_match += loss_net
            loss_gan_match = loss_gan_match / len(real_D_feat)
            loss_terms["gan_match" + tt] = loss_gan_match * self.weight["gan_match"]
        return loss_terms
    
    def forward_generator(self, data):
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>> Data Preprocess >>>>>>>>>>>>>>>>>>>>>>>>>
        device = self.device

        src_img = data.get("src_img").to(device)                # (B, 3, 256, 256), val:[-1, 1]
        src_img_crop = data.get("src_img_crop").to(device)      # (B, 3, 224, 224), val:[0, 1]
        src_tform = data.get("src_tform").to(device)            # (B, 3, 3)

        drv_img = data.get("drv_img").to(device)                # (B, 3, 256, 256), val:[-1, 1]
        drv_img_crop = data.get("drv_img_crop").to(device)      # (B, 3, 224, 224), val:[0, 1]
        drv_tform = data.get("drv_tform").to(device)            # (B, 3, 3)

        # >>>>>>>>>>>>>>>>>>>>>>>> network forward >>>>>>>>>>>>>>>>>>>>>>>>
        outputs = self.net_G(src_img_crop, drv_img_crop, src_img, drv_img, src_tform, drv_tform, is_train=False)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Loss Calculation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        loss_terms = {}
        # loss on drv image
        loss_drv = self.image_loss(outputs["pred_drv_img"], drv_img, outputs["pred_drv_mask"], outputs["drv_mask_gt"], tt="_drv")
        loss_terms.update(loss_drv)

        # >>>>>>>>>>>>>>>>>>>>>>> Total Loss >>>>>>>>>>>>>>>>>>>>>>
        loss = 0
        for k, v in loss_terms.items():
            loss += v

        return loss, loss_terms, outputs, src_img, drv_img

    def discriminator_loss(self, pred_img, target_img, mask):
        random_bg_color = torch.rand(pred_img.shape[0], 3, 1, 1, dtype=pred_img.dtype, device=pred_img.device)

        pred_img = pred_img.detach().clone()            
        pred_img_n = (pred_img + 1) * 0.5                   # [-1,1] -> [0, 1]
        pred_img_n = pred_img_n * mask + random_bg_color * (1 - mask)       # [0, 1]
        pred_img = 2 * pred_img_n - 1                       # [0, 1] -> [-1, 1] with random bg

        target_img_n = (target_img + 1) * 0.5       # [-1,1] -> [0, 1]
        target_img_n = target_img_n * mask + random_bg_color * (1 - mask)     # [0, 1]
        target_img = 2 * target_img_n - 1         # [0, 1] -> [-1, 1] with random bg

        _, pred_D_scores = self.net_D(pred_img)      # list (B, 1, h, w)  should be close to 0
        _, real_D_scores = self.net_D(target_img)       # list (B, 1, h, w)  should be close to 1

        # >>>>>>>>>>>>>>>>>>> Hinge Loss >>>>>>>>>>>>>>>>>>>>>>>>>>>>
        loss_real = 0
        for real_scores_net in real_D_scores:
            loss_real += torch.relu(1.0 - real_scores_net).mean()     # real score, close to 1
        loss_real /= len(real_D_scores)

        loss_fake = 0
        for fake_scores_net in pred_D_scores:
            loss_fake += torch.relu(1.0 + fake_scores_net).mean()     # fake score, close to -1      
        loss_fake /= len(pred_D_scores)

        loss = loss_real + loss_fake
        return loss 


    """
    Evaluation
    """
    @torch.no_grad()
    def evaluate(self, val_loader, train_step):
        device = self.device
        self.net_G.eval()
        metric_logger = MetricLogger(delimiter="  ")    # sync logs among different GPUs 
        loader = val_loader
        for idx, data in enumerate(loader):
            # >>>>>>>>>>>>>>>>>>>>>>>> network forward >>>>>>>>>>>>>>>>>>>>>>>>
            # src
            src_img = data.get("src_img").to(device)                 # (B, 3, 256, 256), val:[-1,1]
            src_img_crop = data.get("src_img_crop").to(device)       # (B, 3, 224, 224), val:[0,1]
            src_tform = data.get("src_tform").to(device)             # (B, 3, 3)
            # drv
            drv_imgs = data.get("drv_img").to(device)           # (B, T, 3, 256, 256), val:[-1,1]
            drv_imgs_crop = data.get("drv_img_crop")            # (B, T, 3, 224, 224), val:[0,1]
            drv_tforms = data.get("drv_tform")                  # (B, T, 3, 3)

            B = src_img.shape[0]
            pred_drv_imgs, drv_depth_imgs, drv_masks = [], [], []
            for t in range(drv_imgs.shape[1]):
                drv_img_crop = drv_imgs_crop[:, t].to(device)
                drv_tform = drv_tforms[:, t].to(device)
                outputs = self.net_G(src_img_crop, drv_img_crop, src_img, drv_imgs[:, t], src_tform, drv_tform, is_train=False) 

                pred_drv_imgs.append(outputs["pred_drv_img"])       # (B, 3, 256, 256), val:[-1,1]
                drv_depth_imgs.append(outputs["drv_depth_img"])     # (B, 3, 256, 256)
                # drv_masks.append(outputs["drv_mask_gt"])          # (B, 3, 256, 256), hard mask
                drv_masks.append(outputs["pred_drv_mask"])          # val: [0, 1], soft mask

            pred_drv_imgs = torch.stack(pred_drv_imgs, dim=1)       # (B, T, 3, 256, 256), [-1,1]
            drv_depth_imgs = torch.stack(drv_depth_imgs, dim=1)     # (B, T, 3, 256, 256)
            drv_masks = torch.stack(drv_masks, dim=1)               # (B, T, 1, 256, 256), [0,1] soft mask
            
            drv_masks = (drv_masks > 0.6).float()                   # (B, T, 1, 256, 256), {0,1} hard mask

            # >>>>>>>>>>>>>>>>>>>>>>>> Metric calculation >>>>>>>>>>>>>>>>>>>>>>>>
            mask = drv_masks.flatten(0, 1)                   # (B * T, 1, 256, 256) val:[0,1] soft mask
            
            # normalize from [-1,1] to [0,1]
            pred = 0.5 * pred_drv_imgs.flatten(0, 1) + 0.5      # (B * T, 3, 256, 256), val:[0, 1] RGB
            pred = pred * mask + (1 - mask)                     # (B * T, 3, 256, 256), val:[0, 1] RGB + white bg
            gt = 0.5 * drv_imgs.flatten(0, 1)+ 0.5              # (B * T, 3, 256, 256), val:[0, 1]
            gt = gt * mask + (1 - mask)                         # (B * T, 3, 256, 256), val:[0, 1] RGB + white bg

            l1 = self.l1_mean(pred, gt)
   
            metric_logger.meters['L1'].update(self.l1_mean(pred, gt).cpu().item(), n=B)
            metric_logger.meters['MSE'].update(((pred - gt) ** 2).mean().cpu().item(), n=B)
            metric_logger.meters['SSIM'].update(self.ssim(pred, gt).mean().cpu().item(), n=B)
            metric_logger.meters['LPIPS'].update(self.lpips(pred, gt).cpu().item(), n=B)
            metric_logger.meters['MS_SSIM'].update(self.ms_ssim(pred, gt).mean().cpu().item(), n=B)

            # Visualization 
            if idx == 0:
                src_img = src_img.permute(0, 2, 3, 1).cpu().numpy()                     # (B, H, W, 3), val:[-1,1]
                pred_imgs = pred_drv_imgs.permute(0, 1, 3, 4, 2).cpu().numpy()          # (B, T, 256, 256, 3), val:[-1,1]
                gt_imgs = drv_imgs.permute(0, 1, 3, 4, 2).cpu().numpy()                 # (B, T, 256, 256, 3), val:[-1,1]
                drv_depth_imgs = drv_depth_imgs.permute(0, 1, 3, 4, 2).cpu().numpy()    # (B, T, 256, 256, 1)

                # normalize from [-1,1] to [0, 1]
                src_img = 0.5 * (src_img + 1)
                pred_imgs = 0.5 * (pred_imgs + 1)
                gt_imgs = 0.5 * (gt_imgs + 1)

                drv_masks = (drv_masks > 0.6).float()                         # (B, T, 256, 256, 1), hard mask, val:{0,1}
                drv_masks = drv_masks.permute(0, 1, 3, 4, 2).cpu().numpy()      # (B, T, 256, 256, 1)
                pred_imgs = pred_imgs * drv_masks + (1 - drv_masks)             # apply mask to predicted image, val:[0, 1]
                rrr = get_rank()
                columns = [("Src", src_img, "image")]
                for t in range(min(10, pred_imgs.shape[1])):
                    columns.append(("GT", gt_imgs[:, t], 'image'))
                    columns.append(("Pred", pred_imgs[:, t], 'image'))
                    columns.append(("Depth", drv_depth_imgs[:, t], "image"))
                output_img_path = os.path.join(self.out_dir, f'renders-val-stage{self.stage}-{train_step}-{rrr}')
                draw_visualization_grid(columns, output_img_path)
                logging.info("Done Visualization ...")

        # >>>>>>>>>>>>>>>>> average metric of entire val dataset >>>>>>>>>>>>>>>>> 
        metric_logger.synchronize_between_processes()
        avg_psnr = float(10 * torch.log10(1 / torch.tensor(metric_logger.MSE.global_avg)))
        metrics = {
            "L1": round(metric_logger.L1.global_avg, 4), 
            "PSNR": round(avg_psnr, 4),
            "SSIM": round(metric_logger.SSIM.global_avg, 4), 
            "LPIPS": round(metric_logger.LPIPS.global_avg, 4),
            "MS_SSIM": round(metric_logger.MS_SSIM.global_avg, 4), 
        }
        logging.info('Evaluation results:' + str(metrics))
        return metrics


