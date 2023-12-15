import numpy as np
from torch.utils.data import Dataset
import torch

import glob
import json
import pickle
import os 

from skimage.transform import estimate_transform, warp, resize, rescale
from skimage import io, img_as_float32
from imageio import mimread
import random 
from PIL import Image


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]



def get_deca_tform(kpt, scale=1.25, resolution_inp=224):
    # (68, 2) keypoint coordinates in original image scale 

    left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
    top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
    
    old_size = (right - left + bottom - top) / 2 * 1.1
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    
    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
    DST_PTS = np.array([[0,0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)  # linear transformation (3, 3), [0, 0, 1]   from original image to DECA cropped image
    return tform


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, meta_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True):
        
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)

        self.id_sampling = id_sampling

        # >>>>>>>>>>>>>>>>>>>>>>>> train / test split >>>>>>>>>>>>>>>>>>>>>>>>>>>> 
        if is_train:
            # 17293 videos
            valid_video_frames = json.load(open( os.path.join(meta_dir, "train_imgs.json") ))       # {"id10280#NXjT3732Ekg#001093#001192": [1,2,3]}
            if id_sampling:
                # 
                subject_videos = {}
                for video in valid_video_frames.keys():
                    sub_id = os.path.basename(video).split('#')[0]
                    if sub_id not in subject_videos:
                        subject_videos[sub_id] = [video]
                    else:
                        subject_videos[sub_id].append(video)
                
                train_videos = list(subject_videos.keys())           # subject ID only
                self.subject_videos = subject_videos                 # 422 subjects


            else:
                train_videos = list(valid_video_frames.keys())     # all video clips
            self.videos = train_videos
        
        else:
            valid_video_frames = json.load(open( os.path.join(meta_dir, "test_imgs.json") ))
            test_videos = list(valid_video_frames.keys())
            
            self.videos = test_videos

        # >>>>>>>>>>>>>>>>>>>>>>> get valid video frames with landmarks >>>>>>>>>>>>>>>>>>>
        self.valid_video_frames = valid_video_frames

        self.root_dir = os.path.join(root_dir, 'train' if is_train else 'test')
        self.meta_dir = os.path.join(meta_dir, 'train' if is_train else 'test')
        # self.mask_dir = os.path.join(root_dir.replace("vox_video", "vox_masks"), "train" if is_train else "test")

        self.is_train = is_train
        self.deca_size = 224
        

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]     # subject ID  id11248
            subject_videos = self.subject_videos[name]      # []
            video_id = random.choice(subject_videos)
            path = os.path.join(self.root_dir, video_id + ".mp4")
            # path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))     # random choose one video for the subject
            # video-preprocessing/vox_video/train/id11248#yiNkInm9OKQ#000286#000613.mp4

        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name + ".mp4")

        video_name = os.path.basename(path)     # id11248#yiNkInm9OKQ#000286#000613.mp4

        # >>>>>>>>>>>>>>>>>>>>>>>>> load mp4 video file >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # video frames with GT landmarks 
        valid_frames = self.valid_video_frames[video_name.split(".")[0]]      # [1,2,3,]

        #  mp4 video file
        video_array = np.array(mimread(path, memtest=False))       # (T, H, W, 3)
        
        if self.is_train:
            # random sample 2 frames in training
            frame_idx = random.sample(valid_frames, 2)
        else:
            total = 10
            step = len(valid_frames) // total
            frame_idx = valid_frames[: total * step : step]
        frame_idx.sort()

        video_array = video_array[frame_idx]        # (2 or T, H, W, 3), val: 0-255
        video_array = video_array / 255.            # val: 0-1

        # >>>>>>>>>>>>>>>>>>>>>>>>> load meta data (2D landmarks) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        f = open(os.path.join(self.meta_dir, video_name.split('.')[0] + '.pkl'), 'rb')
        video_meta = pickle.load(f)
        f.close()

        landmarks, tforms, crop_images, crop_lmks = [], [], [], []
        for idx, vid in enumerate(frame_idx):
            landmark = video_meta[vid]['ldmk']      # (68, 2)
            tform = get_deca_tform(landmark)        # (3, 3)
            # crop image to DECA format
            crop_image = warp(video_array[idx], tform.inverse, output_shape=(self.deca_size, self.deca_size))   # (224, 224, 3), val:[0, 1]

            # 
            cropped_lmk = np.dot(tform.params, np.hstack([landmark, np.ones([landmark.shape[0],1])]).T).T       # landmark in (224, 224)
            cropped_lmk[:,:2] = cropped_lmk[:,:2] / self.deca_size * 2  - 1       # normalized to [-1, 1]

            landmarks.append(landmark)
            tforms.append(tform.params)
            crop_images.append(crop_image)
            crop_lmks.append(cropped_lmk)

        # to tensor
        video_array = torch.from_numpy(video_array).float()     # tensor, (T', H, W, 3), val: [0, 1]
        video_array = video_array.permute(0, 3, 1, 2)           # (T', H, W, 3) --> (T', 3, H, W)
        video_array = (video_array - 0.5) / 0.5                 # (T', 3, H, W), [-1, 1]

        crop_images = torch.tensor(np.asarray(crop_images)).float() # (T', 224, 224, 3), val:0-1
        crop_images = crop_images.permute(0, 3, 1, 2)             # (T', 3, 224, 224), val:0-1

        landmarks = torch.tensor(np.asarray(landmarks)).float()     # (T', 68, 2)   in (256, 256)
        tforms = torch.tensor(np.asarray(tforms)).float()           # (T', 3, 3)
        crop_lmks = torch.tensor(np.asarray(crop_lmks)).float()     # (T', 68, 2)   normalized to [-1, 1]

        # masks = torch.stack(masks, dim=0).float()             # (T', 1, 256, 256)
        # >>>>>>>>>>>>>>>>>>>>>>>>> output >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        out = {}
        out['name'] = video_name
        # source image
        out['src_img'] = video_array[0]         # (3, 256, 256), val: [-1, 1]

        out['src_lmk'] = landmarks[0]           # (68, 2), in (256, 256)
        out["src_tform"] = tforms[0]            # (3, 3)
        out['src_img_crop'] = crop_images[0]    # (3, 224, 224), val:[0, 1]
        out['src_lmk_crop'] = crop_lmks[0]      # (68, 2), in [-1, 1]

        if self.is_train:
            out['drv_img'] = video_array[1]         # (3, 256, 256), val: [-1, 1]
            out['drv_lmk'] = landmarks[1]           # (68, 2), in (256, 256)
            out["drv_tform"] = tforms[1]            # (3, 3)
            out['drv_img_crop'] = crop_images[1]    # (3, 224, 224), val: [0, 1]
            out['drv_lmk_crop'] = crop_lmks[1]      # (68, 2), in [-1, 1]

        else:
            out['drv_img'] = video_array[1:]         # (T-1, 3, 256, 256), val:0-1
            out['drv_img_crop'] = crop_images[1:]    # (T-1, 3, 224, 224), val:0-1
            out["drv_tform"] = tforms[1:]            # (3, 3)

        return out
