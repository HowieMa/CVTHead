from dataset.video_data import FramesDataset, DatasetRepeater
from models.cvthead import CVTHead
from models.discriminator import MultiScaleDiscriminator
import dataset

from utils.trainer import Trainer
from utils.checkpoint import Checkpoint
from utils.common import init_ddp
from utils.logger import set_logger

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

import os
import argparse
import time, datetime
import yaml
import pickle
import logging 


def main(args):

    # >>>>>>>>>>>>>>>>> All Hyper Parameters >>>>>>>>>>>>>>>>>
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.CLoader)

    out_dir = cfg["exp"]
    os.makedirs(out_dir, exist_ok=True)
    set_logger(os.path.join(out_dir, 'train_stage1.log'))

    # >>>>>>>>>>>>>>>>> Distributed Data Parallel set up >>>>>>>>>>>>>>>>>
    rank, world_size = init_ddp()
    device = torch.device(f"cuda:{rank}")

    # >>>>>>>>>>>>>>>>> Initialize datasets >>>>>>>>>>>>>>>>>
    data_path = cfg["data"]["path"]
    meta_path = cfg["data"]["meta"]
    batch_size = cfg['training']['batch_size'] // world_size

    # voxceleb1
    train_dataset = FramesDataset(root_dir=data_path, meta_dir=meta_path, id_sampling=False, is_train=True)
    eval_dataset = FramesDataset(root_dir=data_path, meta_dir=meta_path, id_sampling=False, is_train=False)

    logging.info("--- Total train {}".format(len(train_dataset)))
    logging.info("--- Total eval {}".format(len(eval_dataset)) )

    num_workers = cfg['training']['num_workers'] if 'num_workers' in cfg['training'] else 1
    logging.info(f'--- Using {num_workers} workers per process for data loading.')

    # Initialize data loaders
    train_sampler = val_sampler = None
    shuffle = False

    # DDP
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=False)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset, shuffle=True, drop_last=False)
    else:
        shuffle = True

    # Data Loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False,
        sampler=train_sampler, shuffle=shuffle,
        worker_init_fn=dataset.worker_init_fn, persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, num_workers=num_workers, 
        sampler=val_sampler, shuffle=shuffle,
        pin_memory=False, worker_init_fn=dataset.worker_init_fn, persistent_workers=True)

    # >>>>>>>>>>>>>>>>> Model >>>>>>>>>>>>>>>>>
    generator = CVTHead()                                        # cpu model 
    discriminator = MultiScaleDiscriminator(scales=[1])             # cpu model

    generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
    discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    if world_size > 1:
        generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[rank])
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[rank])
        G_without_ddp = generator.module
        D_without_ddp = discriminator.module
    else:
        G_without_ddp = generator
        D_without_ddp = discriminator

    logging.info(f'-- Number of parameters (G): {sum(p.numel() for p in generator.parameters())/1e6} M\n')
    logging.info(f'-- Number of parameters (D): {sum(p.numel() for p in discriminator.parameters())/1e6} M\n')


    # >>>>>>>>>>>>>>>>> Optimizer >>>>>>>>>>>>>>>>>
    optimizer_G = optim.Adam(filter(lambda x: x.requires_grad, generator.parameters()), 
                                lr=float(cfg["training"]["lr_G"]), betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=float(cfg["training"]["lr_D"]), betas=(0.5, 0.999))


    # Intialize training
    trainer = Trainer(generator, discriminator, optimizer_G, optimizer_D, cfg, device, out_dir, stage=1)

    checkpoint_G = Checkpoint(out_dir, device=device, model=G_without_ddp, optimizer=optimizer_G)
    checkpoint_D = Checkpoint(out_dir, device=device, model=D_without_ddp, optimizer=optimizer_D)


    # >>>>>>>>>>>>>>>>> Training Set up >>>>>>>>>>>>>>>>>
    start_epoch = 0
    epochs = cfg["training"]["epochs"]
    print_every = cfg['training']['print_every']
    
    model_selection_metric = cfg['training']['model_selection_metric']
    if cfg['training']['model_selection_mode'] == 'maximize':
        model_selection_sign = 1
    elif cfg['training']['model_selection_mode'] == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError('model_selection_mode must be either maximize or minimize.')

    # >>>>>>>>>>>>>>>>> Resume >>>>>>>>>>>>>>>>>
    try:
        load_dict_G = checkpoint_G.load("model_G.pt")
    except FileNotFoundError:
        load_dict_G = dict()
    try:
        load_dict_D = checkpoint_D.load("model_D.pt")
    except FileNotFoundError:
        load_dict_D = dict()

    start_epoch = load_dict_G.get('epoch', 0)
    time_elapsed = load_dict_G.get('t', 0.)           # total training time
    metric_val_best = load_dict_G.get(
        'loss_val_best', -model_selection_sign * np.inf)
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Scheduler >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    scheduler_G = MultiStepLR(optimizer_G, cfg["training"]['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    scheduler_D = MultiStepLR(optimizer_D, cfg["training"]['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)

    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run evaluation at the beginning >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if args.evalnow:
        logging.info('Evaluating at initialization...')
        eval_dict = trainer.evaluate(val_loader, start_epoch)
        args.evalnow = False

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training loop >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    for epoch in range(start_epoch + 1, epochs + 1):
        lr = scheduler_G.get_last_lr()[0]

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        logging.info("\n--- Training EPOCH {} ---".format(epoch))
        avg_loss_G, avg_loss_D, time_elapsed = trainer.train_one_epoch(train_loader, epoch, lr, print_every, time_elapsed)
        logging.info("--- Epoch Summary: Training avg G loss:{}. avg D Loss: {}".format(avg_loss_G, avg_loss_D) )
        

        # >>>>>>>>>>>>>>>>>>>>> save model after one epoch >>>>>>>>>>>>>>>>>>>>>>
        if rank == 0:
            checkpoint_scalars = {'epoch': epoch,
                                    't': time_elapsed,
                                    'loss_val_best': metric_val_best}
            checkpoint_G.save('model_G.pt', **checkpoint_scalars)
            checkpoint_D.save("model_D.pt", **checkpoint_scalars)

        # >>>>>>>>>>>>>>>>>>>>> Evaluation after one epochs >>>>>>>>>>>>>>>>>>>>>
        if epoch % cfg["training"]["eval_per_epoch"] == 0:
            logging.info("\n--- Evaluation ---")
            eval_dict = trainer.evaluate(val_loader, epoch)
            metric_val = eval_dict[model_selection_metric]

            # best model
            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                if rank == 0:
                    checkpoint_scalars['loss_val_best'] = metric_val_best
                    logging.info(f'New best model (loss {metric_val_best:.6f})')
                    checkpoint_G.save('model_best_G.pt', **checkpoint_scalars)

        scheduler_G.step()
        scheduler_D.step()


if __name__ == '__main__':
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Arguments >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser = argparse.ArgumentParser(
        description='CVTHead Training.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--evalnow', action='store_true', help='Run evaluation on startup.')
    parser.add_argument('--stage', type=int, default=1, help='Training stage')

    args = parser.parse_args()

    main(args)
