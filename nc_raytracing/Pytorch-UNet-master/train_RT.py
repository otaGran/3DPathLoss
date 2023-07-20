import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate_rt import evaluate
from unet.unet_model_rt import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from RMSLELoss import RMSLELoss

from utils.data_loading_rt import RTDataset
import matplotlib.pyplot as plt
# dir_img = Path('../res')
# dir_mask = Path('./coverage_maps')

from torch.masked import masked_tensor, as_masked_tensor

from sklearn.model_selection import train_test_split



dir_checkpoint = Path('./checkpoints/')

building_height_map_dir = os.path.abspath('../res_plane/Bl_building_npy')
terrain_height_map_dir = os.path.abspath('../res_plane/Bl_terrain_npy')
ground_truth_signal_strength_map_dir = os.path.abspath('/dev/shm/coverage_maps_data_aug_Jul7/')






def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        pathloss = False,
        pathloss_multi_modality = False,
        start_epoch = 0
):
    # 1. Create dataset

    dataset = RTDataset(building_height_map_dir, terrain_height_map_dir,ground_truth_signal_strength_map_dir, img_scale, pathloss = pathloss)
    

    
    
        

    # 2. Split into train / validation partitions
    
    
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    # This is a custom implementation of random split in order to avoid the data leak when using 
    # our data augmentation 
    ids = dataset.get_ids()
    print("# of Total Ids: ",len(ids))
    file_id_set = set([ int(ids_file_name.split("_")[0]) for ids_file_name in ids])
    print("# of Total Ori Index: ",len(file_id_set))
    train_set_ori_idx, val_set_ori_idx = train_test_split(list(file_id_set), test_size=0.2, random_state=42)
    
    print("# of Train Ori Index: ",len(train_set_ori_idx))
    print("# of Val Ori Index: ",len(val_set_ori_idx))
    
    train_set_idx = []
    val_set_idx = []
    for cur_idx, cur_ids in enumerate(ids):
        if int(cur_ids.split("_")[0]) in train_set_ori_idx:
            train_set_idx.append(cur_idx)
        else:
            val_set_idx.append(cur_idx)
            
    n_train = len(train_set_idx)
    n_val = len(val_set_idx)
    print("# of Train Index: ",len(train_set_idx))
    print("# of Val Index: ",len(val_set_idx))
    
    #Do a sanity check
    tmp_train_idx_set = set([ int(ids[tmp_idx].split("_")[0]) for tmp_idx in train_set_idx])
    tmp_val_idx_set = set([ int(ids[tmp_idx].split("_")[0]) for tmp_idx in val_set_idx])
    print(tmp_val_idx_set)
    print("Check this",set(tmp_val_idx_set).intersection(tmp_train_idx_set))
    
    train_set = torch.utils.data.Subset(dataset, train_set_idx)
    val_set = torch.utils.data.Subset(dataset, val_set_idx)
    
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
   
    # (Initialize logging)
    experiment = wandb.init(project='U-Net_RT', resume='allow', anonymous='must')
    experiment.config.update(
        {"epochs":epochs, 
             "batch_size":batch_size, 
             "learning_rate":learning_rate,
             "val_percent":val_percent, 
             "save_checkpoint":save_checkpoint, 
             "img_scale":img_scale, 
             "amp":amp, 
             "samples_size":len(dataset),
             "Pre-processing wiht Path Loss Model":pathloss_multi_modality,
             "Multi-modality with Path Loss Model":pathloss
        }
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Samples size:    {len(dataset)}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Pre-processing wiht Path Loss Model: {pathloss}
        Multi-modality with Path Loss Model: {pathloss_multi_modality}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, foreach=True)
    
    # optimizer = optim.RMsporp(model.parameters(),
    #                           lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,eps=1e-9)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img', leave=False) as pbar:
            for batch in train_loader:
                images, true_masks_cpu = batch['combined_input'], batch['ground_truth']

                # assert images.shape[1] == model.n_channels, \
                #     f'Network has been defined with {model.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks_cpu.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    coverage_map_pred = model(images)
                    # RMSE instead of MSE since the data is spread
                    # Only compute the loss on out door points
                    
                    # compute the mask of building area
                    building_mask = images.clone().detach().squeeze(1)
                    
                    building_mask = building_mask[:,0,:,:].squeeze(1)
                    #print(building_mask.size())
                    building_mask[building_mask != 0] = 1
                    building_mask = building_mask.bool()
                    #print(building_mask)
                    #print(true_masks.size())
                    
                    # Apply the mask on coverage_map_pred
                    masked_coverage_map_pred = coverage_map_pred.clone().squeeze(1)
                    #print(masked_coverage_map_pred.size())
                    #print(building_mask.size())
                    masked_coverage_map_pred[masked_coverage_map_pred < -160] = -160
                    masked_coverage_map_pred[building_mask] = true_masks.float()[building_mask]
                    
                    
                    loss = torch.sqrt(criterion(masked_coverage_map_pred.squeeze(1), true_masks.float()))



                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch + start_epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (1 * batch_size))+1
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                                            # compute the mask of building area

                        val_score = evaluate(model, val_loader, device, amp)
                        #scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        coverage_map_pred_cpu = coverage_map_pred.clone().detach().squeeze(1).cpu().numpy()
                        fig = plt.figure()
                        plt.imshow(coverage_map_pred_cpu[0],vmin=-110,vmax=0)
                        plt.colorbar()
                        
                        fig2 = plt.figure()
                        plt.imshow(true_masks_cpu[0],vmin=-110,vmax=0)
                        plt.colorbar()
                        
                        fig3 = plt.figure()
                        plt.imshow(coverage_map_pred_cpu[0])
                        plt.colorbar()
                        
                        
                        masked_coverage_map_pred_cpu = masked_coverage_map_pred.clone().detach().squeeze(1).cpu().numpy()
    
                        fig4 = plt.figure()
                        plt.imshow(masked_coverage_map_pred_cpu[0],vmin=-110,vmax=0)
                        plt.colorbar()
                
                        
                        fig5 = plt.figure()
                        plt.imshow(images[0,1].cpu())
                        plt.colorbar()
                        plt.title(batch['file_name'][0])
                        
                        
                        fig6 = plt.figure()
                        plt.imshow(images[0,2].cpu())
                        plt.colorbar()
                        #plt.title(batch['file_name'][0])
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Loss': val_score,
                                'Inputs':{
                                    'Building Height Map': wandb.Image(images[0,0].cpu()),
                                    'Path Loss Model Pred': wandb.Image(fig6),
                                    'TX Position Map': wandb.Image(fig5),
                                    'Coverage Map(Ground Truth)': wandb.Image(fig2),
                                },
                                'Outputs': {
                                    'Pred': wandb.Image(fig),
                                    'Pred_auto_range': wandb.Image(fig3),
                                    'Masked_pred': wandb.Image(fig4),
                                    #'Building_mask': wandb.Image(building_mask[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch + start_epoch,
                                **histograms
                            })
                            plt.close('all')
                        except Exception as e:
                            print(e)
                            pass


        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        #state_dict['mask_values'] = dataset.mask_values
        torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
        logging.info(f'Checkpoint {epoch} saved!')
        #logging.info('Training loss score: {}'.format(val_score))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--pathloss',action='store_true', default=False , help='Using the Path Loss Model(3GPP TR 38.901) to pre process the input TX information')
    parser.add_argument('--pathloss_multi_modality',action='store_true', default=True , help='Apply Path Loss Model(3GPP TR 38.901) on the end of model forward.')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Logging wandb with the start of epoch x', dest='start_epoch')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=2, n_classes=1, bilinear=args.bilinear,pathloss=args.pathloss_multi_modality)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            pathloss=args.pathloss,
            pathloss_multi_modality=args.pathloss_multi_modality,
            start_epoch=args.start_epoch
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
