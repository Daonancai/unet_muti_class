import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

from focalloss import *



dir_img = '/home/caidaonan/path_to_dataset/doudou/data/img/'
dir_mask = '/home/caidaonan/path_to_dataset/doudou/data/label/'
#dir_img = 'E:\\dataset\\data\\img\\'
#dir_mask = 'E:\\dataset\\data\\label\\'
dir_checkpoint = 'checkpoints/'




def mseloss_z(groundmask,frontmask,predmask):
    msez = frontmask - predmask
    msez2 = msez*msez
    groundz = msez2*groundmask
    frontz = msez2*frontmask
    gl =groundz.sum()
    fl = frontz.sum()
    zl = gl + fl*200
    return zl/20000


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.1,
              save_cp=True,
              #save_cp=False,
              posw=[1,100,100],
              img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    #val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    #train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    #val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    #optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)

    #loss 1
    pos_wn = np.array(posw,dtype=np.float32)
    pos_w = torch.from_numpy(pos_wn)
    pos_w = pos_w.to(device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=pos_w)

    #loss 2
    pw1 = np.array([1,50],dtype=np.float32)
    pw2 = torch.from_numpy(pw1)
    pos_w1 = pw2.to(device=device, dtype=torch.float32)
    criterion1 = nn.CrossEntropyLoss(weight=pos_w1)

    #criterion1 = nn.BCEWithLogitsLoss(weight=pos_w1)
    #criterions = mseloss_z(groundmask,frontmask,predmask)


    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                #mask2,3
                frontmask = batch['mask1']
                groundmask = batch['mask2']

                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                frontmask = frontmask.to(device=device, dtype=mask_type)
                groundmask = groundmask.to(device=device, dtype=mask_type)
                #print('true_masks shape',true_masks.shape)
                masks_pred = net(imgs)
                #print('pred_masks shape',masks_pred.shape)
                
                #loss
                #idx1 = torch.LongTensor([0,1,2])
                #idx2 = torch.LongTensor([3])
                #masks_pred1 = torch.gather(masks_pred, 1, idx1)
                #masks_pred2 = torch.gather(masks_pred, 1, idx2)
                
                masks_pred1,masks_pred2 = torch.split(masks_pred, 3, 1)
                #masks_pred2 = masks_pred2.view(masks_pred.shape[0],300,300)
                #print('pred_masks shape',masks_pred1.shape,masks_pred2.shape)
                #print('true_masks shape',true_masks.shape,true_masks1.shape)
                
                loss1 = criterion(masks_pred1, true_masks)
                loss2 = criterion1(masks_pred2, frontmask)
                #loss2 = mseloss_z(groundmask,frontmask,masks_pred2)
                
                loss = loss1 + loss2
                #loss = loss2
                
                
                #loss = FocalLoss(alpha=0.1,gamma=2)(masks_pred, true_masks)
                print(loss)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR',type=float, nargs='?',default=0.005,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-p', '--loss_weight', dest='loss_weight', type=list, default=[1,100,100],
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=2.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=5)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  posw=args.loss_weight
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
