import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
#from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset



#python predict1.py --model /home/xzy/unettest/Pytorch_UNet_1/checkpoints/CP_epoch48.pth --scale 0.5 --mask-threshold 0.5




def predict_img1(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    #img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    pil_img = full_img.resize((600, 600))
    w, h = pil_img.size
    newW, newH = int(scale_factor * w), int(scale_factor * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    pil_img = pil_img.resize((newW, newH))

    img_nd = np.array(pil_img)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    if img_trans.max() > 1:
        img_trans = img_trans / 255
    img= torch.from_numpy(img_trans)




    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        outputz = net(img)
        print('outputz shape:',outputz.shape)
        output,masks_pred2 = torch.split(outputz, 3, 1)
        
        
        probs = F.softmax(output, dim=1)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold
    



def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")

    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.7)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()

'''
def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files
'''

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    save_dir='/home/caidaonan/path_to_dataset/doudou/res'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_face_pro='/home/caidaonan/path_to_dataset/doudou/test_Anqi'
    #in_files = args.input
    in_files=[os.path.join(data_face_pro,x) for x in os.listdir(data_face_pro)]
    data_ori='/home/caidaonan/path_to_dataset/doudou/test_Anqi'
    ori_files=os.listdir(data_ori)
    #out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=5)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))
        
        print(fn,'ffffffffffffffff')
        img2 = Image.open(fn)
        img2 = img2.resize((300,300))
        img_faceroi=img2.copy().resize((300,300))
        img_faceroi=np.array(img_faceroi)
        img_ori=Image.open(os.path.join(data_ori,os.path.basename(fn))).resize((300,300))
        img_ori = np.array(img_ori)
        
        mask = predict_img1(net=net,
                           full_img=img2,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        
        '''
        mask = predict_img(net=net,
                           full_img=img2,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        '''
        
        print('mask shape:',mask.shape)
        
        
        import cv2
        img = np.zeros(shape=(mask.shape[1], mask.shape[2], 3), dtype=np.float32)
        img2 = np.array(img2)
        for i in range(mask.shape[1]):
            for j in range(mask.shape[2]):
                if mask[0][i][j]:
                    img[i][j][0] = 0
                    img[i][j][1] = 0
                    img[i][j][2] = 0
                    
                if mask[1][i][j]:
                    img[i][j][0] = 255
                    img[i][j][1] = 255
                    img[i][j][2] = 255
                    img2[i][j][0] = 100
                    img2[i][j][1] = 100
                    img2[i][j][2] = 100
                    
                if mask[2][i][j]:
                    img[i][j][0] = 255
                    img[i][j][1] = 255
                    img[i][j][2] = 255
                    img2[i][j][0] = 255
                    img2[i][j][1] = 255
                    img2[i][j][2] = 255

        # for i in range(mask.shape[1]):
        #     for j in range(mask.shape[2]):
        #         if img_faceroi[i][j][0]==0 and img_faceroi[i][j][1]==0 and img_faceroi[i][j][2]==0:
        #             img2[i,j,:]=img_ori[i,j,:]
        # cv2.imshow('res',img2)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join('/home/caidaonan/path_to_dataset/doudou/foreground',os.path.basename(fn)),img)
        cv2.imwrite(os.path.join(save_dir,os.path.basename(fn)),cv2.cvtColor(img2,cv2.COLOR_RGB2BGR))






























