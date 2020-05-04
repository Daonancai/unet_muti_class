import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet


def predict_img1(net,
                 full_img,
                 device,
                 scale_factor=1.0,
                 out_threshold=0.5):
    net.eval()

    # img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

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
    img = torch.from_numpy(img_trans)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        outputz = net(img)
        print('outputz shape:', outputz.shape)
        # output, masks_pred2 = torch.split(outputz, 2, 1)

        probs = F.softmax(outputz, dim=1)

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


# img = cv2.imread('doudoufenci_10502.jpg')
# img = img[..., [2, 1, 0]]


net = UNet(n_channels=3, n_classes=5)
device = torch.device('cpu')
net.to(device=device)
net.load_state_dict(torch.load('checkpoints_doudouseban/CP_epoch50.pth', map_location=device))

img2 = Image.open('0a420785dc321a3dc275288af298aaa7.jpg')
img2 = img2.resize((300, 300))
img_faceroi = img2.copy().resize((300, 300))
img_faceroi = np.array(img_faceroi)

mask = predict_img1(net=net,
                    full_img=img2,
                    scale_factor=0.5,
                    out_threshold=0.7,
                    device=device)
# mask = np.transpose(mask,axes=[1,2,0])
img = np.zeros(shape=(mask.shape[1], mask.shape[2], 3), dtype=np.float32)
# np.where(mask[...,0])
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

cv2.imwrite('a.png', img)
cv2.imwrite('b.png', cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
