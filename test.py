import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
import scipy
from PIL import Image

parser = argparse.ArgumentParser(description="PyTorch EDSR Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="checkpoint/model_epoch_10.pth", type=str, help="model path")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def load_image(path):
    path = path[:-1]
    img = scipy.misc.imread(path)
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))
    return img
def get_image(index):


    file_l = open("train_l.txt", 'r')
    file_h = open("train_h.txt", 'r')
    img_l_path = file_l.readlines()
    img_h_path = file_h.readlines()
    imgx = load_image(img_l_path[index])
    imgy = load_image(img_h_path[index])
    file_l.close()
    file_h.close()
    return imgx, imgy

opt = parser.parse_args()
cuda = True 

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = torch.load(opt.model)["model"]

im_l, im_gt = get_image(100)
im_gt = im_gt.astype(float).astype(np.uint8)
im_gt = im_gt.transpose(1, 2, 0)
im_l = im_l.astype(float).astype(np.uint8)

im_input = im_l.astype(np.float32)
im_input = im_input.reshape(1,im_input.shape[0],im_input.shape[1],im_input.shape[2])
im_input = Variable(torch.from_numpy(im_input/255.).float())

im_input = Variable(torch.from_numpy(im_input).float())

if cuda:
    model = model.cuda(1)
    im_input = im_input.cuda(1)
else:
    model = model.cpu()
start_time = time.time()
out = model(im_input)
elapsed_time = time.time() - start_time

out = out.cpu()

im_h = out.data[0].numpy().astype(np.float32)

im_h = im_h*255.
im_h[im_h<0] = 0
im_h[im_h>255.] = 255.
im_h = im_h.transpose(1,2,0)

print("Scale=",opt.scale)
print("It takes {}s for processing".format(elapsed_time))

img = Image.fromarray(im_h, 'RGB')
img.save('predicted.bmp')
img = Image.fromarray(im_gt, 'RGB')
img.save('gt.bmp')

# fig = plt.figure()
# ax = plt.subplot("121")
# ax.imshow(im_gt)
# ax.set_title("GT")


# ax = plt.subplot("122")
# ax.imshow(im_h.astype(np.uint8))
# ax.set_title("Output(EDSR)")
# plt.show()
