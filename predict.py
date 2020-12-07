import os, argparse
import numpy as np
import cv2
import torch
import utils

# args
parser = argparse.ArgumentParser()
# Input
parser.add_argument('--image', type = str, default = None, help = 'path to image')
parser.add_argument('--mask', type = str, default = None, help = 'path to mask')
# General parameters
parser.add_argument('--results_path', type = str, default = './results2', help = 'testing samples path that is a folder')
parser.add_argument('--gan_type', type = str, default = 'WGAN', help = 'the type of GAN for training')
parser.add_argument('--gpu_ids', type = str, default = "-1", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
# Training parameters
parser.add_argument('--epoch', type = int, default = 40, help = 'number of epochs of training')
parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
# Network parameters
parser.add_argument('--in_channels', type = int, default = 4, help = 'input RGB image + 1 channel mask')
parser.add_argument('--out_channels', type = int, default = 3, help = 'output RGB image')
parser.add_argument('--latent_channels', type = int, default = 48, help = 'latent channels')
parser.add_argument('--pad_type', type = str, default = 'zero', help = 'the padding type')
parser.add_argument('--activation', type = str, default = 'elu', help = 'the activation type')
parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type')
parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
# Dataset parameters
parser.add_argument('--baseroot', type = str, default = '../../inpainting/dataset/Places/img_set')
parser.add_argument('--baseroot_mask', type = str, default = '../../inpainting/dataset/Places/img_set')
opt = parser.parse_args()

if opt.image is None or opt.mask is None:
    raise Exception('Need to provide image and mask')

# Output dir
if not os.path.exists(opt.results_path):
    os.makedirs(opt.results_path)


# Build networks
generator = utils.create_generator(opt).eval()


# Load model
model_name = os.path.join('pretrained_model', 'deepfillv2_WGAN_G_epoch40_batchsize4.pth')
pretrained_dict = torch.load(model_name, map_location=torch.device('cpu'))
generator.load_state_dict(pretrained_dict)

# Load file to test
img = cv2.imread(opt.image)
mask = cv2.imread(opt.mask)[:, :, 0]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).contiguous()

# Batch 1
img = img.unsqueeze(0)
mask = mask.unsqueeze(0)
print(img.shape)
print(mask.shape)

# Generator output
with torch.no_grad():
    first_out, second_out = generator(img, mask)

# forward propagation
first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
second_out_wholeimg = img * (1 - mask) + second_out * mask      # in range [0, 1]

masked_img = img * (1 - mask) + mask
mask = torch.cat((mask, mask, mask), 1)
img_list = [second_out_wholeimg]
name_list = ['second_out']
utils.save_sample_png(sample_folder = opt.results_path, sample_name = os.path.basename(opt.image)[:-4],
                    img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
