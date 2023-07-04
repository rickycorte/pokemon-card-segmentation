#
# Predict script for baseline model. This script requires torch to work
#
import argparse
import os

import torch
import torchvision.transforms.functional as TF
from torchvision.io.image import ImageReadMode
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm

import settings
from models import UNetBaseline, UnetTimm

available_models = {
    "baseline": {
        "model": UNetBaseline(in_depth=settings.input_channels, out_depth=1, depth_scale=settings.baseline_model_scale),
        "checkpoint": settings.baseline_checkpoint_path
    },
    "tunet": {
        "model": UnetTimm(out_depth=1, backbone_name="efficientnet_b3", pretrained=False, decoder_scale=settings.timmunet_decoder_scale),
        "checkpoint": settings.timmunet_checkpoint_path
    }

}

# command line args
parser = argparse.ArgumentParser(description='Segment pokemon cards')

parser.add_argument("-file", dest="file", help="Input image", type=str, default=None)
parser.add_argument("-folder", dest="folder", help="Folder where images (pngs or jpgs) are located", type=str, default=None)
parser.add_argument("-model", dest="model", help=f"Model name {available_models.keys()}", type=str, default="tunet")

args = parser.parse_args()

prefix_folder = ""

# check files to predict
files = []
if args.file is None and args.folder is None:
    print("You must specify either a file or a folder containing png or jpg files")
    exit(1)

if args.file is not None:
    if os.path.exists(args.file):
        files.append(args.file)
    else:
        print(f"No such file: {args.file}")
        exit(1)

if args.folder is not None:
    tmp = os.listdir(args.folder)
    if len(tmp) <= 0:
        print(f"No imaged found in {args.folder}")
        exit(1)

    prefix_folder = os.path.basename(os.path.normpath(args.folder))
    
    for f in tmp:
        if ".png" in f or ".jpg" in f:
            files.append(os.path.join(args.folder, f))
        
if args.model not in available_models.keys():
    print(f"No model named: {args.file}. Available models: {available_models.keys()}")
    exit(1)

print(f"Using model: {args.model}")

# load model
model = available_models[args.model]["model"]

checkpoint = torch.load(available_models[args.model]["checkpoint"])
weights = checkpoint["state_dict"]
for key in list(weights):
    weights[key.replace("model.", "")] = weights.pop(key)

model.load_state_dict(weights)
model.eval()

if torch.cuda.is_available():
    print("GPU detected")
    model.to("cuda")

if len(files) > 1:
    out_path = os.path.join(settings.output_root, prefix_folder)
    os.makedirs(out_path, exist_ok=True)
    mpath = os.path.join(out_path, "masks")
    opath = os.path.join(out_path, "overlapped")
    os.makedirs(mpath, exist_ok=True)
    os.makedirs(opath, exist_ok=True)
else:
    out_path = os.path.join(settings.output_root)

for f in tqdm(files):
    with torch.no_grad():
        im = read_image(f, ImageReadMode.RGB).to(torch.float) / 255.
        im = TF.resize(im, settings.card_size)
        pred = torch.sigmoid(model(im.unsqueeze(dim=0).to(next(model.parameters()).device))) >= 0.5
        pred = pred.cpu().detach().squeeze()
        
        pred_rgb = torch.zeros((3, *pred.shape[-2:]))
        pred_rgb[:, pred] = 1

        # make overlapped image
        im[:, pred] = im[:, pred] * 0.5 + pred_rgb[:, pred] * 0.7

        fname = os.path.basename(f)

        # single image
        if len(files) == 1:
            save_image(pred_rgb, os.path.join(out_path, f"mask_{fname}"))
            save_image(im, os.path.join(out_path, f"overlapped_{fname}"))
        else:
            save_image(pred_rgb, os.path.join(mpath, fname))
            save_image(im, os.path.join(opath, fname))
            