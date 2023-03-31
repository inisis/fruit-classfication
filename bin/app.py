import io
import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from model.classifier import Classifier  # noqa
from model.utils import get_pred
from easydict import EasyDict as edict
import numpy as np
import cv2
import torch

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")
parser.add_argument('load_path', default=None, metavar='LOAD_PATH', type=str,
                    help="Path to the saved models")
parser.add_argument('--device_id', default='0', type=str, help="GPU indices "
                    "comma separated, e.g. '0,1' ")
args = parser.parse_args()


app = Flask(__name__)
with open(args.cfg_path) as f:
    cfg = edict(json.load(f))
device = torch.device('cuda:{}'.format(args.device_id))    
model = Classifier(cfg).to(device)
model.load_state_dict(torch.load(args.load_path + '/0_best.ckpt', map_location=device)['state_dict'])
model.eval()


def fix_ratio(image):
    h, w, c = image.shape

    if h >= w:
        ratio = h * 1.0 / w
        h_ = cfg.long_side
        w_ = round(h_ / ratio)
    else:
        ratio = w * 1.0 / h
        w_ = cfg.long_side
        h_ = round(w_ / ratio)

    image = cv2.resize(image, dsize=(w_, h_),
                        interpolation=cv2.INTER_LINEAR)
    image = np.pad(
        image,
        ((0, cfg.long_side-h_), (0, cfg.long_side-w_), (0, 0)),
        mode='constant', constant_values=cfg.pixel_mean
    )

    return image

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR).astype(np.float32)
    if cfg.fix_ratio:
        image = fix_ratio(image)
    else:
        image = cv2.resize(image, dsize=(cfg.width, cfg.height),
                            interpolation=cv2.INTER_LINEAR)
    
    # normalization
    image -= cfg.pixel_mean
    # vgg and resnet do not use pixel_std, densenet and inception use.
    if cfg.use_pixel_std:
        image /= cfg.pixel_std
    # normal image tensor :  H x W x C
    # torch image tensor :   C X H X W
    image = image.transpose((2, 0, 1))    
    return image


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    tensor = torch.from_numpy(tensor).unsqueeze(0).to(device)
    output, _ = model(tensor)
    pred = np.zeros((cfg.num_tasks, 1))
    for t in range(cfg.num_tasks):
        pred[t] = get_pred(output, t, cfg)
    return pred


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        pred = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': "ripeness", 'score': ','.join(map(lambda x: '{:.5f}'.format(x), pred[:, 0]))})


if __name__ == '__main__':
    app.run()