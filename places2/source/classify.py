import argparse
import os
import pickle

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as trn

from load_model import load_resnet18_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="places2/images/inpainted/0.1/DF-Net",
                        help='directory of the images to classify')
    parser.add_argument("--label_file", type=str, default="places2/classification_labels.dat",
                        help='path to a list of the labels of the images to be classified (pickle file)')
    parser.add_argument("--weights", type=str, default="places2/weights/resnet18_standard.pth.tar",
                        help='location to weights file, can be regularly trained Places2 weights or weights trained '
                             'with masked images')
    parser.add_argument("--batch_size", type=int, default=50, help='batch size')
    parser.add_argument("--gpu", type=str, default='0', help='GPU id, -1 for CPU')

    args = parser.parse_args()

    os.environ['VISIBLE_CUDA_DEVICES'] = args.gpu
    model = load_resnet18_model(args.weights)
    if args.gpu != '-1':
        model.cuda()

    if 'inpainted' in args.image_dir:
        method = args.image_dir.split('/')[-1]
        mask = float(args.image_dir.split('/')[-2])
    else:
        method = 'Original'
        mask = 0

    with open(args.label_file, 'rb') as f:
        labels = pickle.load(f)
        labels = [int(labels[i]) for i in range(len(labels))]

    y = np.eye(365)[labels]
    x = []

    center_crop = trn.Compose([
        trn.Resize((224, 224)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print('Loading images...')
    for filename in sorted(os.listdir(args.image_dir)):
        if filename.endswith('jpg'):
            image = Image.open(os.path.join(args.image_dir, filename))
            x.append(center_crop(image))

    logits = []
    print('Processing images...')
    for i in range(len(x) // args.batch_size + 1):
        batch = x[i * args.batch_size: (i + 1) * args.batch_size]
        if batch:
            batch = torch.stack(batch)
            if args.gpu != '-1':
                batch = batch.cuda()
            logit = model.forward(batch)
            logits.append(logit.detach().cpu().numpy())

    logits = np.array(logits)
    logits = np.resize(logits, (len(x), 365))
    for k in [1, 5]:
        top_k = np.argsort(-logits)[:, :k]
        temp = np.zeros_like(logits)
        for i in range(len(temp)):
            temp[i][top_k[i]] = 1
        res = np.round(np.sum(temp * y) / len(temp), 4)
        print('Top-{} Accuracy for {}, with {}% mask: {}'.format(k, method, int(mask * 100), res))
