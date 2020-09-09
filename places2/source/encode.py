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
    parser.add_argument("--image_dir", type=str, default="places2/images/inpainted/0.2/DF-Net",
                        help='directory of the images to classify')
    parser.add_argument("--weights", type=str, default="places2/weights/resnet18_standard.pth.tar",
                        help='location to weights file, can be regularly trained Places2 weights or weights trained '
                             'with masked images')
    parser.add_argument("--save_directory", type=str, default='places2/feature_vectors/standard', help='where to save the encoded feature_vectors')
    parser.add_argument("--batch_size", type=int, default=50, help='batch size')
    parser.add_argument("--gpu", type=str, default='0', help='GPU id, -1 for CPU')

    args = parser.parse_args()

    os.environ['VISIBLE_CUDA_DEVICES'] = args.gpu
    model = load_resnet18_model(args.weights)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    if args.gpu != '-1':
        model.cuda()

    if 'inpainted' in args.image_dir:
        method = args.image_dir.split('/')[-1]
        mask = float(args.image_dir.split('/')[-2])
    else:
        method = 'Original'
        mask = 0

    batch_size = 100

    center_crop = trn.Compose([
        trn.Resize(224),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print('Encoding images...')
    encodings = []
    image_list = sorted(os.listdir(args.image_dir))
    with torch.no_grad():
        for i in range(len(image_list) // args.batch_size + 1):
            batch = []
            for j in range(i * args.batch_size, (i + 1) * args.batch_size):
                try:
                    image = Image.open(os.path.join(args.image_dir, image_list[j]))
                    batch.append(center_crop(image))
                except IndexError:
                    break
            if batch:
                batch = torch.stack(batch)
                if args.gpu != '-1':
                    batch = batch.cuda()
                batch_encodings = model.forward(batch).cpu().numpy()
                encodings.append(batch_encodings)
    encodings = np.squeeze(np.concatenate(encodings))
    with open('{}/places2_{}_{}.dat'.format(args.save_directory, mask, method), 'wb') as f:
        pickle.dump(encodings, f)
