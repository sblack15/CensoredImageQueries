import argparse
import os

import cv2
import numpy as np
from skimage.measure import compare_psnr, compare_ssim, compare_nrmse

if __name__ == '__main__':

    # This script computes PSNR, NRMSE, and SSIM (window size = 11) for a given set of inpainted images

    parser = argparse.ArgumentParser()
    parser.add_argument("--inpainted_image_dir", type=str, default="places2/images/inpainted/0.2/DF-Net",
                        help='directory of inpainted images')
    parser.add_argument("--original_image_dir", type=str, default="places2/images/original_images/",
                        help='directory of the original images')

    args = parser.parse_args()

    method = args.image_dir.split('/')[-1]
    mask = float(args.image_dir.split('/')[-2])

    psnrs, nrmses, ssims = [], [], []
    for file in os.listdir(args.inpainted_image_dir):
        if file.endswith('jpg'):
            inpainted_im = cv2.imread(os.path.join(args.inpainted_image_dir, file))
            original_im = cv2.imread(os.path.join(args.original_image_dir, file))
            if method == 'PIC-Net':
                original_im = cv2.resize(original_im, (256, 256))
            psnrs.append(compare_psnr(original_im, inpainted_im))
            nrmses.append(compare_nrmse(original_im, inpainted_im))
            ssims.append(compare_ssim(inpainted_im, original_im, 11, multichannel=True))

    nrmse = np.mean(nrmses)
    psnr = np.mean(psnrs)
    ssim = np.mean(ssims)

    print('NRMSE for {}, with {}% mask: {}'.format(method, int(mask * 100), nrmse))
    print('PSNR for {}, with {}% mask: {}'.format(method, int(mask * 100), psnr))
    print('SSIM for {}, with {}% mask: {}'.format(method, int(mask * 100), ssim))
