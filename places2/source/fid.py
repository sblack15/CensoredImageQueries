import argparse
import os

from pytorch_fid.fid_score import calculate_fid_given_paths

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--inpainted_dir", type=str, default="places2/images/inpainted/0.2/DF-Net",
                        help='directory of the images to classify')
    parser.add_argument("--original_image_dir", type=str, default="places2/images/original",
                        help='directory of the original images')
    parser.add_argument("--batch_size", type=int, default=50, help='batch size')
    parser.add_argument("--gpu", type=str, default='0', help='GPU id, -1 for CPU')

    args = parser.parse_args()

    os.environ['VISIBLE_CUDA_DEVICES'] = args.gpu

    method = args.inpainted_dir.split('/')[-1]
    mask = float(args.inpainted_dir.split('/')[-2])

    fid = calculate_fid_given_paths([args.original_image_dir, args.inpainted_dir], args.batch_size, args.gpu, 2048)
    print('FID for {}, with {}% mask: {}'.format(method, int(mask * 100), fid))
