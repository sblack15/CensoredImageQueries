import argparse
import os

import numpy as np
from PIL import Image
import cv2


def inpaint_black_mask(image, mask):
    mask[mask == 0] = 1
    mask[mask != 1] = 0
    image = image * np.expand_dims(mask, -1)
    return image


if __name__ == '__main__':

    '''
        To inpaint images, you must supply a hook to your desired method's "inpaint" method, taking the image path
        and mask path as input. This script will load all of the images and their corresponding masks at a given mask
        percentage and then save the inpainting in the appropriate output directory
    
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="places2/images/original",
                            help='directory of the images to classify')
    parser.add_argument("--mask_dir", type=str, default="places2/images/mask")
    parser.add_argument("--mask_level", type=str, default='0.2',
                            help="what percentage of masked image to use. Should be one of '0.1', '0.2', '0.3', '0.4', "
                                 "'0.5', '0.6' or '0.7'")
    parser.add_argument("--method_name", type=str, default='NS', help='name of implemented method')
    parser.add_argument("--inpainted_dir", type=str, default='places2/images/inpainted')
    parser.add_argument("--gpu", type=str, default="0", help='GPU id, -1 for CPU')

    args = parser.parse_args()

    output_dir = os.path.join(args.inpainted_dir, args.mask_level, args.method_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(args.image_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(args.image_dir, filename)
            mask_path = os.path.join(args.mask_dir, args.mask_level, filename[:-3] + 'png')
            '''
            Below, insert desired inpainting method. Pass the image path and the mask path to the inpainting method
            Have the method return the inpainted version, as a numpy array

            inpainted_image = method.inpaint(image_path, mask_path)
            Below are two examples: the first being using the Navier-Stokes algorithm from the cv2 library,
            and the second simply inpainting using a black mask (setting all pixel values to 0 in the censored regions)
            '''
            image = np.array(Image.open(image_path))
            mask = np.array(Image.open(mask_path))

            inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
            # inpainted_image = inpaint_black_mask(image, mask)

            # Change the above code to inpaint an image using desired inpainting method

            inpainted_image = Image.fromarray(inpainted_image)
            inpainted_image.save(os.path.join(output_dir, filename))