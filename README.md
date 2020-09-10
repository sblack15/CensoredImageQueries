# CensoredImageQueries
Official Repository for the paper "Evaluation of Inpainting and Augmentation for Censored Image Queries"

# Inpainting

For each of the three datasets ('places2', 'hotels50k', and 'celebA') that we perform our evaluation on, we provide the set of images that are inpainted, the corresponding masks, model weights that are used to generate feature representations of the images, and the database images that are used for retrieval tasks. These are available [here] (https://drive.google.com/drive/folders/13oO5CikiXckYjJ8i5x25Ht1-XiqVQQlK?usp=sharing). We also provide a script in each dataset directory, called 'inpaint.py', which provides a hook for the user to edit and include the desired inpainting method. This script performs the inpainting and saves the images, using the chosen mask and inpainting method. Once the images for a given mask and method are completed, then the chosen evaluation script can be run for that set of images.

Below are the links to the implementations that we use for each inpainting method in our paper.

#TDOO Add links
