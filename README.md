# CensoredImageQueries
Official Repository for the paper "Evaluation of Inpainting and Augmentation for Censored Image Queries"

# Inpainting

For each of the three datasets ('places2', 'hotels50k', and 'celebA') that we perform our evaluation on, we provide the set of images that are inpainted, the corresponding masks, model weights that are used to perform classification and generate feature representations of the images, and the database images that are used for retrieval tasks. These are available [here](https://drive.google.com/drive/folders/13oO5CikiXckYjJ8i5x25Ht1-XiqVQQlK?usp=sharing). We also provide a script in each dataset directory, called 'inpaint.py', which provides a hook for the user to edit and include the desired inpainting method. This script performs the inpainting and saves the images, using the chosen mask and implemented inpainting method. Once the images for a given mask and method are inpainted, then the chosen evaluation script (ex: retrieval, classification) can be run for that set of images.

Below are the list of methods that we include in our paper (links to the Github repos of the learning-based methods are provided):

Navier-Stokes \[1]
Fast-Marching \[2]
PatchMatch \[3]
Planar Structure Guidance \[4]
[Pluralistic Image Completion](https://github.com/lyndonzheng/Pluralistic-Inpainting) \[5]
[Image Inpainting Using Generative Multi-Column Neural Networks](https://github.com/shepnerd/inpainting_gmcnn) \[6]
[Globally and Locally Consistent Image Completion](https://github.com/satoshiiizuka/siggraph2017_inpainting) \[7]
[Generative Image Inpainting with Contextual Attention](https://github.com/JiahuiYu/generative_inpainting/tree/v1.0.0) \[8]
[EdgeConnect](https://github.com/knazeri/edge-connect) \[9]
[Deep-Fusion Network](https://github.com/hughplay/DFNet) \[10]
[Free-Form Image Inpainting with Gated Convolution](https://github.com/JiahuiYu/generative_inpainting) \[11]
[StructureFlow](https://github.com/RenYurui/StructureFlow) \[12]
