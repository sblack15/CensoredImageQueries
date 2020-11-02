# CensoredImageQueries
Official Repository for the paper "Evaluation of Inpainting and Augmentation for Censored Image Queries". Currently, we only have included the code to reproduce our experiments on the Places2 and CelebA dataset. Hotels50k code will be added soon.

# Inpainting

For each of the three datasets ('places2', 'hotels50k', and 'celebA') that we perform our evaluation on, we provide the set of images that are inpainted and the corresponding masks. These are available in a Google drive [folder](https://drive.google.com/drive/folders/13oO5CikiXckYjJ8i5x25Ht1-XiqVQQlK?usp=sharing). We also provide a script in each dataset directory, called 'inpaint.py', which provides a hook for the user to edit and include the desired inpainting method. This script performs the inpainting and saves the images, using the chosen mask and implemented inpainting method. Once the images for a given mask and method are generated, then the chosen evaluation script (ex: retrieval, classification) can be run for that set of images. 

For the retrieval experiments, the set of database images for Places2 and CelebA are included in the Drive folder. For Hotels50k, those images can be downloaded following the instructions outlined [here](https://github.com/GWUvision/Hotels-50K).

For Places2 and Hotels50k, the model weights that are used to perform classification and generate feature representations of the images are also included in the drive folder. For CelebA, we use the [Keras-VGGFace](https://pypi.org/project/keras-vggface/) and [MTCNN face detector](https://pypi.org/project/mtcnn/), whose weights are included in their respective pip packages.

Below are the list of inpainting methods that we evaluate in our paper (links to the Github repos of the learning-based methods are provided):

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


# Training with Occluded Images

For Places2 and Hotels50k, we also provide the training scripts to train the ResNet-18 models using occluded images. Once these models are trained, their weights can be loaded and then can be used to perform the classification / retrieval tasks. To run training, download the complete [places2](http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar) /[hotels50k](https://cs.slu.edu/~stylianou/images/hotels-50k/test.tar.lz4) datasets, and place them in the directory '{places2|hotels50k}/images/complete_set'. We also provide the weights that we trained using the various occlusion methods, as well as those trained without any occlusion, in the Google drive folder. All model weights should be placed in the directory '{places2|hotels50k}/weights'

# References

\[1] Bertalmio  M,  Bertozzi  AL,  Sapiro  G  (2001)  Navier-stokes, fluid dynamics, and image and video inpainting. In: Proc. IEEE Conference on Computer Visionand Pattern Recognition

\[2] Telea A (2004) An image inpainting technique based onthe fast marching method. Journal of Graphics Tools, pp 23–34

\[3] Barnes C, Shechtman E, Finkelstein A, Goldman DB(2009) Patchmatch: A randomized correspondence algorithm for structural image editing. ACM Transactions on Graphics

\[4] Huang  JB,  Kang  SB,  Ahuja  N,  Kopf  J  (2014)  Image completion  using  planar  structure  guidance.  ACM Transactions on Graphic 33(4):1–10

\[5] Zheng C, Cham TJ, Cai J (2019) Pluralistic image completion. In: Proc. IEEE Conference on Computer Vision and Pattern Recognition, pp 1438–1447

\[6] Wang  Y,  Tao  X,  Qi  X,  Shen  X,  Jia  J  (2018)  Image inpainting via generative multi-column convolutionalneural networks. In: Advances in Neural Information Processing Systems, pp 331–340

\[7] Iizuka S, Simo-Serra E, Ishikawa H (2017) Globally and locally  consistent  image  completion.  ACM  Transac-tions on Graphics 36(4):1–14

\[8] Yu J, Lin Z, Yang J, Shen X, Lu X, Huang TS (2018) Generative  image  inpainting  with  contextual  attention. In: Proc. IEEE Conference on Computer Visionand Pattern Recognition, pp 5505–5514

\[9] Nazeri  K,  Ng  E,  Joseph  T,  Qureshi  F,  Ebrahimi  M(2019)  Edgeconnect:  Generative  image  inpaintingwith  adversarial  edge  learning.  IEEE  International Conference on Computer Vision Workshop

\[10] Hong X, Xiong P, Ji R, Fan H (2019) Deep fusion network for image completion. In: ACM Multimedia

\[11] Yu J, Lin Z, Yang J, Shen X, Lu X, Huang TS (2018) Generative  image  inpainting  with  contextual  attention. In: Proc. IEEE Conference on Computer Vision and Pattern Recognition, pp 5505–5514

\[12] Ren  Y,  Yu  X,  Zhang  R,  Li  TH,  Liu  S,  Li  G  (2019)Structureflow:  Image  inpainting  via  structure-awareappearance flow. In: Proc. International Conference on Computer Vision, pp 181–190
