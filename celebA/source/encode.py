import cv2
import os
import argparse
import numpy as np
from mtcnn import MTCNN
import time
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


def extract_face(filename, required_size=(224, 224)):
    pixels = cv2.imread(filename)
    results = detector.detect_faces(pixels)
    try:
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        return cv2.resize(face, required_size)
    except Exception:
        return cv2.resize(pixels, required_size)


def get_embeddings(filenames):
    faces = [extract_face(f) for f in filenames]
    samples = np.asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)
    yhat = model.predict(samples)
    return yhat


if __name__ == '__main__':

    '''
        Run this script first to generate feature vectors from VGGFace of the desired inpainting method / feature combo.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="celebA/images")
    parser.add_argument("--feature", type=str, default="nose", help="inpainted facial feature (choices: face, eyes, nose, mouth")
    parser.add_argument("--model", type=str, default="GMCNN", help="inpainting method")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    image_directory = os.path.join(args.image_dir, args.feature, args.model)
    image_list = []
    embeddings = []
    for path, _, filenames in os.walk(image_directory):
        for f in filenames:
            image_list.append(os.path.abspath(os.path.join(path, f)))
    image_list.sort()
    batch_size = 100
    for i in range(len(image_list) // batch_size + 1):
        embeddings.append(get_embeddings(image_list[i * batch_size: (i + 1) * batch_size]))
    embeddings = np.concatenate(embeddings)
    np.save('{}_{}'.format(args.model, args.feature), embeddings)
