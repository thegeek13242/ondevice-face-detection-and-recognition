from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import sys
import numpy as np
import mxnet as mx
import os
import json

from scipy import misc
import random
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from mxnet import MXNetError
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
from skimage import transform as trans
import matplotlib.pyplot as plt
from mxnet.contrib.onnx.onnx2mx.import_model import import_model


def get_model(ctx, model):
    image_size = (112, 112)
    # Import ONNX model
    sym, arg_params, aux_params = import_model(model)
    # Define and binds parameters to the network
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


for i in range(4):
    mx.test_utils.download(
        dirname='mtcnn-model', url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}-0001.params'.format(i+1))
    mx.test_utils.download(
        dirname='mtcnn-model', url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}-symbol.json'.format(i+1))
    mx.test_utils.download(
        dirname='mtcnn-model', url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}.caffemodel'.format(i+1))
    mx.test_utils.download(
        dirname='mtcnn-model', url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}.prototxt'.format(i+1))

# Determine and set context
if len(mx.test_utils.list_gpus()) == 0:
    ctx = mx.cpu()
else:
    ctx = mx.gpu(0)
# Configure face detector
det_threshold = [0.6, 0.7, 0.8]
mtcnn_path = os.path.join(os.path.dirname('__file__'), 'mtcnn-model')
detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx,
                         num_worker=1, accurate_landmark=True, threshold=det_threshold)


def preprocess(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    # Assert input shape
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size) == 2
        assert image_size[0] == 112
        assert image_size[0] == 112 or image_size[1] == 96

    # Do alignment using landmark points
    if landmark is not None:
        assert len(image_size) == 2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        assert len(image_size) == 2
        warped = cv2.warpAffine(
            img, M, (image_size[1], image_size[0]), borderValue=0.0)
        return warped

    # If no landmark points available, do alignment using bounding box. If no bounding box available use center crop
    if M is None:
        if bbox is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret


def get_input(detector, face_img):
    # Pass input images through face detector
    ret = detector.detect_face(face_img, det_type=0)
    if ret is None:
        return None
    bbox, points = ret
    if bbox.shape[0] == 0:
        return None
    bbox = bbox[0, 0:4]
    points = points[0, :].reshape((2, 5)).T
    # Call preprocess() to generate aligned images
    nimg = preprocess(face_img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2, 0, 1))
    return aligned


def get_feature(model, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    embedding = model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding


# # Download first image
# mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/arcface/player1.jpg')
# # Download second image
# mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/arcface/player2.jpg')
# Download onnx model
mx.test_utils.download(
    'https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100.onnx')
# Path to ONNX model
model_name = 'resnet100.onnx'

# Load ONNX model
model = get_model(ctx, model_name)

embeddings_name = {}

def update_face_data(dirname):
    if not(type(dirname) is list):
        files = os.listdir(dirname)
        for file in files:
            if file.endswith('.jpg'):
                img = cv2.imread(os.path.join(dirname, file))
                pre = get_input(detector, img)
                out = get_feature(model, pre)
                embeddings_name[file] = out
    else:
        files = dirname
        for file in files:
            if file.endswith('.jpg'):
                img = cv2.imread(file)
                pre = get_input(detector, img)
                out = get_feature(model, pre)
                embeddings_name[file] = out

def inference(com_img, ftime, THRESHOLD=0.8):
    # com_img = cv2.imread(image)
    com_pre = get_input(detector, com_img)
    try: 
        com_out = get_feature(model, com_pre)
    except MXNetError:
        print('No face detected')
        return

    min_dist = 1e9
    min_name = ''

    for key, value in embeddings_name.items():
        dist = np.sum(np.square(value - com_out))
        if dist < min_dist:
            min_dist = dist
            min_name = key
    
    if min_dist > THRESHOLD:
    # # if np.dot(com_out, embeddings_name[min_name].T) > 0.8:
        print('No similar face found')
        return " "
    else:
        with open("logfaces.json","r") as f:
            data = json.load(f)
            data.append({"0":[min_name,ftime]})
            print([min_name,ftime])
            print('-'*10+f"recognized {min_name} at time {ftime} at dist {min_dist}"+'-'*10)
            cv2.imwrite(f"crops/{min_name}_{ftime}.jpg", com_img)
        with open("logfaces.json","w") as f:
            json.dump(data,f)
        return min_name
    # print('The most similar image is: ', min_name)
    # print('The distance is: ', min_dist)
    # print('The similarity is: ', np.dot(com_out, embeddings_name[min_name].T))

#   # Load first image
#     img1 = cv2.imread(first)

#     # Preprocess first image
#     pre1 = get_input(detector,img1)

#     # Get embedding of first image
#     out1 = get_feature(model,pre1)

#     # Load second image
#     img2 = cv2.imread(second)

#     # Preprocess second image
#     pre2 = get_input(detector,img2)

#     # Get embedding of second image
#     out2 = get_feature(model,pre2)

#     # Compute squared distance between embeddings
#     dist = np.sum(np.square(out1-out2))
#     # Compute cosine similarity between embedddings
#     sim = np.dot(out1, out2.T)
    # Print predictions


# Run inference
if __name__ == '__main__':
    with open("logfaces.json","w") as f:
        f.write("[]")
    print("Success")
    update_face_data("face_data")
    img = cv2.imread("timg4.jpg")
    inference(img, 32131)
