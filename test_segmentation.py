import gep
from deap import creator, base, tools
import numpy as np
import random
from PIL import Image

import operator 
import math
import datetime
import time

import cv2
import matplotlib.pyplot as plt
from numba import jit
from skimage.measure import compare_ssim
from scipy.spatial import distance
from sklearn.metrics import jaccard_score
import os

#path of the images to test
images_test_path = "data/test/"
images_test = sorted(os.listdir(images_test_path))
test_X = []
test_Y = []

#out = "add(div(IM_13, IM_0),sub(logarithm(div(divc(min_(IM_14, add(IM_3, IM_12))), subc(min_(IM_2, min_(IM_0, IM_14))))), IM_2))"
#out = "add(sub(logarithm(max_(add(IM_3, sub(IM_14, divc(mulc(IM_8)))), IM_4)), IM_2),sub(logarithm(max_(add(IM_3, sub(IM_14, divc(mulc(IM_8)))), IM_4)), IM_2))"
#out = "add(logarithm(mulc(sub(IM_5, addc(mulc(IM_3))))),sqrt(sub(IM_4, max_(IM_5, mult(max_(IM_0, max_(IM_10, div(IM_6, IM_11))), IM_14)))))"
out = "add(sub(logarithm(add(sqrt(max_(IM_9, add(divc(subc(IM_15)), min_(IM_4, IM_1)))), IM_5)), IM_2),logarithm(sub(IM_12, min_(IM_8, IM_15))))"
#out = "add(subc(sub(add(IM_13, logarithm(logarithm(sub(IM_5, div(IM_5, sqrt(IM_2)))))), IM_13)),IM_5)"

base_height = 300
print("Load testing images.......")
for image in images_test:
    im = cv2.imread(images_test_path+image)
    h, w, c = im.shape
    hpercent = (base_height / w)
    wsize = int((float(h) * float(hpercent)))
    h = int(wsize)
    w = int(base_height)
    dim = (w, h)
    im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    hsvimg = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(hsvimg, cv2.COLOR_BGR2GRAY)
    test_X.append(gray)

def normalize(A, minval=0.0000000001):
    new_A = ((A - np.amin(A)) / ((np.amax(A) - np.amin(A)) + minval)) * 255.
    return new_A.astype(int)

def add(A, B):
    return normalize(np.add(A, B))

def sub(A, B):
    return normalize(np.subtract(A, B))
 
def mult(A, B):
    return normalize(np.multiply(A, B))

def div(A, B, minval=0.0000000001):
    min_ = np.amin(B)
    if min_==0:
        B = B.clip(min=minval)
    out = np.divide(A,B)
    return normalize(out)

def max_(A, B):
    return normalize(np.maximum(A,B))

def min_(A, B):
    return normalize(np.minimum(A,B))

def addc(A, constant=2):
    return normalize(A + constant)

def subc(A, constant=2):
    return normalize(A - constant)

def mulc(A, constant=2):
    return normalize(A * constant)

def divc(A, constant=2):
    return normalize(A / constant)

def sqrt(A):
    return normalize(np.sqrt(A))

def logarithm(A, minval=0.0000000001):
    min_ = np.amin(A)
    if min_== 0:
        A = A.clip(min=minval)
    out = normalize(np.log(A))
    return out

def threshold(A):
    ret,thresh1 = cv2.threshold(A.astype(np.uint8),127,255,cv2.THRESH_BINARY)
    return normalize(A)

def mean_w(array):
    return np.mean(array)

def sum_w(array):
    return np.sum(array)

def std_w(array):
    return np.std(array)

def max_w(array):
    return np.amax(array)

def min_w(array):
    return np.amin(array)

def median_w(array):
    return np.median(array)

OPERATORS = {
    'mean': mean_w,
    'sum': sum_w,
    'std': std_w,
    'max': max_w,
    'min': min_w,
    'median': median_w,
}

def custom_filter(image, window_size, step_size, operator):
    padding = int(window_size/2)
    new_image = []
    (w_width, w_height) = (window_size, window_size)
    #height
    if operator == "laplacian":
        ddepth = cv2.CV_16S
        dst = cv2.Laplacian(image.astype(np.uint8), ddepth, window_size)
        return normalize(dst)

    for x in range(-1 * padding, image.shape[0] - w_height + padding + 1, step_size):
        row_image = []
        #width
        for y in range(-1 * padding, image.shape[1] - w_width + padding + 1, step_size):
            pixels = []
            for i in range(x, x + w_height):
                for j in range(y, y + w_width):
                    if i >= 0 and j >= 0 and i < image.shape[0] and j < image.shape[1]:
                        pixels.append(image[i][j])
            pixels = np.array(pixels)
            row_image.append(OPERATORS[operator](pixels))
        new_image.append(row_image)
    new_image = np.array(new_image)
    return new_image

TERMINAL_SETTINGS = dict((
    ('IM_1', {'window_size': 3, 'step_size': 1, 'operator': 'mean'}),
    ('IM_2', {'window_size': 5, 'step_size': 1, 'operator': 'mean'}),
    ('IM_3', {'window_size': 7, 'step_size': 1, 'operator': 'mean'}),
    ('IM_4', {'window_size': 3, 'step_size': 1, 'operator': 'std'}),
    ('IM_5', {'window_size': 5, 'step_size': 1, 'operator': 'std'}),
    ('IM_6', {'window_size': 7, 'step_size': 1, 'operator': 'std'}),
    ('IM_7', {'window_size': 3, 'step_size': 1, 'operator': 'max'}),
    ('IM_8', {'window_size': 5, 'step_size': 1, 'operator': 'max'}),
    ('IM_9', {'window_size': 7, 'step_size': 1, 'operator': 'max'}),
    ('IM_10', {'window_size': 3, 'step_size': 1, 'operator': 'min'}),
    ('IM_11', {'window_size': 5, 'step_size': 1, 'operator': 'min'}),
    ('IM_12', {'window_size': 7, 'step_size': 1, 'operator': 'min'}),
    ('IM_13', {'window_size': 3, 'step_size': 1, 'operator': 'median'}),
    ('IM_14', {'window_size': 5, 'step_size': 1, 'operator': 'median'}),
    ('IM_15', {'window_size': 7, 'step_size': 1, 'operator': 'median'}),
))

TERMINAL_AVAILABLE = ['IM_0',
    'IM_1'
    , 'IM_2', 'IM_3'
    , 'IM_4', 'IM_5', 'IM_6',
    'IM_7', 'IM_8', 'IM_9', 'IM_10', 'IM_11', 'IM_12',
    'IM_13', 'IM_14', 'IM_15',
]
def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

def CalculateBestModelOutput(model, context={}):
    return eval(model, None, context) 

IMAGES_TEST = {}
names = []
count = 0
for name in TERMINAL_AVAILABLE:
    if name == 'IM_0':
        continue
    settings = TERMINAL_SETTINGS[name]
    names.append(name)
    print ('Processing testing image...', name)
    images = np.array([custom_filter(image=image, **settings) for image in np.array(test_X)])
    IMAGES_TEST[name] = images
IMAGES_TEST['IM_0'] = np.array(test_X)
d=0
for k in range(len(IMAGES_TEST[TERMINAL_AVAILABLE[0]])):
    predPE = CalculateBestModelOutput(
        str(out),
        context={im: IMAGES_TEST[im][k] for im in TERMINAL_AVAILABLE}
    )
    predPE = predPE.astype(np.uint8)

    xmax, xmin = predPE.max(), predPE.min()
    x = (predPE  - xmin)/(xmax - xmin + 0.0000000001)
    n = round_down(np.mean(x), 1)
    for j in range(len(x)):
        for i in range(len(x[j])):
            if x[j][i] > n:
                x[j][i]=255
            else:
                x[j][i]=0
    #path of the results
    path_results = "segmentation_result/"
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    cv2.imwrite(path_results+images_test[d], x)   
    d+=1    