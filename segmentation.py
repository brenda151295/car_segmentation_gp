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
import csv
import matplotlib.pyplot as plt
from numba import jit
from skimage.measure import compare_ssim
from scipy.spatial import distance
from sklearn.metrics import jaccard_score

# for reproduction
s = 0
random.seed(s)
np.random.seed(s)

import os


fgbg = cv2.createBackgroundSubtractorMOG2(
    history=10,
    varThreshold=2,
    detectShadows=False)

#Plotar Grafico
def plot_log_curves(logbook):

    gen = logbook.select("gen")
    min_ = logbook.select("min")
    avg = logbook.select("avg")
    max_ = logbook.select("max")

    row = [gen, min_, avg, max_]


    import matplotlib.pyplot as plt


    plt.plot(gen, min_, "b-", label="Average Fitness")
    plt.plot(gen, avg, "g-", label="Average Fitness")
    plt.plot(gen, max_, "y-", label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")


    plt.savefig("output/graph.png")

#Generate logs
def plot_log(logbook):
    gen = logbook.select("gen")
    min_ = logbook.select("min")
    avg = logbook.select("avg")
    max_ = logbook.select("max")

    row = [gen, min_, avg, max_]


    import matplotlib.pyplot as plt

    plt.plot(gen, min_, "b-", label="Average Fitness")
    plt.plot(gen, avg, "g-", label="Average Fitness")
    plt.plot(gen, max_, "y-", label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc='best')
    plt.savefig("output/graph.png")

    plt.close()
    with open('output/graph.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row[0])
        writer.writerow(row[1])
        writer.writerow(row[2])
        writer.writerow(row[3])
    csvFile.close()

#doublecheck the data is there
images_train_path = "Train/"
images_test_path = "Test/"

images_train = sorted(os.listdir(images_train_path))
images_test = sorted(os.listdir(images_test_path))

train_X = []
train_Y = []

test_X = []
test_Y = []

base_height = 100

print("Load training images.......")
for image in images_train:
    im = cv2.imread(images_train_path+image)
    h, w, c = im.shape
    hpercent = (base_height / w)
    wsize = int((float(h) * float(hpercent)))
    h = int(wsize)
    w = int(base_height)
    dim = (w, h)
    im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    if (len(image) > 7):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        train_Y.append(thresh1)
    else:
        hsvimg = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(hsvimg, cv2.COLOR_BGR2GRAY)
        train_X.append(gray)
    print ("image train:", gray.shape)


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
    if (len(image) > 7):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        test_Y.append(thresh1)
    else:
        hsvimg = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(hsvimg, cv2.COLOR_BGR2GRAY)
        test_X.append(gray)
    print ("image test:", gray.shape)

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

def canny(A):
    h, w, _ = A.shape()

    gray_filtered = cv2.bilateralFilter(A, 7, 50, 50)

    edges_foreground = cv2.bilateralFilter(A, 9, 75, 75)
    foreground = fgbg.apply(edges_foreground)

    kernel = np.ones((50,50),np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    edges_filtered = cv2.Canny(edges_foreground, 60, 120)
    cropped = (foreground // 255) * edges_filtered

    return cropped

#------------------------------------
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
TERMINAL_AVAILABLE = [
    'IM_0','IM_1', 'IM_2', 'IM_3', 'IM_4', 'IM_5', 'IM_6',
    'IM_7', 'IM_8', 'IM_9', 'IM_10', 'IM_11', 'IM_12',
    'IM_13', 'IM_14', 'IM_15',
]


IMAGES = {}
names = ['IM_0']
for name in TERMINAL_AVAILABLE:
    if name == 'IM_0':
        continue
    settings = TERMINAL_SETTINGS[name]
    names.append(name)
    print ('Processing image ', name)
    IMAGES[name] = np.array([custom_filter(image=image, **settings) for image in np.array(train_X)])
IMAGES['IM_0'] = np.array(train_X)
IM_0 = np.array(train_X)


Y = np.array(train_Y)

pset = gep.PrimitiveSet('Main', input_names=names)
pset.add_function(add, 2)
pset.add_function(sub, 2)
pset.add_function(mult, 2)
pset.add_function(div, 2)
pset.add_function(max_, 2)
pset.add_function(min_, 2)
pset.add_function(addc, 1)
pset.add_function(subc, 1)
pset.add_function(mulc, 1)
pset.add_function(divc, 1)
pset.add_function(sqrt, 1)
pset.add_function(logarithm, 1)
#pset.add_function(canny, 1)
#pset.add_function(threshold, 1)

from deap import creator, base, tools

creator.create("FitnessMin", base.Fitness, weights=(-1,))  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

h = 12          # head length
n_genes = 2    # number of genes in a chromosome

toolbox = gep.Toolbox()
toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=h)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes,linker=add)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# compile utility: which translates an individual into an executable function (Lambda)
toolbox.register('compile', gep.compile_, pset=pset)

# as a test I'm going to try and accelerate the fitness function
#@jit("void(i1[:])", nopython=True)
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images 
    print (err)
    return err
#@jit("void(i1[:])", nopython=True)
def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

#@jit("void(i1[:])", nopython=True)
def evaluate(individual):
    """Evalute the fitness of an individual: MAE (mean absolute error)"""
    func = toolbox.compile(individual)
    # below call the individual as a function over the inputs
    fitness = []
    Yp = np.array(list(map(func, *IMAGES.values())))
    sum_ = 0
    set_images = np.absolute(Y - Yp)

    for i in range(len(Yp)):
        ret,thresh1 = cv2.threshold(Yp[i].astype(np.uint8),127,255,cv2.THRESH_BINARY)
        fitness.append(mse(Y[i],thresh1))
    fitness_np = np.array(fitness)
    return np.mean(fitness_np),

toolbox.register('evaluate', evaluate)
toolbox.register('select', tools.selTournament, tournsize=3)

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# size of population and number of generations# size o 
n_pop = 100
n_gen = 10

champs = 3

pop = toolbox.population(n=n_pop) # 
hof = tools.HallOfFame(champs)   # only record the best three individuals ever found in all generations

startDT = datetime.datetime.now()
print (str(startDT))

# start evolution
pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1,
                          stats=stats, hall_of_fame=hof, verbose=True)

print ("Evolution times were:\n\nStarted:\t", startDT, "\nEnded:   \t", str(datetime.datetime.now()))

# print the best we found:
best_ind = hof[0]
print('BEST:', best_ind)
'''
symplified_best = gep.simplify(best_ind)

print('\n', key,'\t', str(symplified_best), '\n\nwhich formally is presented as:\n\n')'''

def CalculateBestModelOutput(model, context={}):
#                            IM_1, IM_2,
#                            IM_2,IM_3,IM_4,IM_5,IM_6,IM_7,
#                            IM_8,IM_9,IM_10,IM_11,IM_12,IM_13,IM_14,
#                            IM_15,
                            
    # pass in a string view of the "model" as str(symplified_best)
    # this string view of the equation may reference any of the other inputs, AT, V, AP, RH we registered
    # we then use eval of this string to calculate the answer for these inputs
    return eval(model, None, context) 


IMAGES_TEST = {}
names = []
for name in TERMINAL_AVAILABLE:
    if name == 'IM_0':
        continue
    settings = TERMINAL_SETTINGS[name]
    names.append(name)
    print ('Processing testing image...', name)
    IMAGES_TEST[name] = np.array([custom_filter(image=image, **settings) for image in np.array(test_X)])
IMAGES_TEST['IM_0'] = np.array(test_X)


print('Finish load testing images')
IM_0 = np.array(test_X)
for k in range(len(IMAGES_TEST[TERMINAL_AVAILABLE[0]])):
    predPE = CalculateBestModelOutput(
        str(best_ind),
        context={im: IMAGES_TEST[im][k] for im in TERMINAL_AVAILABLE}
    )
    cv2.imwrite("output/segmentation/imathresh"+str(k)+"A.png", predPE)
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
    cv2.imwrite("output/segmentation/imathresh"+str(k)+"B.png", x)          
    
print (len(IMAGES_TEST))
print (predPE.shape)
plot_log(log)
plot_log_curves(log)
