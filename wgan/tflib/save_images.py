# -*- coding:utf-8 -*-
"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
import scipy.misc
from scipy.misc import imsave

def save_images(X, save_path): # x shape(100,3,32,32)
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples)) # 行数是样本数的开平方取整，最好样本数还能整除rows
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples/rows # 设定img的高度(行数)和宽度(列数，一列代表一类)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2] # 每一张小图片的宽高，都为32,32
        img = np.zeros((h*nh, w*nw, 3)) # 原来他是按照数组的方式存储的图片,一张大图片的边长等于十张32x32的小图片的和，所以img的高H和宽W都为32*10 = 320
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):  # X中每个样本的索引、值枚举出来；而且X已经转换过维度了
        j = n/nw # n/10
        i = n%nw # n%10
        img[j*h : j*h+h , i*w : i*w+w] = x

    imsave(save_path, img)

def show_images(X):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99 * X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples / rows

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0, 2, 3, 1)
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw))

    for n, x in enumerate(X):
        j = n / nw
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w] = x

    imshow(img)