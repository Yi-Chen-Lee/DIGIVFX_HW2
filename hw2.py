import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from scipy.spatial.distance import cdist

filename1 = 'mountain1'
bgr1 = cv2.imread(f'{filename1}.png')
rgb1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)

#print('bgr1 shape', bgr1.shape)

filename2 = 'mountain2'
bgr2 = cv2.imread(f'{filename2}.png')
rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)

def cylinder_warping1(image, file_name, focal=1800):
    print('Cylinder Warping1')
    h, w, c = image.shape
    print('h', h, 'w', w)
    s = focal
    res = np.zeros([h, w, 3])
    y_origin = np.floor(h/2)
    x_origin = np.floor(w/2)
    
    for c in  range(3):
        for y_pic in range(h):
            for x_pic in range(w):
                y_prime = y_pic - y_origin
                x_prime = x_pic - x_origin
                x = focal * np.tan(x_prime/s)
                y = np.sqrt(x**2 + focal**2) / s * y_prime
                y += y_origin
                x += x_origin
                x_floor = min(max(int(np.floor(x)), 0), w-1)
                y_floor = min(max(int(np.floor(y)), 0), h-1)
                x_ceil = min(max(int(np.ceil(x)), 0), w-1)
                y_ceil = min(max(int(np.ceil(y)), 0), h-1)
                
                a = x - x_floor
                b = y - y_floor
                print('prime', y_prime, x_prime, 'real', x, y, 'floor', y_floor, x_floor, 'ceiling', y_ceil, x_ceil)
                res[y_pic][x_pic][c] = (1-a) * (1-b) * image[y_floor][x_floor][c] + a * (1-b) * image[y_floor][x_ceil][c] + a * b * image[y_ceil][x_ceil][c] + (1-a) * b * image[y_ceil][x_floor][c]
          
def cylinder_warping2(image, file_name, focal=1800):
    print('Cylinder Warping2')
    h, w, c = image.shape
    print('h', h, 'w', w)
    s = focal
    res = np.zeros([h, w, 3])
    y_origin = np.floor(h/2)
    x_origin = np.floor(w/2)

    y, x = np.indices((h, w))
    y_prime = y - y_origin
    x_prime = x - x_origin

    x = focal * np.tan(x_prime/s)
    y = np.sqrt(x**2 + focal**2) / s * y_prime
    y += y_origin
    x += x_origin

    x_floor = np.clip(np.floor(x).astype(int), 0, w-1)
    y_floor = np.clip(np.floor(y).astype(int), 0, h-1)
    x_ceil = np.clip(np.ceil(x).astype(int), 0, w-1)
    y_ceil = np.clip(np.ceil(y).astype(int), 0, h-1)

    idx = np.ones([h,w])
    idx[np.floor(x)<0] = 0; idx[np.floor(y)<0]=0; idx[np.ceil(x)>w-1]=0; idx[np.ceil(y)>h-1]=0

    a = x - x_floor
    b = y - y_floor

    for i in range(c):
        res[..., i] = (1-a) * (1-b) * image[y_floor, x_floor, i] + a * (1-b) * image[y_floor, x_ceil, i] + a * b * image[y_ceil, x_ceil, i] + (1-a) * b * image[y_ceil, x_floor, i]
    res[idx==0] = [0, 0, 0]
    cv2.imwrite(f'results/{file_name}_warp.png', res)
    return res


warp_1 = cylinder_warping2(bgr1, filename1, 1800)
warp_2 = cylinder_warping2(bgr2, filename2, 1800)

def calculate_R(gray, ksize=9, S=3, k=0.04):
    print('Calculating R')
    K = (ksize, ksize)

    gray_blur = cv2.GaussianBlur(gray, K, S)
    Iy, Ix = np.gradient(gray_blur)

    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    Sx2 = cv2.GaussianBlur(Ix2, K, S)
    Sy2 = cv2.GaussianBlur(Iy2, K, S)
    Sxy = cv2.GaussianBlur(Ixy, K, S)

    detM = (Sx2 * Sy2) - (Sxy ** 2)
    traceM = Sx2 + Sy2

    R = detM - k * (traceM ** 2)

    return R, Ix, Iy, Ix2, Iy2

def local_max_R(R, thres=0.01):
    print('Calculating local max')
    kernels = []
    for y in range(3):
        for x in range(3):
            if x == 1 and y == 1: continue
            k = np.zeros((3, 3), dtype=np.float32)
            k[1, 1] = 1
            k[y, x] = -1
            kernels.append(k)

    localMax = np.ones(R.shape, dtype=np.uint8)
    localMax[R <= np.max(R) * thres] = 0

    filtered_images = []

    for k in kernels:
        d = np.zeros(R.shape, dtype=np.float32)
        for i in range(1, R.shape[0]-1):
            for j in range(1, R.shape[1]-1):
                d[i, j] = np.sum(R[i-1:i+2, j-1:j+2] * k)
        filtered_images.append(d)

    for d in filtered_images:
        d[d < 0] = 0
        localMax &= np.uint8(np.sign(d))

    print('# corners:', np.sum(localMax))
    feature_points = np.where(localMax > 0)
    
    return feature_points

gray1 = cv2.cvtColor(bgr1, cv2.COLOR_RGB2GRAY).astype(np.float32)
gray2 = cv2.cvtColor(bgr2, cv2.COLOR_RGB2GRAY).astype(np.float32)
h1, w1 = gray1.shape
h2, w2 = gray2.shape

R1, Ix1, Iy1, Ix21, Iy21 = calculate_R(gray1)
R2, Ix2, Iy2, Ix22, Iy22 = calculate_R(gray2)

fpts1 = local_max_R(R1)
fpts2 = local_max_R(R2)
#print('len', len(fpts1))
#print('points 0', fpts1[0][1], fpts1[0][0])
img_fps1 = np.copy(bgr1)
for i in range(len(fpts1[0])):
    cv2.circle(img_fps1, (fpts1[1][i], fpts1[0][i]), radius=1, color=[0, 0, 255], thickness=1, lineType=1)

img_arrows1 = np.copy(bgr1)
    
for i in range(len(fpts1[0])):
    x = fpts1[1][i]
    y = fpts1[0][i]
    ex, ey = int(x + Ix1[y, x]*2), int(y + Iy1[y, x]*2)
    ex, ey = np.clip(ex, 0, w1), np.clip(ey, 0, h1)
    cv2.arrowedLine(img_arrows1, (x, y), (ex, ey), (0, 0, 255), 1)

cv2.imwrite(f'results/{filename1}_fps.png', img_fps1)
cv2.imwrite(f'results/{filename1}_arrows.png', img_arrows1)

img_fps2 = np.copy(bgr2)
for i in range(len(fpts2[0])):
    cv2.circle(img_fps2, (fpts2[1][i], fpts2[0][i]), radius=1, color=[0, 0, 255], thickness=1, lineType=1)

img_arrows2 = np.copy(bgr2)
    
for i in range(len(fpts2[0])):
    x = fpts2[1][i]
    y = fpts2[0][i]
    ex, ey = int(x + Ix1[y, x]*2), int(y + Iy1[y, x]*2)
    ex, ey = np.clip(ex, 0, w1), np.clip(ey, 0, h1)
    cv2.arrowedLine(img_arrows2, (x, y), (ex, ey), (0, 0, 255), 1)

cv2.imwrite(f'results/{filename2}_fps.png', img_fps2)
cv2.imwrite(f'results/{filename2}_arrows.png', img_arrows2)