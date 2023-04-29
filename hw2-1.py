# reference https://github.com/qhan1028/Image-Stitching/tree/ae88b023ead6c86cb56cc216bd353fe7f9b260a0

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import random
from scipy import signal
import os.path as osp
import os


# filename1 = 'grail1'
# bgr1 = cv2.imread(f'{filename1}.jpg')
# rgb1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)

# # #print('bgr1 shape', bgr1.shape)

# filename2 = 'grail2'
# bgr2 = cv2.imread(f'{filename2}.jpg')
# rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)

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


# warp_1 = cylinder_warping2(bgr1, filename1, 600)
# # print("warpped shape", warp_1.shape)
# warp_2 = cylinder_warping2(bgr2, filename2, 603)

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

    return R, Ix, Iy
# def local_max_R(R, thres=0.01):
#     print('Calculating local max')
#     kernels = []
#     for y in range(3):
#         for x in range(3):
#             if x == 1 and y == 1: continue
#             k = np.zeros((3, 3), dtype=np.float32)
#             k[1, 1] = 1
#             k[y, x] = -1
#             kernels.append(k)

#     localMax = np.ones(R.shape, dtype=np.uint8)
#     localMax[R <= np.max(R) * thres] = 0

#     filtered_images = []

#     for k in kernels:
#         d = np.zeros(R.shape, dtype=np.float32)
#         for i in range(1, R.shape[0]-1):
#             for j in range(1, R.shape[1]-1):
#                 d[i, j] = np.sum(R[i-1:i+2, j-1:j+2] * k)
#         filtered_images.append(d)

#     for d in filtered_images:
#         d[d < 0] = 0
#         localMax &= np.uint8(np.sign(d))

#     print('# corners:', np.sum(localMax))
#     feature_points = np.where(localMax > 0)
    
#     return feature_points

def local_max_R(R, thres=0.01):
    print('Calculating local max')

    # create a single kernel
    kernel = np.ones((3, 3), dtype=np.float32) * -1
    kernel[1, 1] = 8

    # apply thresholding to the input image
    localMax = R > np.max(R) * thres

    # perform convolution using the kernel
    filtered_image = np.abs(signal.convolve2d(localMax, kernel, mode='same'))

    # find local maxima in the filtered image
    localMax = (filtered_image == np.max(filtered_image))

    print('# corners:', np.sum(localMax))
    feature_points = np.argwhere(localMax)
    
    return feature_points

# gray1 = cv2.cvtColor(warp_1.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
# gray2 = cv2.cvtColor(warp_2.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
# h1, w1 = gray1.shape
# h2, w2 = gray2.shape

# R1, Ix1, Iy1, Ix21, Iy21 = calculate_R(gray1)
# R2, Ix2, Iy2, Ix22, Iy22 = calculate_R(gray2)

# fpts1 = local_max_R(R1)
# fpts2 = local_max_R(R2)
#print('len', len(fpts1))
#print('points 0', fpts1[0][1], fpts1[0][0])
# img_fps1 = np.copy(bgr1)
# for i in range(len(fpts1[0])):
#     cv2.circle(img_fps1, (fpts1[1][i], fpts1[0][i]), radius=1, color=[0, 0, 255], thickness=1, lineType=1)

# img_arrows1 = np.copy(bgr1)
    
# for i in range(len(fpts1[0])):
#     x = fpts1[1][i]
#     y = fpts1[0][i]
#     ex, ey = int(x + Ix1[y, x]*2), int(y + Iy1[y, x]*2)
#     ex, ey = np.clip(ex, 0, w1), np.clip(ey, 0, h1)
#     cv2.arrowedLine(img_arrows1, (x, y), (ex, ey), (0, 0, 255), 1)

# cv2.imwrite(f'results/{filename1}_fps.png', img_fps1)
# cv2.imwrite(f'results/{filename1}_arrows.png', img_arrows1)

# img_fps2 = np.copy(bgr2)
# for i in range(len(fpts2[0])):
#     cv2.circle(img_fps2, (fpts2[1][i], fpts2[0][i]), radius=1, color=[0, 0, 255], thickness=1, lineType=1)

# img_arrows2 = np.copy(bgr2)
    
# for i in range(len(fpts2[0])):
#     x = fpts2[1][i]
#     y = fpts2[0][i]
#     ex, ey = int(x + Ix1[y, x]*2), int(y + Iy1[y, x]*2)
#     ex, ey = np.clip(ex, 0, w1), np.clip(ey, 0, h1)
#     cv2.arrowedLine(img_arrows2, (x, y), (ex, ey), (0, 0, 255), 1)

# cv2.imwrite(f'results/{filename2}_fps.png', img_fps2)
# cv2.imwrite(f'results/{filename2}_arrows.png', img_arrows2)

# class MSOP_descripter():
#     def __init__(self, coordinate, ori, patch):
#         self.coordinate = coordinate
#         self.ori = ori
#         self.patch = patch

def get_patches(rotated_img, pos):
    up = pos[0] - 20 if (pos[0] - 20) >= 0 else 0
    down = pos[0] + 20 if (pos[0] + 20) < rotated_img.shape[0] else rotated_img.shape[0]
    left = pos[1] - 20 if (pos[1] - 20) >= 0 else 0
    right = pos[1] + 20 if (pos[1] + 20) < rotated_img.shape[1] else rotated_img.shape[1]
    return rotated_img[up : down, left : right]

def get_MSOP_descripters(src_img, feature_pos, Ix, Iy):
    # src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    len_fp = len(feature_pos[0])
    desc_left_list = []
    desc_right_list = []
    for i in range(len_fp):
        pos = (int(feature_pos[0][i]), int(feature_pos[1][i]))
        # print("pos is ", pos)
        rotated_shape = cv2.getRotationMatrix2D(pos, math.atan2(Iy[pos], Ix[pos]), scale=1.0)
        rotated_img = cv2.warpAffine(src_img, rotated_shape, (src_img.shape[1], src_img.shape[0]))
        # if i == 0:
        #     cv2.imwrite(f'results/desc_test.png', rotated_img)
        patch_40 = get_patches(rotated_img, pos)
        # if i == 0:
        #     cv2.imwrite(f'results/40_test.png', patch_40)
        gauss_patch_40 = gaussian_filter(patch_40, sigma=1) # sigma = 4.5?
        patch_8 = cv2.resize(gauss_patch_40, (8, 8))
        # if i == 0:
        #     cv2.imwrite(f'results/8_test.png', patch_8)
        normal_patch_8 = (patch_8 - np.mean(patch_8)) / (patch_8 + 1e-8)
        # if i == 0:
        #     cv2.imwrite(f'results/normal_test.png', normal_patch_8)

        desc = {"coordinate": pos, "orientation": (Iy[pos], Ix[pos]), "patch": normal_patch_8.flatten().tolist()}
        if pos[1] < src_img.shape[1] // 2:
            desc_left_list.append(desc)
        else:
            desc_right_list.append(desc)
    return desc_left_list, desc_right_list

# desc_left_list1, desc_right_list1 = get_MSOP_descripters(bgr1, fpts1, Ix1, Iy1)
# desc_left_list2, desc_right_list2 = get_MSOP_descripters(bgr2, fpts2, Ix2, Iy2)
# print(desc_left_list, desc_right_list)

def match_feature(desc_right, desc_left, thresh_hold = 0.8):
    img1_right_desc = pd.DataFrame(desc_right)
    img2_left_desc = pd.DataFrame(desc_left)
    img1_all_right_patches = img1_right_desc.loc[:]["patch"].tolist()
    img2_all_left_patches = img2_left_desc.loc[:]["patch"].tolist()
    # print(img1_all_right_patches[0])
    all_combination_dist = cdist(img1_all_right_patches, img2_all_left_patches)
    sorted_index = np.argsort(all_combination_dist, axis=1)
    # print(sorted_index)
    # print(len(sorted_index))
    # print(len(sorted_index[0]))
    matched_indexes = []
    for i, j in enumerate(sorted_index):
        first_closest = all_combination_dist[i, j[0]]
        second_closest = all_combination_dist[i, j[1]]
        if first_closest / second_closest < thresh_hold:
            matched_indexes.append([i, j[0]])
    return matched_indexes

# matched_indexes = match_feature(desc_right_list1, desc_left_list2)
# print(matched_indexes)
# print(len(matched_indexes))
def warp_feature(h, w, desc_list, focal=1800):
    y_origin = h // 2
    x_origin = w // 2
    numm_descs = len(desc_list)
    for i in range(numm_descs):
        #print("original", desc_list[i]["coordinate"])
        x_prime = math.atan(((desc_list[i]["coordinate"][1] - x_origin) / focal)) * focal + x_origin
        y_prime =  focal * (desc_list[i]["coordinate"][0] - y_origin) /\
        math.sqrt((desc_list[i]["coordinate"][1] - x_origin)**2 + focal**2)+ y_origin
        desc_list[i]["coordinate"] = (round(y_prime), round(x_prime))
        #print("after", (round(y_prime), round(x_prime)))
    return
ransac_list = []
def ransac(desc_right, desc_left, matched_indexes):
    max_inlier = 0
    shift_xy = []
    for i in range(100000):
        num_chosen = 1
        chosen = random.sample(matched_indexes, k=num_chosen)
        # if i == 0:
        #     print("chosen", chosen)
        curr_shift = [desc_right[chosen[0][0]]["coordinate"][0] - desc_left[chosen[0][1]]["coordinate"][0], \
                      desc_right[chosen[0][0]]["coordinate"][1] - desc_left[chosen[0][1]]["coordinate"][1]]
        for j in range(1, num_chosen):
            curr_shift[0] += desc_right[chosen[j][0]]["coordinate"][0] - desc_left[chosen[j][1]]["coordinate"][0]
            curr_shift[1] += desc_right[chosen[j][0]]["coordinate"][1] - desc_left[chosen[j][1]]["coordinate"][1]
        curr_shift[0] /= num_chosen
        curr_shift[1] /= num_chosen
        # print(curr_shift)
        curr_inlier = 0
        choosed = []
        for g, h in matched_indexes:
            # print(i, j)
            if [g, h] not in chosen:
                other_shift = [desc_right[g]["coordinate"][0] - desc_left[h]["coordinate"][0], \
                               desc_right[g]["coordinate"][1] - desc_left[h]["coordinate"][1]]
                if abs(curr_shift[0] - other_shift[0]) < 4 and abs(curr_shift[1] - other_shift[1]) < 4:
                    curr_inlier += 1
                    choosed.append([g, h])
        # print(curr_inlier)
        if max_inlier < curr_inlier:
            max_inlier = curr_inlier
            for g, h in choosed:
                curr_shift[0] += desc_right[g]["coordinate"][0] - desc_left[h]["coordinate"][0]
                curr_shift[1] += desc_right[g]["coordinate"][1] - desc_left[h]["coordinate"][1]
            curr_shift[0] /= (max_inlier + 1)
            curr_shift[1] /= (max_inlier + 1)
            shift_xy = curr_shift
            global ransac_list
            ransac_list = choosed
            # print(curr_shift)
    # print(shift_xy)
    shift_xy[0] = round(shift_xy[0])
    shift_xy[1] = round(shift_xy[1])
    print("inliner num", max_inlier)
    return shift_xy

# shift_xy = ransac(desc_right_list1, desc_left_list2, matched_indexes)
# print(shift_xy)
# print(warp_1.shape)

def init_stitching_space(images, total_shifts):
    h, w, c = images[0].shape
    if total_shifts[0] > 0:
        h += total_shifts[0] # might need to check this
    w += total_shifts[1]
    print('init_snitch h w c', h, w, c)
    return np.zeros((h, w, c), dtype=np.uint8)

def image_stitching(stitching_space, images, all_shifts):
    cumulated_shifts = [0, 0]
    cumulated_h, cumulated_w, temp_c = images[0].shape
    num_images = len(images)
    stitching_space[:images[0].shape[0], :images[0].shape[1],:] = images[0]
    for i in range(num_images - 1):
        # print(i)
        cumulated_shifts[0] = cumulated_shifts[0] + all_shifts[i][0]
        cumulated_shifts[1] = cumulated_shifts[1] + all_shifts[i][1]
        overlapped_col = cumulated_w - cumulated_shifts[1]
        left_part = overlapped_col - 1
        right_part = 1 
        h, w, c = images[i + 1].shape
        if cumulated_shifts[0] >= 0:
            start_row = cumulated_shifts[0]  
            end_row = h + start_row
        else:
            start_row = 0
            end_row = h + cumulated_shifts[0]

        if cumulated_shifts[0] >= 0:
            im_start_row = 0
        else:
            im_start_row = -cumulated_shifts[0]
        
        for j in range(cumulated_shifts[1], cumulated_w):
            if (stitching_space[(start_row+(end_row)) // 2, (j - 1), 1] == 0):
                stitching_space[start_row:end_row, (j - 1), :] = images[i + 1][im_start_row:, (j - cumulated_shifts[1]), :]
            elif (images[i + 1][h // 2, (j - cumulated_shifts[1]), 1] == 0):
                pass
            else:
                # if (stitching_space[start_row:end_row, (j - 1), :].shape != images[i + 1][im_start_row:, (j - cumulated_shifts[1]), :].shape):
                stitching_space[start_row:end_row, (j - 1), :] = \
                stitching_space[start_row:end_row, (j - 1), :]  / overlapped_col * left_part + \
                images[i + 1][im_start_row:, (j - cumulated_shifts[1]), :] / overlapped_col * right_part
            left_part -= 1
            right_part += 1
        stitching_space[start_row:end_row, cumulated_w-1:w+cumulated_shifts[1], :] = \
        images[i + 1][im_start_row:, (cumulated_w - cumulated_shifts[1] - 1):, :]
        cumulated_h += all_shifts[i][0]
        cumulated_w += all_shifts[i][1]
        print(f"end {i}")
    return stitching_space.astype(np.uint8)

def image_stitching2(stitching_space, images, all_shifts):
    cumulated_shifts = [0, 0]
    cumulated_h, cumulated_w, temp_c = images[0].shape
    num_images = len(images)
    stitching_space[:images[0].shape[0], :images[0].shape[1],:] = images[0]
    for i in range(num_images - 1):
        # print(i)
        cumulated_shifts[0] = cumulated_shifts[0] + all_shifts[i][0]
        cumulated_shifts[1] = cumulated_shifts[1] + all_shifts[i][1]
        overlapped_col = cumulated_w - cumulated_shifts[1]
        left_part = overlapped_col - 1
        right_part = 1 
        h, w, c = images[i + 1].shape
        if cumulated_shifts[0] >= 0:
            start_row = cumulated_shifts[0]  
            end_row = h + start_row
        else:
            start_row = 0
            end_row = h + cumulated_shifts[0]

        if cumulated_shifts[0] >= 0:
            im_start_row = 0
        else:
            im_start_row = -cumulated_shifts[0]
        
        for j in range(cumulated_shifts[1], cumulated_w):
            if (j > (cumulated_shifts[1] + cumulated_w) / 2):
                stitching_space[start_row:end_row, (j - 1), :] = images[i + 1][im_start_row:, (j - cumulated_shifts[1]), :]
            # else:
            #     stitching_space[start_row:end_row, (j - 1), :] = \
            #     stitching_space[start_row:end_row, (j - 1), :]
            left_part -= 1
            right_part += 1
        stitching_space[start_row:end_row, cumulated_w-1:w+cumulated_shifts[1], :] = \
        images[i + 1][im_start_row:, (cumulated_w - cumulated_shifts[1] - 1):, :]
        cumulated_h += all_shifts[i][0]
        cumulated_w += all_shifts[i][1]
        print(f"end {i}")
    return stitching_space.astype(np.uint8)

filenames = []

bgrs = []

focals = [3085.73, 3082.65, 3087.25]

dir = 'pictures2'

for filename in np.sort(os.listdir(dir)):
    if osp.splitext(filename)[1] in ['.png', '.jpg']:
        print(filename)
        filenames.append(filename)
        im = cv2.imread(osp.join(dir, filename))
        bgrs.append(im) 

n = 3

warp = [cylinder_warping2(bgrs[i], filenames[i], focal=focals[i]) for i in range(n)]

grays = []

for i in range(n):
    grays.append(cv2.cvtColor(bgrs[i], cv2.COLOR_BGR2GRAY).astype(np.float32))

R, Ix, Iy = [], [], []

for i in range(n):
    r, ix, iy = calculate_R(grays[i])
    R.append(r)
    Ix.append(ix)
    Iy.append(iy)

def find_local_max_R(R, rthres=0.01):
    kernels = []
    for y in range(3):
        for x in range(3):
            if x == 1 and y == 1: continue
            k = np.zeros((3, 3), dtype=np.float32)
            k[1, 1] = 1
            k[y, x] = -1
            kernels.append(k)

    localMax = np.ones(R.shape, dtype=np.uint8)
    localMax[R <= np.max(R) * rthres] = 0

    for k in kernels:
        d = np.sign(cv2.filter2D(R, -1, k))
        d[d < 0] = 0
        localMax &= np.uint8(d)

    print('# corners:', np.sum(localMax))
    feature_points = np.where(localMax > 0)
    
    return feature_points

fpts = []

for i in range(n):
    fpts.append(find_local_max_R(R[i]))

desc_left_list, desc_right_list = [], []

for i in range(n):
    left, right = get_MSOP_descripters(bgrs[i], fpts[i], Ix[i], Iy[i])
    desc_left_list.append(left)
    desc_right_list.append(right)
# desc_left_list1, desc_right_list1 = get_MSOP_descripters(warp_1, fpts1, Ix1, Iy1)
# desc_left_list2, desc_right_list2 = get_MSOP_descripters(warp_2, fpts2, Ix2, Iy2)
# desc_left_list3, desc_right_list3 = get_MSOP_descripters(warp_3, fpts3, Ix3, Iy3)

test_height = bgrs[0].shape[0]
test_width = bgrs[0].shape[1]

for i in range(n):
    warp_feature(test_height, test_width, desc_left_list[i], focals[i])
    warp_feature(test_height, test_width, desc_right_list[i], focals[i])

matches = []
all_shifts = []
for i in range(n-1):
    matches.append(match_feature(desc_right_list[i], desc_left_list[i+1], thresh_hold=1))
    all_shifts.append(ransac(desc_right_list[i], desc_left_list[i+1], matches[i]))
print(all_shifts)

max_shifts = [0, 0]
for i, j in all_shifts:
    max_shifts[0] += max(max_shifts[0], i)
    max_shifts[1] += max(max_shifts[1], j)
print(max_shifts)
# stitching_space = init_stitching_space([warp_1, warp_2, warp_3], max_shifts)
# stitched = image_stitching(stitching_space, [warp_1, warp_2, warp_3], all_shifts)

stitching_space = init_stitching_space(bgrs, max_shifts)
stitched = image_stitching2(stitching_space, warp, all_shifts)

cv2.imwrite(f'results/stitched.png', stitched)

def find_range(array):
    sum_x = np.sum(array, axis=0)
    sum_y = np.sum(array, axis=1)
    
    index_x = np.where(sum_x > 0)[0]
    sx = index_x[0]
    ex = index_x[-1] + 1 # slicing
    index_y = np.where(sum_y > 0)[0]
    sy = index_y[0]
    ey = index_y[-1] + 1 # slicing
    
    return sx, ex, sy, ey

def bundle_adjust(img):
    h, w, _ = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    for x in range(0, w):
        find = False
        for y in range(0, h):
            if img_gray[y, x] > 0:
                upper_left = [x, y]
                find = True
                break
        if find:
            break

    for x in range(0, w):
        find = False
        for y in range(h - 1, -1, -1):
            if img_gray[y, x] > 0:
                bottom_left = [x, y]
                find = True
                break
        if find:
            break

    for x in range(w-1, -1, -1):
        find = False
        for y in range(0, h):
            if img_gray[y, x] > 0:
                upper_right = [x, y]
                find = True
                break
        if find:
            break

    for x in range(w-1, -1, -1):
        find = False
        for y in range(h-1, -1, -1):
            if img_gray[y, x] > 0:
                bottom_right = [x, y]
                find = True
                break
        if find:
            break

    corner = np.float32([upper_left, upper_right, bottom_left, bottom_right])
    img_corner = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    M = cv2.getPerspectiveTransform(corner, img_corner)
    pano_adjust = cv2.warpPerspective(img, M, (w, h))

    return pano_adjust


bundle = bundle_adjust(stitched)

cv2.imwrite(f'results/bundle.png', bundle)

