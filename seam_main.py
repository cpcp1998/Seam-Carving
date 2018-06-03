#!/usr/bin/env python3
import argparse
import sys
import cv2
import numpy as np
import torch
import skimage.filters
import skimage.morphology
from multiprocessing import Pool
import itertools
import numba
from numba import jit, njit, prange
import math
import copy

device = torch.device('cuda')
# torch.set_num_threads(72)
pool_size = 20

# filters for computation of energy
filter_hori = torch.tensor([[[[[1, -1]]]]], dtype=torch.float32, device=device)
filter_vert = torch.tensor([[[[[1], [-1]]]]], dtype=torch.float32, device=device)
filter_diag = torch.tensor([[[[[1, 0], [0, -1]]]]], dtype=torch.float32, device=device)
filter_anti = torch.tensor([[[[[0, 1], [-1, 0]]]]], dtype=torch.float32, device=device)
filter_hori2 = torch.tensor([[[[[1, 0, -1]]]]], dtype=torch.float32, device=device)
sum_hori = torch.tensor([[[[1, 1]]]], dtype=torch.float32, device=device)
sum_vert = torch.tensor([[[[1], [1]]]], dtype=torch.float32, device=device)
sum_diag = torch.tensor([[[[1, 0], [0, 1]]]], dtype=torch.float32, device=device)
sum_anti = torch.tensor([[[[0, 1], [1, 0]]]], dtype=torch.float32, device=device)


def regular_energy(image):
    _, height, width = image.shape
    energy = torch.zeros((height, width), dtype=torch.float32, device=device)

    image = image.reshape((1, 1, 3, height, width))
    image = image.to(device)

    diff_hori = torch.nn.functional.conv3d(image, filter_hori)
    diff_vert = torch.nn.functional.conv3d(image, filter_vert)
    diff_diag = torch.nn.functional.conv3d(image, filter_diag)
    diff_anti = torch.nn.functional.conv3d(image, filter_anti)

    diff_hori = diff_hori.abs().mean(2)
    diff_vert = diff_vert.abs().mean(2)
    diff_diag = diff_diag.abs().mean(2)
    diff_anti = diff_anti.abs().mean(2)

    diff_hori = torch.nn.functional.conv2d(diff_hori, sum_hori, padding=(0, 1))
    diff_vert = torch.nn.functional.conv2d(diff_vert, sum_vert, padding=(1, 0))
    diff_diag = torch.nn.functional.conv2d(diff_diag, sum_diag, padding=(1, 1))
    diff_anti = torch.nn.functional.conv2d(diff_anti, sum_anti, padding=(1, 1))

    energy += diff_hori.reshape(height, width)
    energy += diff_vert.reshape(height, width)
    energy += diff_diag.reshape(height, width)
    energy += diff_anti.reshape(height, width)

    # energy[:,1:] += diff_hori
    # energy[:,:-1] += diff_hori
    # energy[1:,:] += diff_vert
    # energy[:-1,:] += diff_vert
    # energy[1:,1:] += diff_diag
    # energy[:-1,:-1] += diff_diag
    # energy[1:,:-1] += diff_anti
    # energy[:-1,1:] += diff_anti

    # image = image.reshape((height, width, 3))
    # directions = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    # for dir in directions:
    #    x0 = 1 if dir[0] == 1 else 0
    #    x1 = height - (1 if dir[0] == -1 else 0)
    #    y0 = 1 if dir[1] == 1 else 0
    #    y1 = width - (1 if dir[1] == -1 else 0)
    #    origin = image[x0:x1, y0:y1, :]
    #    new = image[height-x1:height-x0, width-y1:width-y0, :]
    #    diff = abs(origin-new).mean(2)
    #    energy[x0:x1, y0:y1] += diff

    weight = torch.full_like(energy, 8, device=device)
    weight[0, :] = 5
    weight[-1, :] = 5
    weight[:, 0] = 5
    weight[:, -1] = 5
    weight[0, 0] = 3
    weight[0, -1] = 3
    weight[-1, 0] = 3
    weight[-1, -1] = 3
    energy = energy / weight

    return energy.cpu()


def to_grayscale(image):
    _, height, width = image.shape
    image = image * 255.0
    image = image.to(torch.uint8)
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


@njit(parallel=True)
def local_entropy_master(image, entropy, split):
    height, width = image.shape

    for thread in prange(len(split)):
        local_entropy_worker(image, entropy, split[thread])


@njit(parallel=True)
def local_entropy_worker(image, entropy, params):
    height, width = image.shape
    radius = 4

    x0, x1, y0, y1 = params

    bin = [0] * 256
    x = x0
    y = y0
    direction = 1
    total = 0
    nplogp = 0

    for xx in range(max(0, x - radius), min(height, x + radius + 1)):
        for yy in range(max(0, y - radius), min(width, y + radius + 1)):
            bin[image[xx, yy]] += 1
            total += 1

    for i in range(256):
        if bin[i]:
            nplogp -= bin[i] * math.log(bin[i])

    while x < x1:
        res = math.log(total) + nplogp / total
        res /= math.log(2)

        entropy[x, y] = res

        y += direction

        if y >= y1 or y < y0:
            y -= direction
            direction = -direction
            x += 1
            if x >= x1:
                break

            xx = x - radius - 1
            if xx >= 0:
                for yy in range(max(0, y - radius), min(width, y + radius + 1)):
                    idx = image[xx, yy]
                    nplogp += bin[idx] * math.log(bin[idx])
                    bin[idx] -= 1
                    total -= 1
                    if bin[idx]:
                        nplogp -= bin[idx] * math.log(bin[idx])

            xx = x + radius
            if xx < height:
                for yy in range(max(0, y - radius), min(width, y + radius + 1)):
                    idx = image[xx, yy]
                    if bin[idx]:
                        nplogp += bin[idx] * math.log(bin[idx])
                    bin[idx] += 1
                    total += 1
                    nplogp -= bin[idx] * math.log(bin[idx])

        elif direction == 1:
            yy = y - radius - 1
            if yy >= 0:
                for xx in range(max(0, x - radius), min(height, x + radius + 1)):
                    idx = image[xx, yy]
                    nplogp += bin[idx] * math.log(bin[idx])
                    bin[idx] -= 1
                    total -= 1
                    if bin[idx]:
                        nplogp -= bin[idx] * math.log(bin[idx])

            yy = y + radius
            if yy < width:
                for xx in range(max(0, x - radius), min(height, x + radius + 1)):
                    idx = image[xx, yy]
                    if bin[idx]:
                        nplogp += bin[idx] * math.log(bin[idx])
                    bin[idx] += 1
                    total += 1
                    nplogp -= bin[idx] * math.log(bin[idx])

        else:
            yy = y + radius + 1
            if yy < width:
                for xx in range(max(0, x - radius), min(height, x + radius + 1)):
                    idx = image[xx, yy]
                    nplogp += bin[idx] * math.log(bin[idx])
                    bin[idx] -= 1
                    total -= 1
                    if bin[idx]:
                        nplogp -= bin[idx] * math.log(bin[idx])

            yy = y - radius
            if yy >= 0:
                for xx in range(max(0, x - radius), min(height, x + radius + 1)):
                    idx = image[xx, yy]
                    if bin[idx]:
                        nplogp += bin[idx] * math.log(bin[idx])
                    bin[idx] += 1
                    total += 1
                    nplogp -= bin[idx] * math.log(bin[idx])


# checkboard parallel
def local_entropy(image):
    _, height, width = image.shape
    image = to_grayscale(image)
    entropy = np.empty_like(image, dtype=np.float32)

    step = 80

    x_len = len(range(0, height, step))
    y_len = len(range(0, width, step))
    xy_ranges = list(itertools.product(range(0, height, step), range(0, width, step)))
    calc_ranges = [(x[0], min(height, x[0] + step), x[1], min(width, x[1] + step)) for x in xy_ranges]

    local_entropy_master(image, entropy, calc_ranges)

    # std = skimage.filters.rank.entropy(image, skimage.morphology.square(9))
    # print(abs(std - entropy).max())

    entropy = torch.from_numpy(entropy)
    entropy = entropy.to(torch.float32)

    return entropy


def energy_driver(image, type):
    if type >= 3:
        raise NotImplementedError

    energy = regular_energy(image)

    energy += local_entropy(image)

    return energy


def cumulate(energy, forward, image=None):
    """
    Return the cumulated energy matrix and the best choice at each pixel.
    0 stands for up-left, 1 stands for up, 2 stands for up-right
    """

    height, width = energy.shape
    output = energy.clone()
    choice = torch.empty_like(energy, dtype=torch.uint8)

    if not forward:
        for x in range(1, height):
            up = output[x - 1, :]
            left = torch.empty_like(up)
            right = torch.empty_like(up)
            left[1:] = up[:-1]
            left[0] = up[0]
            right[:-1] = up[1:]
            right[-1] = up[-1]
            choices = torch.stack((left, up, right), 1)
            min, choice[x, :] = torch.min(choices, 1)
            output[x, :] += min
    else:
        image = image.reshape((1, 1, 3, height, width))
        image = image.to(device)
        diff_hori2 = torch.nn.functional.conv3d(image, filter_hori2)
        diff_diag = torch.nn.functional.conv3d(image, filter_diag)
        diff_anti = torch.nn.functional.conv3d(image, filter_anti)
        diff_hori2 = diff_hori2.abs().mean(2).reshape(height, width - 2).cpu()
        diff_diag = diff_diag.abs().mean(2).reshape(height - 1, width - 1).cpu()
        diff_anti = diff_anti.abs().mean(2).reshape(height - 1, width - 1).cpu()
        diff_hori2 = torch.nn.functional.pad(diff_hori2, (1, 1))

        output += diff_hori2
        for x in range(1, height):
            up = output[x - 1, :]
            left = torch.empty_like(up)
            right = torch.empty_like(up)
            left[1:] = up[:-1] + diff_diag[x - 1, :]
            left[0] = up[0]
            right[:-1] = up[1:] + diff_anti[x - 1, :]
            right[-1] = up[-1]
            choices = torch.stack((left, up, right), 1)
            min, choice[x, :] = torch.min(choices, 1)
            output[x, :] += min

    return output, choice


def find_seam(cumulative_map, choice):
    height, width = cumulative_map.shape
    output = torch.empty(height, dtype=torch.int32)
    output[-1] = torch.argmin(cumulative_map[-1, :])
    for x in range(height - 2, -1, -1):
        c = choice[x + 1, output[x + 1]].to(torch.int32)
        c += output[x + 1] - 1
        c = max(0, c)
        c = min(width - 1, c)
        output[x] = c
    return output  # output is the seam


# 我们有两种方案，一种是我们搞出红线来,另一种是我们把红线删掉，各自有利有弊，所以应该保留这两种模式

"""   
def local_entropy_update(image,width, height):
    return

def local_gradient_update(image,width, height):
    return

def energy_update(image, width, height, seam)
    return
 """


# 把待处理的seam的坐标进行转换
def cumu_seam(seam_list, seam, numofseam, height):
    for x in range(numofseam):
        for y in range(height):
            if seam_list[x][y] < seam[y]:
                seam[y] += 2
    seam_list = np.insert(seam_list, numofseam, seam, 0)
    return seam_list


# 处理最终图像
def aug_image(image, seam_list, numofseam, height, width):
    print(image.size())
    #print(seam_list)
    for x in range(numofseam):
        # remain check
        #print(image.size())
        #print(seam_list.size)
        chunk = np.zeros((3, height))
        image = np.insert(image, width, chunk, 2)
        width += 1
        #print(image.size())
        image_t = copy.deepcopy(image)
        for y in range(height):
            #print(seam_list[x][y])
            #image[:, y, seam_list[x][y]:-1]
            image[:, y, seam_list[x][y] + 1:] = image_t[:, y, seam_list[x][y]:-1]
            # image[:, y, seam_list[x][y] + 1:] = image_t
    print(image.size())
    return image





# 实验证明转置几乎没有代价
# 暂时不考虑DP的优化,主要时间花在能量和entropy的计算
def process_driver(image, width, height, type):  # 这里的宽高指的是输入的宽高
    if type > 1:
        raise NotImplementedError
    # 我们先删列再删行
    energy = energy_driver(image, type)
    image_height, image_width = energy.shape
    # 放大的数量
    chunksize = 100
    if image_width >= width:
        while image_width > width:
            cumulative_map, choice = cumulate(energy, True, image)
            seam = find_seam(cumulative_map, choice)
            # image = np.delete(image,seam,1)

            for x in range(image_height):
                if x == image_height - 1:
                    continue
                image[:, x, seam[x]:-1] = image[:, x, seam[x] + 1:]
            image_width = image_width - 1
            image = np.delete(image, image_width, 2)
            energy = energy_driver(image, type)
    else:

        auc_size = width - image_width
        updated_width = image_width
        while auc_size > 0:
            energy = energy_driver(image, type)
            print(image.size())
            image_mask = copy.deepcopy(image)
            image_width = updated_width
            this_size = 0
            if auc_size >= chunksize:
                this_size = chunksize
                auc_size -= chunksize
            else:
                this_size = auc_size
                auc_size = 0
            numofseam = 0
            #seam_list = np.array([[0]])
            for need_deal in range(this_size):
                cumulative_map, choice = cumulate(energy, True, image_mask)
                seam = find_seam(cumulative_map, choice)
                # image = np.delete(image,seam,1)
                for x in range(image_height):
                    if x == image_height - 1:
                        continue
                    image_mask[:, x, seam[x]:-1] = image_mask[:, x, seam[x] + 1:]
                image_width = image_width - 1
                image_mask = np.delete(image_mask, image_width, 2)
                energy = energy_driver(image_mask, type)
                if numofseam == 0:
                    seam_list = np.array([seam.numpy()])
                else:
                    seam_list = cumu_seam(seam_list, seam.numpy(), numofseam, image_height)
                numofseam += 1
            image = aug_image(image, seam_list, numofseam, image_height, updated_width)
            updated_width += this_size

    image = np.transpose(image, (0, 2, 1))
    energy = energy_driver(image, type)
    while image_height > height:
        # energy = energy_driver(image, type)
        cumulative_map, choice = cumulate(energy, True, image)
        seam = find_seam(cumulative_map, choice)
        # image = np.delete(image,seam,1)

        for x in range(image_width):
            if x == image_width - 1:
                continue
            image[:, x, seam[x]:-1] = image[:, x, seam[x] + 1:]
        image_height = image_height - 1
        image = np.delete(image, image_height, 2)
        energy = energy_driver(image, type)
    image = np.transpose(image, (0, 2, 1))
    return image


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description='Resize images by seam carving.',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('filename_in', metavar='SOURCE',
                        help='path to the input image')
    parser.add_argument('width', metavar='W', type=int,
                        help='number of columns of the resized output image')
    parser.add_argument('height', metavar='H', type=int,
                        help='number of rows of the resized output image')
    parser.add_argument('energy_type', metavar='TYPE', type=int,
                        choices=range(0, 4), help='energy options, where\n'
                                                  '0 = regular energy without entropy term\n'
                                                  '1 = regular energy with entropy term\n'
                                                  '2 = forward energy\n'
                                                  '3 = deep-based energy')
    parser.add_argument('filename_out', metavar='DEST',
                        help='path to the output image')
    args = parser.parse_args()

    # read the input
    image_in = cv2.imread(args.filename_in)
    if image_in is None:
        print('Cannot open the input image.')
        sys.exit(1)
    image_in = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
    image_in = np.transpose(image_in, (2, 0, 1))
    image_in = torch.from_numpy(image_in)
    image_in = image_in.to(torch.float32)
    image_in /= 255.0
    # image_in should be a 3*H*W pytorch tensor of type float32

    # process image
    image_out = process_driver(image_in, args.width, args.height, args.energy_type)

    # image_out should be a 3*H*W pytorch tensor of type float32
    # write the output
    image_out *= 255.0
    image_out = image_out.to(torch.uint8)
    image_out = image_out.numpy()
    image_out = np.transpose(image_out, (1, 2, 0))
    image_out = cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.filename_out, image_out)


if __name__ == "__main__":
    main()
