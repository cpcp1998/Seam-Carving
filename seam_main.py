#!/usr/bin/env python3
import argparse
import sys
import cv2
import numpy as np
import torch
import skimage.filters
import skimage.morphology
from multiprocessing import Pool

device = torch.device('cuda')
#torch.set_num_threads(72)
pool_size = 20

#filters for computation of energy
filter_hori = torch.tensor([[[[[1,-1]]]]], dtype=torch.float32, device=device)
filter_vert = torch.tensor([[[[[1],[-1]]]]], dtype=torch.float32, device=device)
filter_diag = torch.tensor([[[[[1,0],[0,-1]]]]], dtype=torch.float32, device=device)
filter_anti = torch.tensor([[[[[0,1],[-1,0]]]]], dtype=torch.float32, device=device)
filter_hori2 = torch.tensor([[[[[1,0,-1]]]]], dtype=torch.float32, device=device)
sum_hori = torch.tensor([[[[1,1]]]], dtype=torch.float32, device=device)
sum_vert = torch.tensor([[[[1],[1]]]], dtype=torch.float32, device=device)
sum_diag = torch.tensor([[[[1,0],[0,1]]]], dtype=torch.float32, device=device)
sum_anti = torch.tensor([[[[0,1],[1,0]]]], dtype=torch.float32, device=device)

def regular_energy(image):
    _, height, width = image.shape
    energy = torch.zeros((height, width), dtype=torch.float32, device=device)

    image = image.reshape((1, 1, 3, height, width))

    diff_hori = torch.nn.functional.conv3d(image, filter_hori)
    diff_vert = torch.nn.functional.conv3d(image, filter_vert)
    diff_diag = torch.nn.functional.conv3d(image, filter_diag)
    diff_anti = torch.nn.functional.conv3d(image, filter_anti)

    diff_hori = diff_hori.abs().mean(2)
    diff_vert = diff_vert.abs().mean(2)
    diff_diag = diff_diag.abs().mean(2)
    diff_anti = diff_anti.abs().mean(2)

    diff_hori = torch.nn.functional.conv2d(diff_hori, sum_hori, padding=(0,1))
    diff_vert = torch.nn.functional.conv2d(diff_vert, sum_vert, padding=(1,0))
    diff_diag = torch.nn.functional.conv2d(diff_diag, sum_diag, padding=(1,1))
    diff_anti = torch.nn.functional.conv2d(diff_anti, sum_anti, padding=(1,1))

    energy += diff_hori.reshape(height, width)
    energy += diff_vert.reshape(height, width)
    energy += diff_diag.reshape(height, width)
    energy += diff_anti.reshape(height, width)

    #energy[:,1:] += diff_hori
    #energy[:,:-1] += diff_hori
    #energy[1:,:] += diff_vert
    #energy[:-1,:] += diff_vert
    #energy[1:,1:] += diff_diag
    #energy[:-1,:-1] += diff_diag
    #energy[1:,:-1] += diff_anti
    #energy[:-1,1:] += diff_anti

    #image = image.reshape((height, width, 3))
    #directions = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    #for dir in directions:
    #    x0 = 1 if dir[0] == 1 else 0
    #    x1 = height - (1 if dir[0] == -1 else 0)
    #    y0 = 1 if dir[1] == 1 else 0
    #    y1 = width - (1 if dir[1] == -1 else 0)
    #    origin = image[x0:x1, y0:y1, :]
    #    new = image[height-x1:height-x0, width-y1:width-y0, :]
    #    diff = abs(origin-new).mean(2)
    #    energy[x0:x1, y0:y1] += diff

    weight = torch.full_like(energy, 8, device=device)
    weight[0,:] = 5
    weight[-1,:] = 5
    weight[:,0] = 5
    weight[:,-1] =5
    weight[0,0] = 3
    weight[0,-1] = 3
    weight[-1,0] = 3
    weight[-1,-1] = 3
    energy = energy / weight

    return energy

def to_grayscale(image):
    _, height, width = image.shape
    image = image.cpu()
    image *= 255.0
    image = image.to(torch.uint8)
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def local_entropy_worker(image):
    return skimage.filters.rank.entropy(image, skimage.morphology.square(9))

#process pool for local entropy calculation
pool = Pool(pool_size)

def local_entropy(image):
    _, height, width = image.shape
    image = to_grayscale(image)

    if height <= 9:
        return skimage.filters.rank.entropy(image, skimage.morphology.square(9))

    pool_size_eff = min(pool_size, height-8)
    lines = height + 8 * (pool_size_eff - 1)
    calc_ranges = [(lines*i//pool_size_eff-i*8, lines*(i+1)//pool_size_eff-i*8)
            for i in range(pool_size_eff)]
    image_slice = [image[x[0]:x[1]] for x in calc_ranges]
    entropy_slice = pool.map(local_entropy_worker, image_slice)

    for i in range(1, pool_size_eff - 1):
        entropy_slice[i] = entropy_slice[i][4:-4]

    entropy_slice[0] = entropy_slice[0][:-4]
    entropy_slice[-1] = entropy_slice[-1][4:]

    entropy = np.concatenate(entropy_slice)
    entropy = torch.from_numpy(entropy)
    entropy = entropy.to(torch.float32)
    entropy = entropy.to(device)

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
            up = output[x-1,:]
            left = torch.empty_like(up)
            right = torch.empty_like(up)
            left[1:] = up[:-1]
            left[0] = up[0]
            right[:-1] = up[1:]
            right[-1] = up[-1]
            choices = torch.stack((left,up,right),1)
            min, choice[x,:] = torch.min(choices, 1)
            output[x,:] += min
    else:
        image = image.reshape((1, 1, 3, height, width))
        diff_hori2 = torch.nn.functional.conv3d(image, filter_hori2)
        diff_diag = torch.nn.functional.conv3d(image, filter_diag)
        diff_anti = torch.nn.functional.conv3d(image, filter_anti)
        diff_hori2 = diff_hori2.abs().mean(2).reshape(height, width-2)
        diff_diag = diff_diag.abs().mean(2).reshape(height-1, width-1)
        diff_anti = diff_anti.abs().mean(2).reshape(height-1, width-1)
        diff_hori2 = torch.nn.functional.pad(diff_hori2, (1,1))

        output+= diff_hori2
        for x in range(1, height):
            up = output[x-1,:]
            left = torch.empty_like(up)
            right = torch.empty_like(up)
            left[1:] = up[:-1] + diff_diag[x-1,:]
            left[0] = up[0]
            right[:-1] = up[1:] + diff_anti[x-1,:]
            right[-1] = up[-1]
            choices = torch.stack((left,up,right),1)
            min, choice[x,:] = torch.min(choices, 1)
            output[x,:] += min

    return output, choice

def find_seam(cumulative_map, choice):
    height, width = cumulative_map.shape
    output = torch.empty(height, dtype=torch.int32, device=device)
    output[-1] = torch.argmin(cumulative_map[-1,:])
    for x in range(height-2, -1, -1):
        c = choice[x+1, output[x+1]].to(torch.int32)
        c += output[x+1] - 1
        c = max(0, c)
        c = min(width-1, c)
        output[x] = c
    return output

def process_driver(image, width, height, type):
    if type > 1:
        raise NotImplementedError

    #calculate energy
    energy = energy_driver(image, type)

    #calculate cumulative map by dynamic programming
    cumulative_map, choice = cumulate(energy, False, image)

    seam = find_seam(cumulative_map, choice)

    height, width = energy.shape

    for x in range(height):
        y = seam[x]
        image[0, x, y] = 1.0
        image[1, x, y] = image[2, x, y] = 0.0

    return image

def main():
    #parse command line arguments
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
            choices=range(0,4), help='energy options, where\n'
            '0 = regular energy without entropy term\n'
            '1 = regular energy with entropy term\n'
            '2 = forward energy\n'
            '3 = deep-based energy')
    parser.add_argument('filename_out', metavar='DEST',
            help='path to the output image')
    args = parser.parse_args()

    #read the input
    image_in = cv2.imread(args.filename_in)
    if image_in is None:
        print('Cannot open the input image.')
        sys.exit(1)
    image_in = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
    image_in = np.transpose(image_in, (2, 0, 1))
    image_in = torch.from_numpy(image_in)
    image_in = image_in.to(torch.float32)
    image_in = image_in.to(device)
    image_in /= 255.0
    #image_in should be a 3*H*W pytorch tensor of type float32

    #process image
    image_out = process_driver(image_in, args.width, args.height, args.energy_type)

    #image_out should be a 3*H*W pytorch tensor of type float32
    #write the output
    image_out *= 255.0
    image_out = image_out.cpu()
    image_out = image_out.to(torch.uint8)
    image_out = image_out.numpy()
    image_out = np.transpose(image_out, (1, 2, 0))
    image_out = cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.filename_out, image_out)


if __name__ == "__main__":
    main()
