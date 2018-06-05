#!/usr/bin/env python3
import argparse
import sys
import cv2
import numpy as np
import numba
from numba import jit, njit, prange
import math


@njit(parallel=False, cache=True)
def remove_seam_energy(energy, seam):
    height, width = energy.shape
    for x in prange(height):
        if seam[x] == width - 1:
            continue
        energy[x, seam[x]:-1] = energy[x, seam[x] + 1:]

@njit(parallel=False, cache=True)
def seam_range_worker(res, height, width, seam, radius):
    blocks = 0

    left = width
    right = 0
    first = 0

    criteria = 2 * (2 * radius + 1)

    for x in range(height):
        if right - left > criteria and x - first > criteria:
            res[blocks, 0] = max(0, first - radius)
            res[blocks, 1] = min(height, x + radius)
            res[blocks, 2] = left
            res[blocks, 3] = right
            blocks += 1
            left = width
            right = 0
            first = x

        if seam[x] - radius < left:
            left = max(0, seam[x] - radius)
        if seam[x] + radius > right:
            right = min(seam[x] + radius, width)

    res[blocks, 0] = max(0, first - radius)
    res[blocks, 1] = height
    res[blocks, 2] = left
    res[blocks, 3] = right
    blocks += 1

    return blocks

def seam_range(height, width, seam, radius):
    res = np.empty((height, 4), dtype=np.int)
    blocks = seam_range_worker(res, height, width, seam, radius)
    return res[:blocks, :]


@njit(parallel=False, cache=True)
def regular_energy_master(image, energy, split):
    for thread in prange(len(split)):
        regular_energy_worker(image, energy, split[thread])

@njit(parallel=False, cache=True)
def regular_energy_worker(image, energy, params):
    _, height, width = image.shape
    x0, x1, y0, y1 = params

    for x in range(x0, x1):
        for y in range(y0, y1):
            tot = -3
            diff = 0.0
            for xx in (x - 1, x, x + 1):
                for yy in (y - 1, y, y + 1):
                    if xx < 0 or xx >= height or yy < 0 or yy >= width:
                        continue

                    tot += 3
                    diff += abs(image[0, x, y] - image[0, xx, yy])
                    diff += abs(image[1, x, y] - image[1, xx, yy])
                    diff += abs(image[2, x, y] - image[2, xx, yy])

            energy[x, y] = diff / tot

#@jit(parallel=False)
def regular_energy(image):
    _, height, width = image.shape
    energy = np.empty((height, width), dtype=np.float32)

    #std = regular_energy_old(image)

    step = 80

    x_len = len(range(0, height, step))
    y_len = len(range(0, width, step))
    calc_ranges = np.empty((x_len * y_len, 4), dtype=np.int)
    for idx1, x in enumerate(range(0, height, step)):
        for idx2, y in enumerate(range(0, width, step)):
            idx = idx1 * y_len + idx2
            calc_ranges[idx, 0] = x
            calc_ranges[idx, 1] = min(height, x + step)
            calc_ranges[idx, 2] = y
            calc_ranges[idx, 3] = min(width, y + step)

    regular_energy_master(image, energy, calc_ranges)

    #print(abs(std - energy).max())
    return energy

def update_regular_energy(image, energy, seam):
    _, height, width = image.shape
    calc_ranges = seam_range(height, width, seam, 1)
    regular_energy_master(image, energy, calc_ranges)

@njit(parallel=False, cache=True)
def cvtColor_worker(input, output, params):
    x0, x1, y0, y1 = params

    for x in range(x0, x1):
        for y in range(y0, y1):
            output[x, y] = int(255 * (input[0, x, y] * 0.299
                    + input[1, x, y] * 0.587 + input[2, x, y] * 0.114))

@njit(parallel=False, cache=True)
def cvtColor_master(input, output, split):
    for thread in prange(len(split)):
        cvtColor_worker(input, output, split[thread])

def to_grayscale(image, split):
    _, height, width = image.shape
    output = np.empty((height, width), np.uint8)
    cvtColor_master(image, output, split)

    return output

@njit(parallel=False, cache=True)
def local_entropy_master(image, entropy, split):
    height, width = image.shape

    for thread in prange(len(split)):
        local_entropy_worker(image, entropy, split[thread])


@njit(parallel=False, cache=True)
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
    entropy = np.empty((height, width), dtype=np.float32)

    step = 80

    x_len = len(range(0, height, step))
    y_len = len(range(0, width, step))
    calc_ranges = np.empty((x_len * y_len, 4), dtype=np.int)
    for idx1, x in enumerate(range(0, height, step)):
        for idx2, y in enumerate(range(0, width, step)):
            idx = idx1 * y_len + idx2
            calc_ranges[idx, 0] = x
            calc_ranges[idx, 1] = min(height, x + step)
            calc_ranges[idx, 2] = y
            calc_ranges[idx, 3] = min(width, y + step)
    y_len = len(range(0, width, step))

    image = to_grayscale(image, calc_ranges)
    local_entropy_master(image, entropy, calc_ranges)

    # std = skimage.filters.rank.entropy(image, skimage.morphology.square(9))
    # print(abs(std - entropy).max())

    return entropy

def update_local_entropy(image, energy, seam):
    _, height, width = image.shape
    calc_ranges = seam_range(height, width, seam, 4)
    image_ranges = seam_range(height, width, seam, 8)
    image = to_grayscale(image, image_ranges)
    local_entropy_master(image, energy, calc_ranges)

last_regular_energy = None
last_local_entropy = None

def energy_driver(image, type, seam=None):
    global last_regular_energy
    global last_local_entropy

    if type >= 3:
        raise NotImplementedError

    if not seam is None:
        remove_seam_energy(last_regular_energy, seam)
        last_regular_energy = last_regular_energy[:, :-1]
        update_regular_energy(image, last_regular_energy, seam)
    else:
        last_regular_energy = regular_energy(image)

    energy = 8 * last_regular_energy

    if not seam is None:
        remove_seam_energy(last_local_entropy, seam)
        last_local_entropy = last_local_entropy[:, :-1]
        update_local_entropy(image, last_local_entropy, seam)
    else:
        last_local_entropy = local_entropy(image)

    energy += last_local_entropy

    return energy



@njit(parallel=False, cache=True)
def cumulate_upper_worker(energy, image, output, choice, params):
    height, width = energy.shape

    x0, x1, y0, y1 = params

    for i, x in enumerate(range(x0, x1)):
        if x >= height:
            break
        for y in range(y0 + i, y1 - i):
            if y < 0:
                continue
            if y >= width:
                break
            left_edge = (y == 0)
            right_edge = (y == width - 1)

            base = energy[x, y]
            if not left_edge and not right_edge:
                diff = abs(image[0, x, y - 1] - image[0, x, y + 1])
                diff += abs(image[1, x, y - 1] - image[1, x, y + 1])
                diff += abs(image[2, x, y - 1] - image[2, x, y + 1])
                base += diff / 3

            best = output[x - 1, y]
            best_choice = 1
            if not left_edge:
                left = output[x - 1, y - 1]
                diff = abs(image[0, x, y] - image[0, x - 1, y - 1])
                diff += abs(image[1, x, y] - image[1, x - 1, y - 1])
                diff += abs(image[2, x, y] - image[2, x - 1, y - 1])
                left += diff / 3
                if left < best:
                    best = left
                    best_choice = 0

            if not right_edge:
                right = output[x - 1, y + 1]
                diff = abs(image[0, x, y] - image[0, x - 1, y + 1])
                diff += abs(image[1, x, y] - image[1, x - 1, y + 1])
                diff += abs(image[2, x, y] - image[2, x - 1, y + 1])
                right += diff / 3
                if right < best:
                    best = right
                    best_choice = 2

            output[x, y] = best + base
            choice[x, y] = best_choice


@njit(parallel=False, cache=True)
def cumulate_lower_worker(energy, image, output, choice, params):
    height, width = energy.shape

    x0, x1, y0, y1 = params

    for i, x in enumerate(range(x0, x1)):
        if x >= height:
            break
        for y in range(y0 - i, y1 + i):
            if y < 0:
                continue
            if y >= width:
                break
            left_edge = (y == 0)
            right_edge = (y == width - 1)

            base = energy[x, y]
            if not left_edge and not right_edge:
                diff = abs(image[0, x, y - 1] - image[0, x, y + 1])
                diff += abs(image[1, x, y - 1] - image[1, x, y + 1])
                diff += abs(image[2, x, y - 1] - image[2, x, y + 1])
                base += diff / 3

            best = output[x - 1, y]
            best_choice = 1
            if not left_edge:
                left = output[x - 1, y - 1]
                diff = abs(image[0, x, y] - image[0, x - 1, y - 1])
                diff += abs(image[1, x, y] - image[1, x - 1, y - 1])
                diff += abs(image[2, x, y] - image[2, x - 1, y - 1])
                left += diff / 3
                if left < best:
                    best = left
                    best_choice = 0

            if not right_edge:
                right = output[x - 1, y + 1]
                diff = abs(image[0, x, y] - image[0, x - 1, y + 1])
                diff += abs(image[1, x, y] - image[1, x - 1, y + 1])
                diff += abs(image[2, x, y] - image[2, x - 1, y + 1])
                right += diff / 3
                if right < best:
                    best = right
                    best_choice = 2

            output[x, y] = best + base
            choice[x, y] = best_choice

@njit(parallel=False, cache=True)
def cumulate_helper(energy, image):
    height, width = energy.shape
    output = energy.copy()
    choice = np.empty_like(energy, dtype=np.uint8)

    for y in range(1, width - 1):
        output[0, y] += abs(image[0, 0, y - 1] - image[0, 0, y + 1]) / 3
        output[0, y] += abs(image[1, 0, y - 1] - image[1, 0, y + 1]) / 3
        output[0, y] += abs(image[2, 0, y - 1] - image[2, 0, y + 1]) / 3

    x_step = 40
    y_step = x_step * 2 - 1

    y_len_1 = len(range(0, width, y_step))
    y_len_2 = len(range(-1, width + x_step - 2, y_step))

    for x in range(1, height, x_step):
        for idx in prange(y_len_1):
            y = idx * y_step
            cumulate_upper_worker(energy, image, output, choice, (x, x + x_step, y, y + y_step))
        for idx in prange(y_len_2):
            y = -1 + idx * y_step
            cumulate_lower_worker(energy, image, output, choice, (x + 1, x + x_step, y, y + 2))

    return output, choice

@njit(parallel=False, cache=True)
def cumulate(energy, forward, image=None):
    """
    Return the cumulated energy matrix and the best choice at each pixel.
    0 stands for up-left, 1 stands for up, 2 stands for up-right
    """

    #std, std_c = cumulate_old(energy, forward, image)

    height, width = energy.shape
    if forward:
        pass
    else:
        image = np.zeros((3, height, width), dtype=np.float32)

    output, choice = cumulate_helper(energy, image)

    #print(abs(std - output).max())

    return output, choice

@njit(parallel=False, cache=True)
def find_seam(cumulative_map, choice):
    height, width = cumulative_map.shape
    output = np.empty(height, dtype=np.int32)
    output[-1] = np.argmin(cumulative_map[-1, :])
    for x in range(height - 2, -1, -1):
        c = choice[x + 1, output[x + 1]]
        c += output[x + 1] - 1
        c = max(0, c)
        c = min(width - 1, c)
        output[x] = c
    return output  # output is the seam


# 我们有两种方案，一种是我们搞出红线来,另一种是我们把红线删掉，各自有利有弊，所以应该保留这两种模式


#def local_entropy_update(image,width, height):
#    return
#
#def local_gradient_update(image,width, height):
#    return
#
#def energy_update(image, width, height, seam)
#    return



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
        image_t = image.copy()
        for y in range(height):
            #print(seam_list[x][y])
            #image[:, y, seam_list[x][y]:-1]
            image[:, y, seam_list[x][y] + 1:] = image_t[:, y, seam_list[x][y]:-1]
            # image[:, y, seam_list[x][y] + 1:] = image_t
    print(image.size())
    return image

@njit(parallel=False, cache=True)
def remove_seam(image, seam):
    _, height, width = image.shape
    for x in prange(height):
        if seam[x] == width - 1:
            continue
        image[:, x, seam[x]:-1] = image[:, x, seam[x] + 1:]


# 实验证明转置几乎没有代价
# 暂时不考虑DP的优化,主要时间花在能量和entropy的计算
def process_driver(image, width, height, type):  # 这里的宽高指的是输入的宽高
    if type > 1:
        raise NotImplementedError
    # 我们先删列再删行
    _, image_height, image_width = image.shape
    # 放大的数量
    chunksize = 100
    if image_width >= width:
        seam = None
        while image_width > width:
            energy = energy_driver(image, type, seam)
            cumulative_map, choice = cumulate(energy, True, image)
            seam = find_seam(cumulative_map, choice)
            # image = np.delete(image,seam,1)

            remove_seam(image, seam)
            image_width = image_width - 1
            image = image[:,:,:-1]
    else:
        auc_size = width - image_width
        updated_width = image_width
        while auc_size > 0:
            energy = energy_driver(image, type)
            print(image.size())
            image_mask = image.copy()
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
    image_in = image_in.astype(np.float32)
    image_in /= 255.0
    # image_in should be a 3*H*W numpy array of type float32

    # process image
    image_out = process_driver(image_in, args.width, args.height, args.energy_type)

    # image_out should be a 3*H*W numpy array of type float32
    # write the output
    image_out *= 255.0
    image_out = image_out.astype(np.uint8)
    image_out = np.transpose(image_out, (1, 2, 0))
    image_out = cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.filename_out, image_out)


if __name__ == "__main__":
    main()
