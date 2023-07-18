import os
import glob
import argparse
import random
import math
import numpy as np
import tifffile as tiff
import cv2
from tqdm import tqdm


def preprocess_and_save_sample(data_tensor, label, save_dir, dataset_name, idx):
    image = data_tensor
    image = image.astype(np.float32)
    min_value = np.min(image)
    image = image - min_value
    max_value = np.max(image)
    max_value = max(max_value, 1)
    image = image / max_value
    image = image * 255
    image = image.astype(np.uint8)

    output_filename = os.path.join(save_dir, '{}_{:03d}.npy'.format(dataset_name, idx))
    # print('Writing into {}'.format(output_filename))
    np.savez_compressed(output_filename, B = image, L = label)


def divide_images_into_tiles(image, tile_size, channels=False):
    n = image.shape[0]
    num_subarrays = n // tile_size  # Number of subarrays in each dimension

    image = image[:(num_subarrays * tile_size), :(num_subarrays * tile_size)]
    if channels == True:
        subarrays = np.reshape(image, (num_subarrays, tile_size, num_subarrays, tile_size, 4))
    else:
        subarrays = np.reshape(image, (num_subarrays, tile_size, num_subarrays, tile_size))

    subarrays = np.swapaxes(subarrays, 1, 2)

    if channels == True:
        return subarrays.reshape(-1, tile_size, tile_size, 4)
    else:
        return subarrays.reshape(-1, tile_size, tile_size)

def random_crop(image, label, crop_size):
    image_height = image.shape[0]
    image_width = image.shape[1]

    random_x = random.randint(0, (image_width - crop_size))
    random_y = random.randint(0, (image_height - crop_size))

    image = image[random_y:random_y + crop_size, random_x:random_x + crop_size]
    label = label[random_y:random_y + crop_size, random_x:random_x + crop_size]
    # print(np.min(image))
    return image, label

def main():
    # file = glob.glob(os.path.join(args.input_dir, '*.tif'))
    # label = glob.glob(os.path.join(args.label_dir, 'label.tif'))

    filepath = "../rgb_nir/rgb_nir.tif"
    labelpath = "../label.tif"
    image = np.array(tiff.imread(filepath))
    labels = np.array(tiff.imread(labelpath))
    # print(np.m(image))

    # number of values with value 255 in labels
    # print(np.shape(np.where(labels == 0)))
    X = divide_images_into_tiles(image, 512, True)
    y = divide_images_into_tiles(labels, 512, False)

    print("Making random crops")
    random_X = []
    random_y = []
    for i in tqdm(range(len(X))):
        for j in range(10):
            X_, y_ = random_crop(X[i], y[i], 256)
            random_X.append(X_)
            random_y.append(y_)

    random_X = np.array(random_X)
    random_y = np.array(random_y)

    shuffled_indices = np.arange(random_X.shape[0])
    np.random.shuffle(shuffled_indices)
    print(shuffled_indices)
    # random_X = random_X[shuffled_indices]
    # random_y = random_y[shuffled_indices]

    train_size = int(len(random_X) * 0.8)
    # X_train = random_X[:train_size]
    # y_train = random_y[:train_size]
    # X_test = random_X[train_size:]
    # y_test = random_y[train_size:]
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    print("Saving npy files")
    for i in tqdm(range(len(train_indices))):
        preprocess_and_save_sample(random_X[train_indices[i]], random_y[train_indices[i]], "./processed_data/train", "train", i)

    for i in tqdm(range(len(test_indices))):
        preprocess_and_save_sample(random_X[test_indices[i]], random_y[test_indices[i]], "./processed_data/test", "test", i)
        



if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_dir', type=str, default='E:/global_water_dataset',
#                         help='path to the directory where the images will be read from')
#     parser.add_argument('--label_dir', type=str, default='E:/global_water_dataset',)
#     parser.add_argument('--output_dir', type=str, default='E:/npy_files',
#                         help='path to the directory where the .npy files will be saved to')
#     parser.add_argument('--num_test_images', type=float, default=100,
#                         help='number of images in the test set')
#     parser.add_argument('--seed', type=int, default=42,
#                         help='random seed for repeatable train/test splits')
#     args = parser.parse_args()
    main()