import os
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import cv2
import imageio
import torch


def read_image(path):
    f = open(path, 'r')
    data = f.readlines()
    data_size = data[0].replace('\n', '').split(' ')
    data_size = list(map(eval, data_size))
    x = []

    for i in range(3, len(data)):
        x.append(int(data[i]))

    x = np.array(x)
    x = x.reshape((data_size[2], data_size[1], data_size[0])).astype(np.uint8)

    return x


def Create_3D_Model_Dataset(path):
    original_TI = read_image('./fold_continuous.txt')
    z, y, x = original_TI.shape

    data_cube = []

    for i in range(0, z, 5):
        for j in range(0, y, 5):
            for k in range(0, x, 5):
                if (i + 64) < z and (j + 64) < y and (k + 64) < x:
                    temp = original_TI[i:i + 64, j:j + 64, k:k + 64]
                    data_cube.append(temp)

    frame = 32
    count = 0

    for cube in data_cube:
        total_frame = cube.shape[0]

        for num in range(0, total_frame, frame):
            temp = cube[num:num + frame, :, :].astype(np.uint8)
            temp_frame = temp.shape[0]

            gif_images = []
            for s in range(temp_frame):
                gif_images.append(temp[s, :, :])

            imageio.mimsave('%s/%d.gif' % (path, count), gif_images, fps=1)
            count += 1


def gif2numpy(path):
    frames = imageio.mimread(path)
    cv_img = []

    for num, f in enumerate(frames, 0):
        # if num == 0:
        #     cv_img.append(np.array(f))
        # else:
        #     temp = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2GRAY)
        #     cv_img.append(temp)
        temp = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2GRAY)
        cv_img.append(temp)

    cv_img = np.array(cv_img)
    z, y, x = cv_img.shape
    return cv_img.reshape((z, -1, y, x))


def data_provider(train_batch_size, test_batch_size, path):
    total_number = len(os.listdir(path))

    train_dataset_input = []
    test_dataset_input = []

    for num in range(total_number):
        temp_data = gif2numpy('%s/%d.gif' % (path, num))
        temp_data = temp_data / 255.0

        if (num + 1) <= 10000:
            train_dataset_input.append(temp_data)
        else:
            test_dataset_input.append(temp_data)

    train_dataset_input = np.array(train_dataset_input)

    test_dataset_input = np.array(test_dataset_input)

    train_dataset_input = torch.from_numpy(train_dataset_input).type(torch.cuda.FloatTensor).permute(0, 1, 3, 4, 2)

    test_dataset_input = torch.from_numpy(test_dataset_input).type(torch.cuda.FloatTensor).permute(0, 1, 3, 4, 2)

    train_batch = TensorDataset(train_dataset_input, train_dataset_input)
    test_batch = TensorDataset(test_dataset_input, test_dataset_input)

    dataloader_train = DataLoader(dataset=train_batch, batch_size=train_batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset=test_batch, batch_size=test_batch_size, shuffle=False)

    return dataloader_train, dataloader_test
