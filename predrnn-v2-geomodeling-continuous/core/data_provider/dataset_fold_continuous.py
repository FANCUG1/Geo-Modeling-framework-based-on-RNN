import os
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import cv2
import imageio
import torch
import torch.nn.functional as F
from PIL import Image, ImageSequence


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


# path是存放gif的路径，该函数是把立方体以gif图的形式存储在文件夹中
def Create_3D_Model_Dataset(path):
    original_TI = read_image('./fold_continuous.txt')  # 读取原始的训练图像，这里不能归一化，否则gif无法显式，应在load_data里面进行归一化
    z, y, x = original_TI.shape

    # 用于存储64*64*64的立方体，然后再基于这些立方体，做一个64*64*16的training sequence，这个training sequence可以看成是[1,1,16,64,64]的tensor
    data_cube = []

    for i in range(0, z, 5):
        for j in range(0, y, 5):
            for k in range(0, x, 5):
                if (i + 64) < z and (j + 64) < y and (k + 64) < x:
                    temp = original_TI[i:i + 64, j:j + 64, k:k + 64]
                    data_cube.append(temp)

    frame = 32  # 训练集一个16帧
    count = 0  # 用于计数，对gif的图片进行命名

    for cube in data_cube:
        # cube = cube * 255  # [0, 1] -> [0, 255]
        total_frame = cube.shape[0]  # 以x-y方向作为图片，z轴的方向代表视频的帧，为64

        for num in range(0, total_frame, frame):
            temp = cube[num:num + frame, :, :].astype(np.uint8)
            temp_frame = temp.shape[0]  # 32

            gif_images = []
            for s in range(temp_frame):
                gif_images.append(temp[s, :, :])

            # 对生成的images进行保存
            imageio.mimsave('%s/%d.gif' % (path, count), gif_images, fps=1)
            count += 1

    # count = 0
    # for cube in data_cube:
    #     cube = cube * 255.
    #     cube = cube.astype(np.uint8)
    #     total_frame = cube.shape[0]
    #
    #     gif_images = []
    #     for s in range(total_frame):
    #         gif_images.append(cube[s, :, :])
    #
    #     imageio.mimsave('%s/%d.gif' % (path, count), gif_images, fps=1)
    #     count += 1


def gif2numpy(path):
    frames = imageio.mimread(path)
    cv_img = []

    for num, f in enumerate(frames, 0):
        if num == 0:
            cv_img.append(np.array(f))
        else:
            temp = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2GRAY)
            cv_img.append(temp)

    # 将list类型的cv_img转换成numpy类型
    cv_img = np.array(cv_img)
    z, y, x = cv_img.shape
    return cv_img.reshape((z, -1, y, x))  # [frame,channel,height,weight]


# 先定义一个gif转numpy类型的函数，然后再用dataloader来制作训练集和测试集
# 用1帧来预测后面的63帧，64帧合并在一块就是一个子模型
def data_provider(train_batch_size, test_batch_size, path):
    '''
    train_batch_size: 训练集的batchsize
    test_batch_size: 测试集的batchsize
    path: 存放gif图片的路径
    frame: 预测的帧数，这里在训练过程中，默认用32帧来预测32帧，但测试的话，是用1帧来预测后面的63帧甚至更多

    Returns: 返回训练集和测试集的dataloader

    '''
    total_number = len(os.listdir(path))  # 获取训练数据集的总数，5184

    train_dataset_input = []  # 训练数据集所放入的视频帧数，为[train_frame,64,64]
    test_dataset_input = []  # 测试数据集所放入的视频帧数，为[test_frame,64,64]

    for num in range(total_number):
        # 遍历文件夹中的每一个gif图片
        temp_data = gif2numpy('%s/%d.gif' % (path, num))  # 读取每一张gif图片，单通道灰度图,区间范围为[0,255]
        temp_data = temp_data / 255.0  # 连续属性，[0,255]的范围，需要归一化至[0,1]的区间内，
        # temp_data = (temp_data - 127.5) / 127.5  # 归一化至[-1,1]的区间内

        if (num + 1) <= 10000:
            train_dataset_input.append(temp_data)
        else:
            test_dataset_input.append(temp_data)

    # 将训练集和测试集由list转换成numpy的形式
    train_dataset_input = np.array(train_dataset_input)  # [bs, frame, channel, height, weight]

    test_dataset_input = np.array(test_dataset_input)

    # 将numpy类型转换成tensor类型
    train_dataset_input = torch.from_numpy(train_dataset_input).type(torch.cuda.FloatTensor).permute(0, 1, 3, 4, 2)

    test_dataset_input = torch.from_numpy(test_dataset_input).type(torch.cuda.FloatTensor).permute(0, 1, 3, 4, 2)

    train_batch = TensorDataset(train_dataset_input, train_dataset_input)
    test_batch = TensorDataset(test_dataset_input, test_dataset_input)

    dataloader_train = DataLoader(dataset=train_batch, batch_size=train_batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset=test_batch, batch_size=test_batch_size, shuffle=False)

    return dataloader_train, dataloader_test


if __name__ == '__main__':
    # Create_3D_Model_Dataset(path='./Continuous_3D_Model')
    # train_dataloader, test_dataloader = data_provider(train_batch_size=1, test_batch_size=1, path='./Categorical_3D_Model')

    # test_data, _ = test_dataloader.dataset[0]
    # print(test_data.shape)

    # for (i, data) in enumerate(train_dataloader, 0):
    #     print(data[0].shape)
    #     print(data[1].shape)
    #     ims = data[0]
    #
    #     break

    # train_input_handle, test_input_handle = data_provider(train_batch_size=8, test_batch_size=1, path='./Continuous_3D_Model')
    # itr = 1
    # for epoch in range(1, 55):
    #     for i, (ims, _) in enumerate(train_input_handle, 1):
    #         itr += 1
    #
    # print(itr)

    sample = gif2numpy(path='./Continuous_3D_Model/9756.gif')  # [32,1,64,64]
    # sample = sample / 255.
    sample = sample.reshape((-1, 1))

    file = open('./data_sample/sample_9756.txt', 'w')
    file.write(str(64) + ' ' + str(64) + ' ' + str(32) + '\n')
    file.write(str(1) + '\n')
    file.write('facies' + '\n')

    for i in range(len(sample)):
        file.write(str(int(sample[i])) + '\n')

    file.close()
