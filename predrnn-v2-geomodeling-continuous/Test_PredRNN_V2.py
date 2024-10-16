import torch
import numpy as np
import argparse
from core.models import predrnn_v2
from core.utils import preprocess
from torch.utils.data import DataLoader, TensorDataset
import os


def read_image(path):
    f = open(path, 'r')
    data = f.readlines()
    data_size = data[0].replace('\n', '').split(' ')
    data_size = list(map(eval, data_size))
    x = []

    for i in range(3, len(data)):
        x.append(int(data[i]))

    x = np.array(x)
    x = x.reshape((data_size[2], data_size[1], data_size[0]))

    return x


def test_data_provider(path, configs):
    original_TI = read_image(path)
    z, y, x = original_TI.shape

    data_cube = []

    for i in range(0, z, configs.seg_step):
        for j in range(0, y, configs.seg_step):
            for k in range(0, x, configs.seg_step):
                if (i + configs.img_width) < z and (j + configs.img_width) < y and (k + configs.img_width) < x:
                    temp = original_TI[i: i + configs.img_width, j: j + configs.img_width, k: k + configs.img_width]
                    temp = temp / 255.
                    data_cube.append(temp)

    test_data_array = np.array(data_cube).reshape((len(data_cube), configs.img_width, -1, configs.img_width, configs.img_width))
    test_dataset_tensor = torch.from_numpy(test_data_array).type(torch.cuda.FloatTensor).permute(0, 1, 3, 4, 2)

    test_dataset = TensorDataset(test_dataset_tensor, test_dataset_tensor)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)
    return test_dataloader


def save_real_tensor_as_txt(tensor, path, epoch):
    tensor = preprocess.reshape_patch_back(patch_tensor=tensor, patch_size=args.patch_size)
    tensor = tensor.permute(0, 1, 4, 2, 3)
    tensor = tensor * 255.
    tensor = tensor.clamp(0, 255)

    b, f, c, h, w = tensor.size()

    temp = tensor.detach().cpu().numpy()
    temp = temp.reshape((f, h, w))

    file = open('%s/real-%d.txt' % (path, epoch), 'w')
    file.write(str(w) + ' ' + str(h) + ' ' + str(f) + '\n')
    file.write(str(1) + '\n')
    file.write('facies' + '\n')

    temp = temp.reshape((f * h * w, 1))

    for j in range(len(temp)):
        if abs(temp[j] - int(temp[j])) < 0.5:
            value = int(temp[j])
            file.write(str(value) + '\n')
        else:
            value = int(temp[j]) + 1
            file.write(str(value) + '\n')

    file.close()


def save_generative_tensor_as_txt(tensor, path, epoch):
    tensor = preprocess.reshape_patch_back(patch_tensor=tensor, patch_size=args.patch_size)
    tensor = tensor.permute(0, 1, 4, 2, 3)
    tensor = tensor * 255.
    tensor = tensor.clamp(0, 255)

    b, f, c, h, w = tensor.size()

    for i in range(b):
        temp = tensor.detach().cpu().numpy()
        temp = temp.reshape((f, h, w))

        file = open('%s/reconstruction_%d.txt' % (path, epoch), 'w')
        file.write(str(w) + ' ' + str(h) + ' ' + str(f) + '\n')
        file.write(str(1) + '\n')
        file.write('facies' + '\n')

        temp = temp.reshape((f * h * w, 1))

        for j in range(len(temp)):
            if abs(temp[j] - int(temp[j])) < 0.5:
                value = int(temp[j])
                file.write(str(value) + '\n')
            else:
                value = int(temp[j]) + 1
                file.write(str(value) + '\n')

        file.close()


if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

    # training/test
    parser.add_argument('--device', type=str, default='cuda:0')

    # data
    parser.add_argument('--input_length', type=int, default=2)
    parser.add_argument('--total_length', type=int, default=64)
    parser.add_argument('--img_width', type=int, default=64)
    parser.add_argument('--img_channel', type=int, default=1)
    parser.add_argument('--seg_step', type=int, default=20)

    # model
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_hidden', type=list, default=[64, 64, 64, 64])
    parser.add_argument('--filter_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--layer_norm', type=int, default=1)
    parser.add_argument('--decouple_beta', type=float, default=0.1)
    parser.add_argument('--saved_model', default='./checkpoints/predrnn-v2/model-80.pth')

    # reverse scheduled sampling
    parser.add_argument('--reverse_scheduled_sampling', type=int, default=1)

    # visualization of memory decoupling
    parser.add_argument('--visual', type=int, default=0)
    parser.add_argument('--visual_path', type=str, default='./decoupling_visual')

    args = parser.parse_args()

    test_data = test_data_provider(path='./core/data_provider/fold_continuous.txt', configs=args)

    model = predrnn_v2.RNN(num_layers=args.num_layers, num_hidden=args.num_hidden, configs=args).cuda()
    stats = torch.load(args.saved_model)
    model.load_state_dict(stats['net_param'])

    for i, (data, _) in enumerate(test_data, 1):
        del _
        bs = data.shape[0]
        data_reshape = preprocess.reshape_patch(img_tensor=data, patch_size=args.patch_size)
        con_frame = data_reshape[:, 0:1, :, :, :]

        mask_true = torch.zeros(bs, args.total_length - 2, args.img_width // args.patch_size,
                                args.img_width // args.patch_size, args.img_width // args.patch_size).cuda()

        mask_true[:, :args.input_length - 1, :, :, :] = 1.0

        next_frames, _ = model.forward(frames_tensor=data_reshape, mask_true=mask_true)

        pred_result = torch.concat([con_frame, next_frames], dim=1)

        save_path = './Prediction_results/%d' % (i)
        try:
            os.makedirs('%s' % save_path)
        except OSError:
            pass

        save_real_tensor_as_txt(tensor=data_reshape, path=save_path, epoch=i)
        save_generative_tensor_as_txt(tensor=pred_result, path=save_path, epoch=i)





