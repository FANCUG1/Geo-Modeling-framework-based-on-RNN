import os.path
import datetime
import cv2
import numpy as np
from core.utils import preprocess
import torch
import torchvision.transforms as T


def train(model, ims, real_input_flag, configs, itr):
    cost = model.train(ims, real_input_flag)

    if configs.reverse_input:
        ims_ = ims.detach().cpu().numpy()
        ims_rev = np.flip(ims_, axis=1).copy()
        ims_rev = torch.from_numpy(ims_rev).type(torch.cuda.FloatTensor)

        cost += model.train(ims_rev, real_input_flag)
        cost = cost / 2

    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print('training loss: ' + str(cost))


def test(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')

    res_path = os.path.join(configs.gen_frm_dir, str(itr))

    try:
        os.makedirs(res_path)
    except OSError:
        pass

    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = np.zeros(
        (1, configs.total_length - mask_input - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0

    real_input_flag = torch.from_numpy(real_input_flag).type(torch.cuda.FloatTensor)
    test_dat = preprocess.reshape_patch(test_input_handle, configs.patch_size)

    img_gen = model.test(test_dat, real_input_flag)
    con_frame = test_dat[:, 0:1, :, :, :]
    img_out = torch.concat([con_frame, img_gen], dim=1)

    img_out = preprocess.reshape_patch_back(img_out, configs.patch_size)

    save_tensor_as_gif(tensor=img_out, path=res_path)
    save_tensor_as_txt(tensor=img_out, path=res_path)


def save_tensor_as_gif(tensor, path):
    tensor = tensor.permute(0, 1, 4, 2, 3)
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)

    bs = tensor.size(0)
    for k in range(bs):
        temp_images = map(T.ToPILImage(), tensor[k].unbind(dim=0))
        first_img, *rest_imgs = temp_images
        first_img.save(path + '/%d.gif' % (k + 1), save_all=True, append_images=rest_imgs, duration=120, loop=0, optimize=True)


def save_tensor_as_txt(tensor, path):
    tensor = tensor.permute(0, 1, 4, 2, 3)
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)
    tensor = tensor * 255.
    b, f, c, h, w = tensor.size()

    for i in range(b):
        temp = tensor.detach().cpu().numpy()
        temp = temp.reshape((f, h, w))

        file = open('%s/reconstruction_%d.txt' % (path, i + 1), 'w')
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

