import os
import shutil
import argparse
import numpy as np
import math
from core.data_provider import dataset_checker_categorical
from core.models.model_factory import Model
from core.utils import preprocess
import core.trainer as trainer
import torch


def reserve_schedule_sampling_exp(itr):
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample((args.batch_size, args.input_length - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample((args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)

    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))

    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - 2):

            if j < args.input_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

            else:
                if true_token[i, j - (args.input_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - 2,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))

    real_input_flag = torch.from_numpy(real_input_flag).type(torch.cuda.FloatTensor)
    return real_input_flag


def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample((args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)

    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    real_input_flag = torch.from_numpy(real_input_flag).type(torch.cuda.FloatTensor)
    return eta, real_input_flag


def train_wrapper(model):
    if args.pretrained_model:
        model.load(args.pretrained_model)
    # load data
    train_input_handle, test_input_handle = dataset_checker_categorical.data_provider(train_batch_size=args.batch_size, test_batch_size=1,
                                                                                   path='./core/data_provider/Checker_3D_Model')

    eta = args.sampling_start_value
    itr = 1

    for epoch in range(1, args.max_iterations + 1):
        print('The %d training epoch is begin' % epoch)
        for i, (ims, _) in enumerate(train_input_handle, 1):
            del _
            ims = preprocess.reshape_patch(ims, args.patch_size)

            if args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(itr)
            else:
                eta, real_input_flag = schedule_sampling(eta, itr)

            trainer.train(model, ims, real_input_flag, args, itr)

            itr += 1

        model.save(epoch)

        test_data, _ = test_input_handle.dataset[0]
        del _
        test_data = test_data.unsqueeze(dim=0).to(args.device)
        trainer.test(model, test_data, args, epoch)


if __name__ == '__main__':

    # -----------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

    # training/test
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')

    # data
    parser.add_argument('--dataset_name', type=str, default='mnist')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/predrnn-v2')
    parser.add_argument('--gen_frm_dir', type=str, default='./results/checker_categorical_predrnn_v2')
    parser.add_argument('--input_length', type=int, default=2)
    parser.add_argument('--total_length', type=int, default=32)
    parser.add_argument('--img_width', type=int, default=64)
    parser.add_argument('--img_channel', type=int, default=1)

    # model
    parser.add_argument('--model_name', type=str, default='predrnn_v2')
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
    parser.add_argument('--filter_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--layer_norm', type=int, default=1)
    parser.add_argument('--decouple_beta', type=float, default=0.1)

    # reverse scheduled sampling
    parser.add_argument('--reverse_scheduled_sampling', type=int, default=1)
    parser.add_argument('--r_sampling_step_1', type=float, default=25000)
    parser.add_argument('--r_sampling_step_2', type=int, default=50000)
    parser.add_argument('--r_exp_alpha', type=int, default=5000)
    # scheduled sampling
    parser.add_argument('--scheduled_sampling', type=int, default=1)
    parser.add_argument('--sampling_start_value', type=float, default=1.0)

    # optimization
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--reverse_input', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_iterations', type=int, default=80)
    parser.add_argument('--display_interval', type=int, default=100)
    parser.add_argument('--n_gpu', type=int, default=1)

    # visualization of memory decoupling
    parser.add_argument('--visual', type=int, default=0)  # 0
    parser.add_argument('--visual_path', type=str, default='./decoupling_visual')

    args = parser.parse_args()

    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    if os.path.exists(args.gen_frm_dir):
        shutil.rmtree(args.gen_frm_dir)
    os.makedirs(args.gen_frm_dir)

    print('Initializing models')

    model = Model(args)

    if args.is_training:
        train_wrapper(model)
