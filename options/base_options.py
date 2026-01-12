import argparse
import os
# from util import util
import torch
# import util.util


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # data params
        self.parser.add_argument('--dataroot', default="/home/zhuxunyang/coding/simply/datasets/Bkgm0630" ,help='path to meshes (should have subfolders train, test)')
        self.parser.add_argument('--valroot', default="/home/zhuxunyang/coding/banding_detect/datasets/banding_dataset/test")   #0429
        self.parser.add_argument('--dataset_mode', choices={"classification", "segmentation", "banding"}, default='Banding')
        # self.parser.add_argument('--ninput_edges', type=int, default=750, help='# of input edges (will include dummy edges)')
        self.parser.add_argument('--ninput_edges', type=int, default=3000, help='# of input edges (will include dummy edges)')
        self.parser.add_argument('--max_dataset_size', type=int, default=200, help='Maximum number of samples per epoch')
        # network params
        self.parser.add_argument('--batch_size', type=int, default = 1, help='input batch size')
        # self.parser.add_argument('--arch', type=str, default='mconvnet', help='selects network to use') #todo add choice
        # self.parser.add_argument('--resblocks', type=int, default=0, help='# of res blocks')
        # self.parser.add_argument('--fc_n', type=int, default=4000, help='# between fc and nclasses') #todo make generic
        # self.parser.add_argument('--ncf', nargs='+', default=[32, 64, 128, 64, 32, 1], type=int, help='conv filters')        #[32, 64, 64]
        # self.parser.add_argument('--pool_res', nargs='+', default=[3500, 4000, 4500, 4000, 3500, 3000], type=int, help='pooling res')
        

        self.parser.add_argument('--arch', type=str, default='meshunet', help='selects network to use') #todo add choices
        self.parser.add_argument('--resblocks', type=int, default=3, help='# of res blocks')
        self.parser.add_argument('--fc_n', type=int, default=100, help='# between fc and nclasses') #todo make generic
        self.parser.add_argument('--ncf', nargs='+', default=[32, 64, 128, 256], type=int, help='conv filters')         #[32, 64, 128, 256]   [32, 48, 64, 96]
        self.parser.add_argument('--pool_res', nargs='+', default=[4000, 3500, 3000], type=int, help='pooling res')      #1800, 1350, 600  [3500, 3200, 3000]

        self.parser.add_argument('--norm', type=str, default='batch',help='instance normalization or batch normalization or group normalization')
        self.parser.add_argument('--num_groups', type=int, default=16, help='# of groups for groupnorm')
        self.parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # general params
        self.parser.add_argument('--num_threads', default=3, type=int, help='# threads for loading data')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='debug', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='/home/zhuxunyang/coding/banding_detect/checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes meshes in order, otherwise takes them randomly')
        self.parser.add_argument('--seed', type=int, help='if specified, uses seed')
        # visualization params
        self.parser.add_argument('--export_folder', type=str, default='', help='exports intermediate collapses to this folder')
        #
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args()
        self.opt.is_train = self.is_train   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        # self.opt.device = torch.device('cpu')

        args = vars(self.opt)

        if self.opt.seed is not None:
            import numpy as np
            import random
            torch.manual_seed(self.opt.seed)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)

        # if self.opt.export_folder:
        #     self.opt.export_folder = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.export_folder)
        #     util.mkdir(self.opt.export_folder)

        if self.is_train:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            # util.mkdir(expr_dir)

            # file_name = os.path.join(expr_dir, 'opt.txt')
            # with open(file_name, 'wt') as opt_file:
            #     opt_file.write('------------ Options -------------\n')
            #     for k, v in sorted(args.items()):
            #         opt_file.write('%s: %s\n' % (str(k), str(v)))
            #     opt_file.write('-------------- End ----------------\n')
        return self.opt
