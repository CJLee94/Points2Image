"""run_train.py

Main HoVer-Net training script.

Usage:
  run_train.py [--gpu=<id>] [--view=<dset>]
  run_train.py (-h | --help)
  run_train.py --version

Options:
  -h --help       Show this string.
  --version       Show version.
  --gpu=<id>      Comma separated GPU list. [default: 0,1,2,3]
  --view=<dset>   Visualise images after augmentation. Choose 'train' or 'valid'.
"""

import cv2

cv2.setNumThreads(0)
import argparse
import glob
# import importlib
import inspect
import json
import os
# import shutil

import matplotlib
import numpy as np
import torch
# from docopt import docopt
from tensorboardX import SummaryWriter
from torch.nn import DataParallel  # TODO: switch to DistributedDataParallel
from torch.utils.data import DataLoader

from config import Config
from dataloader.train_loader import FileLoader, H5FileLoader
from misc.utils import rm_n_mkdir
from run_utils.engine import RunEngine
from run_utils.utils import (
    # check_log_dir,
    check_manual_seed,
    colored,
    convert_pytorch_checkpoint,
)


#### have to move outside because of spawn
# * must initialize augmentor per worker, else duplicated rng generators may happen
def worker_init_fn(worker_id):
    # ! to make the seed chain reproducible, must use the torch random, not numpy
    # the torch rng from main thread will regenerate a base seed, which is then
    # copied into the dataloader each time it created (i.e start of each epoch)
    # then dataloader with this seed will spawn worker, now we reseed the worker
    worker_info = torch.utils.data.get_worker_info()
    # to make it more random, simply switch torch.randint to np.randint
    worker_seed = torch.randint(0, 2 ** 32, (1,))[0].cpu().item() + worker_id
    # print('Loader Worker %d Uses RNG Seed: %d' % (worker_id, worker_seed))
    # retrieve the dataset copied into this worker process
    # then set the random seed for each augmentation
    worker_info.dataset.setup_augmentor(worker_id, worker_seed)
    return

# def custom_collate(data): #(2)
#     imgs = torch.cat([torch.tensor(d['img'])[None] for d in data], 0) #(3)
#     np_maps = torch.cat([torch.tensor(d['np_map'])[None] for d in data], 0)
#     hv_maps = torch.cat([torch.tensor(d['hv_map'])[None] for d in data], 0)
#     augs = [d['augmenter'] for d in data]
#     if 'tp_map' in data[0].keys():
#         tp_maps = torch.cat([torch.tensor(d['tp_map']) for d in data], 0)
#         return { #(6)
#             'img': imgs,
#             'np_map': np_maps,
#             'hv_map': hv_maps,
#             'augmenter': augs,
#             'tp_map': tp_maps
#         }
#     else:
#         return { #(6)
#             'img': imgs,
#             'np_map': np_maps,
#             'hv_map': hv_maps,
#             'augmenter': augs
#         }

# def worker_init_fn_generator(worker_id):
#     # ! to make the seed chain reproducible, must use the torch random, not numpy
#     # the torch rng from main thread will regenerate a base seed, which is then
#     # copied into the dataloader each time it created (i.e start of each epoch)
#     # then dataloader with this seed will spawn worker, now we reseed the worker
#     worker_info = torch.utils.data.get_worker_info()
#     # to make it more random, simply switch torch.randint to np.randint
#     worker_seed = torch.randint(0, 2 ** 32, (1,))[0].cpu().item() + worker_id
#     # print('Loader Worker %d Uses RNG Seed: %d' % (worker_id, worker_seed))
#     # retrieve the dataset copied into this worker process
#     # then set the random seed for each augmentation
#     worker_info.dataset.set_worker_id_seed(worker_id, worker_seed)
#     return


####
class TrainManager(Config):
    """Either used to view the dataset or to initialise the main training loop."""

    def __init__(self):
        super().__init__()
        return

    ####
    def view_dataset(self, mode="train"):
        """
        Manually change to plt.savefig or plt.show 
        if using on headless machine or not
        """
        self.nr_gpus = 1
        import matplotlib.pyplot as plt
        check_manual_seed(self.seed)
        # TODO: what if each phase want diff annotation ?
        phase_list = self.model_config["phase_list"][0]
        target_info = phase_list["target_info"]
        prep_func, prep_kwargs = target_info["viz"]
        dataloader = self._get_datagen(2, mode, target_info["gen"])
        for batch_id, batch_data in enumerate(dataloader):  
            # convert from Tensor to Numpy
            batch_data = {k: v.numpy() for k, v in batch_data.items()}
            viz = prep_func(batch_data, is_batch=True, **prep_kwargs)
            plt.imshow(viz)
            plt.savefig('valid_{}.png'.format(batch_id))
            plt.close()
        self.nr_gpus = -1
        return

    ####
    def _get_datagen(self, batch_size, run_mode, target_gen, nr_procs=0, fold_idx=0):
        nr_procs = nr_procs if not self.debug else 0
        # ! Hard assumption on file type
        file_list = []
        if run_mode == "train":
            data_dir_list = self.train_dir_list
        else:
            data_dir_list = self.valid_dir_list
        for dir_path in data_dir_list:
            if dir_path[-3:] == '.h5':
                h5dataset = True
                file_list.append(dir_path)
            else:
                h5dataset = False
                file_list.extend(glob.glob("%s/*.npy" % dir_path))
        file_list.sort()  # to always ensure same input ordering

        assert len(file_list) > 0, (
            "No .npy found for `%s`, please check `%s` in `config.py`"
            % (run_mode, "%s_dir_list" % run_mode)
        )
        print("Dataset %s: %d" % (run_mode, len(file_list)))

        if h5dataset:
            print('Using H5File Loader')
            input_dataset = H5FileLoader(
                file_list,
                mode=run_mode,
                with_type=self.type_classification,
                setup_augmentor=nr_procs == 0,
                target_gen=target_gen,
                **self.shape_info[run_mode]
            )
        else:
            input_dataset = FileLoader(
                file_list,
                mode=run_mode,
                with_type=self.type_classification,
                setup_augmentor=nr_procs == 0,
                target_gen=target_gen,
                **self.shape_info[run_mode]
            )

        dataloader = DataLoader(
            input_dataset,
            num_workers=nr_procs,
            batch_size=batch_size * self.nr_gpus,
            shuffle=run_mode == "train",
            drop_last=run_mode == "train",
            worker_init_fn=worker_init_fn,
        )
        return dataloader

    ####
    def run_once(self, opt, run_engine_opt, log_dir, prev_log_dir=None, fold_idx=0, phase_id=None):
        """Simply run the defined run_step of the related method once."""
        check_manual_seed(self.seed)

        log_info = {}
        if self.logging:
            # check_log_dir(log_dir)
            rm_n_mkdir(log_dir)

            tfwriter = SummaryWriter(log_dir=log_dir)
            json_log_file = log_dir + "/stats.json"
            with open(json_log_file, "w") as json_file:
                json.dump({}, json_file)  # create empty file
            log_info = {
                "json_file": json_log_file,
                "tfwriter": tfwriter,
            }

        ####
        loader_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            loader_dict[runner_name] = self._get_datagen(
                opt["batch_size"][runner_name],
                runner_name,
                opt["target_info"]["gen"],
                nr_procs=runner_opt["nr_procs"],
                fold_idx=fold_idx,
            )
        ####
        def get_last_chkpt_path(prev_phase_dir, net_name):
            stat_file_path = prev_phase_dir + "/stats.json"
            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)
            epoch_list = [int(v) for v in info.keys()]
            last_chkpts_path = "%s/best.tar" % (
                prev_phase_dir,
            )
            return last_chkpts_path

        # TODO: adding way to load pretrained weight or resume the training
        # parsing the network and optimizer information
        net_run_info = {}
        net_info_opt = opt["run_info"]
        for net_name, net_info in net_info_opt.items():
            assert inspect.isclass(net_info["desc"]) or inspect.isfunction(
                net_info["desc"]
            ), "`desc` must be a Class or Function which instantiate NEW objects !!!"
            net_desc = net_info["desc"]()

            # TODO: customize print-out for each run ?
            # summary_string(net_desc, (3, 270, 270), device='cpu')

            pretrained_path = net_info["pretrained"]
            if pretrained_path is not None:
                if pretrained_path == -1:
                    # * depend on logging format so may be broken if logging format has been changed
                    pretrained_path = get_last_chkpt_path(prev_log_dir, net_name)
                    net_state_dict = torch.load(pretrained_path)["desc"]
                else:
                    chkpt_ext = os.path.basename(pretrained_path).split(".")[-1]
                    if chkpt_ext == "npz":
                        net_state_dict = dict(np.load(pretrained_path))
                        net_state_dict = {
                            k: torch.from_numpy(v) for k, v in net_state_dict.items()
                        }
                    elif chkpt_ext == "tar":  # ! assume same saving format we desire
                        net_state_dict = torch.load(pretrained_path)["desc"]

                colored_word = colored(net_name, color="red", attrs=["bold"])
                print(
                    "Model `%s` pretrained path: %s" % (colored_word, pretrained_path)
                )

                # load_state_dict returns (missing keys, unexpected keys)
                net_state_dict = convert_pytorch_checkpoint(net_state_dict)
                load_feedback = net_desc.load_state_dict(net_state_dict, strict=False)
                # * uncomment for your convenience
                print("Missing Variables: \n", load_feedback[0])
                print("Detected Unknown Variables: \n", load_feedback[1])

            # * extremely slow to pass this on DGX with 1 GPU, why (?)
            net_desc = DataParallel(net_desc)
            net_desc = net_desc.to("cuda")
            # print(net_desc) # * dump network definition or not?
            optimizer, optimizer_args = net_info["optimizer"]
            optimizer = optimizer(net_desc.parameters(), **optimizer_args)
            # TODO: expand for external aug for scheduler
            nr_iter = opt["nr_epochs"] * len(loader_dict["train"])
            scheduler = net_info["lr_scheduler"](optimizer)
            net_run_info[net_name] = {
                "desc": net_desc,
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                # TODO: standardize API for external hooks
                "extra_info": net_info["extra_info"],
            }

        # parsing the running engine configuration
        assert (
            "train" in run_engine_opt
        ), "No engine for training detected in description file"

        # initialize runner and attach callback afterward
        # * all engine shared the same network info declaration
        runner_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            runner_dict[runner_name] = RunEngine(
                dataloader=loader_dict[runner_name],
                engine_name=runner_name,
                run_step=runner_opt["run_step"],
                run_info=net_run_info,
                log_info=log_info,
                phase_id=phase_id,
            )

        for runner_name, runner in runner_dict.items():
            callback_info = run_engine_opt[runner_name]["callbacks"]
            for event, callback_list, in callback_info.items():
                for callback in callback_list:
                    if callback.engine_trigger:
                        triggered_runner_name = callback.triggered_engine_name
                        callback.triggered_engine = runner_dict[triggered_runner_name]
                    runner.add_event_handler(event, callback)

        # retrieve main runner
        main_runner = runner_dict["train"]
        main_runner.state.logging = self.logging
        main_runner.state.log_dir = log_dir
        # start the run loop
        main_runner.run(opt["nr_epochs"])

        print("\n")
        print("########################################################")
        print("########################################################")
        print("\n")
        return

    ####
    def run(self):
        """Define multi-stage run or cross-validation or whatever in here."""
        self.nr_gpus = torch.cuda.device_count()
        print('Detect #GPUS: %d' % self.nr_gpus)

        phase_list = self.model_config["phase_list"]
        engine_opt = self.model_config["run_engine"]

        prev_save_path = None
        for phase_idx, phase_info in enumerate(phase_list):
            if len(phase_list) == 1:
                save_path = self.log_dir
            else:
                save_path = self.log_dir + "/%02d/" % (phase_idx)
            self.run_once(
                phase_info, engine_opt, save_path, prev_log_dir=prev_save_path, phase_id=phase_idx,
            )
            prev_save_path = save_path


####
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', default=None)
    parser.add_argument('--gpu', default='0,1,2,3')
    parser.add_argument('--train_dir', default=None, nargs='+')
    parser.add_argument('--valid_dir', default=None)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--otf', default=None)
    parser.add_argument('--otf_dataroot', default='/home/cj/Research/Points2Image_old/processed_data/MoNuSeg_train_v4_enhanced_pcorrected.h5')
    parser.add_argument('--valid_shape', default=None, type=int)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    args = parser.parse_args()
    # args = docopt(__doc__, version="HoVer-Net v1.0")
    trainer = TrainManager()
    trainer.load_config_from_args(args)

    if args.view:
        if args.view != "train" and args.view != "valid":
            raise Exception('Use "train" or "valid" for --view.')
        trainer.view_dataset(args.view)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        trainer.run()
