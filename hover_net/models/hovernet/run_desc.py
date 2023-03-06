import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from misc.utils import center_pad_to_shape, cropping_center
from .utils import crop_to_shape, dice_loss, mse_loss, msge_loss, xentropy_loss
from generative_models import create_model
from metrics.eval_metrics import compute_metrics
from .post_proc import process

from collections import OrderedDict

####
def train_step(batch_data, run_info, phase_id=None):
    # TODO: synchronize the attach protocol
    run_info, state_info = run_info
    loss_func_dict = {
        "bce": xentropy_loss,
        "dice": dice_loss,
        "mse": mse_loss,
        "msge": msge_loss,
    }
    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {"EMA": {}}
    track_value = lambda name, value: result_dict["EMA"].update({name: value})

    ####
    model = run_info["net"]["desc"]
    optimizer = run_info["net"]["optimizer"]

    ####
    # import pdb
    # pdb.set_trace()
        # imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]

    # HWC
    true_np = true_np.to("cuda").type(torch.int64)
    true_hv = true_hv.to("cuda").type(torch.float32)

    true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
    true_dict = {
        "np": true_np_onehot,
        "hv": true_hv,
    }

    if run_info["net"]['extra_info']["generator"] is None:
        imgs = batch_data["img"]
        imgs = imgs.to("cuda").type(torch.float32)  # to NCHW
        imgs = imgs.permute(0, 3, 1, 2).contiguous()
    else:
        generator = run_info["net"]['extra_info']["generator"]
        augmentor = run_info["net"]['extra_info']["augmentor"]
        with torch.no_grad():
            imgs = generator.netG_A(torch.cat([true_hv.permute(0,3,1,2).contiguous(), true_np[:,None]], 1))
        imgs = torch.clamp(255.0*(imgs+1)/2.0, 0, 255)
        A = augmentor.setup_augmentor(batch_data["worker_id"][0].item())
        input_augs = A['input_augmenter']
            
        for sample_id, img in enumerate(imgs):
            input_aug_d = input_augs.to_deterministic()
            img = img.detach().cpu().numpy().transpose(1,2,0).astype(np.uint8)
            img_auged = input_aug_d.augment_image(img)
            imgs[sample_id] = torch.from_numpy(img_auged).permute(2,0,1).contiguous()

        
    # else:
    #     generator = run_info["net"]['extra_info']["generator"]
    #     augmentor = run_info["net"]['extra_info']["augmentor"]
    #     augmentor.setup_augmentor(batch_data['worker_id'].numpy(), batch_data['worker_seed'].numpy())

    #     generator.set_input(batch_data)
    #     with torch.no_grad():
    #         input_image = generator.netG_A(generator.real_A)  # G_A(A)
    #         target_mask = generator.real_A
    #     imgs = torch.clamp(255.0*(input_image+1)/2.0, 0, 255)

    #     true_hv = target_mask[:,:2].permute(0,2,3,1).contiguous()
    #     true_np = target_mask[:, -1].type(torch.int64)

    #     for sample_idx, img in enumerate(imgs):
    #         input_augs = augmentor.input_augs.to_deterministic()
    #         img = img.cpu().numpy().transpose(1,2,0).astype(np.uint8)

    #         imgs[sample_idx] = torch.from_numpy(input_augs.augment_image(img).copy()).permute(2,0,1)


    #     true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)

    #     true_dict = {
    #         "np": true_np_onehot,
    #         "hv": true_hv,
    #     }
    # import pdb
    # pdb.set_trace()    

    if model.module.nr_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
        true_tp_onehot = F.one_hot(true_tp, num_classes=model.module.nr_types)
        true_tp_onehot = true_tp_onehot.type(torch.float32)
        true_dict["tp"] = true_tp_onehot

    ####
    model.train()
    model.zero_grad()  # not rnn so not accumulate
    
    pred_dict = model(imgs)
    pred_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
    )
    pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
    if model.module.nr_types is not None:
        pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)

    ####
    loss = 0
    loss_opts = run_info["net"]["extra_info"]["loss"]
    for branch_name in pred_dict.keys():
        for loss_name, loss_weight in loss_opts[branch_name].items():
            loss_func = loss_func_dict[loss_name]
            loss_args = [true_dict[branch_name], pred_dict[branch_name]]
            if loss_name == "msge":
                loss_args.append(true_np_onehot[..., 1])
            # import pdb
            # pdb.set_trace()
            term_loss = loss_func(*loss_args)
            track_value("loss_%s_%s" % (branch_name, loss_name), term_loss.cpu().item())
            loss += loss_weight * term_loss

    track_value("overall_loss", loss.cpu().item())
    # * gradient update

    # torch.set_printoptions(precision=10)
    loss.backward()
    optimizer.step()
    ####

    # pick 2 random sample from the batch for visualization
    sample_indices = torch.randint(0, true_np.shape[0], (2,))

    imgs = (imgs[sample_indices]).byte()  # to uint8
    imgs = imgs.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    pred_dict["np"] = pred_dict["np"][..., 1]  # return pos only
    pred_dict = {
        k: v[sample_indices].detach().cpu().numpy() for k, v in pred_dict.items()
    }

    true_dict["np"] = true_np
    true_dict = {
        k: v[sample_indices].detach().cpu().numpy() for k, v in true_dict.items()
    }

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict["raw"] = {  # protocol for contents exchange within `raw`
        "img": imgs,
        "np": (true_dict["np"], pred_dict["np"]),
        "hv": (true_dict["hv"], pred_dict["hv"]),
        "phase_id": phase_id
    }
    return result_dict


####
def valid_step(batch_data, run_info, phase_id=None):
    run_info, state_info = run_info
    ####
    model = run_info["net"]["desc"]
    model.eval()  # infer mode

    ####
    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]

    imgs_gpu = imgs.to("cuda").type(torch.float32)  # to NCHW
    imgs_gpu = imgs_gpu.permute(0, 3, 1, 2).contiguous()

    # HWC
    # import pdb
    # pdb.set_trace()
    true_np = true_np.type(torch.int64)
    true_hv = true_hv.type(torch.float32)
    # true_np = torch.squeeze(true_np).type(torch.int64)
    # true_hv = torch.squeeze(true_hv).type(torch.float32)

    true_dict = {
        "np": true_np,
        "hv": true_hv,
    }

    if model.module.nr_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = torch.squeeze(true_tp).type(torch.int64)
        true_dict["tp"] = true_tp

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        pred_dict = model(imgs_gpu)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1]
        if model.module.nr_types is not None:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=False)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict = {  # protocol for contents exchange within `raw`
        "raw": {
            "imgs": imgs.numpy(),
            "true_np": true_dict["np"].numpy(),
            "true_hv": true_dict["hv"].numpy(),
            "true_inst": batch_data["inst_map"].numpy(),
            "prob_np": pred_dict["np"].cpu().numpy(),
            "pred_hv": pred_dict["hv"].cpu().numpy(),
            "phase_id": phase_id,
        }
    }
    if model.module.nr_types is not None:
        result_dict["raw"]["true_tp"] = true_dict["tp"].numpy()
        result_dict["raw"]["pred_tp"] = pred_dict["tp"].cpu().numpy()
    return result_dict


####
def infer_step(batch_data, model):

    ####
    patch_imgs = batch_data

    patch_imgs_gpu = patch_imgs.to("cuda").type(torch.float32)  # to NCHW
    patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

    ####
    model.eval()  # infer mode

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        pred_dict = model(patch_imgs_gpu)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
        if "tp" in pred_dict:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map
        pred_output = torch.cat(list(pred_dict.values()), -1)

    # * Its up to user to define the protocol to process the raw output per step!
    return pred_output.cpu().numpy()


####
def viz_step_output(raw_data, nr_types=None):
    """
    `raw_data` will be implicitly provided in the similar format as the 
    return dict from train/valid step, but may have been accumulated across N running step
    """

    imgs = raw_data["img"]
    true_np, pred_np = raw_data["np"]
    true_hv, pred_hv = raw_data["hv"]
    if nr_types is not None:
        true_tp, pred_tp = raw_data["tp"]

    aligned_shape = [list(imgs.shape), list(true_np.shape), list(pred_np.shape)]
    aligned_shape = np.min(np.array(aligned_shape), axis=0)[1:3]

    cmap = plt.get_cmap("jet")

    def colorize(ch, vmin, vmax):
        """
        Will clamp value value outside the provided range to vmax and vmin
        """
        ch = np.squeeze(ch.astype("float32"))
        ch[ch > vmax] = vmax  # clamp value
        ch[ch < vmin] = vmin
        ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
        # take RGB from RGBA heat map
        ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
        # ch_cmap = center_pad_to_shape(ch_cmap, aligned_shape)
        return ch_cmap

    viz_list = []
    for idx in range(imgs.shape[0]):
        # img = center_pad_to_shape(imgs[idx], aligned_shape)
        img = cropping_center(imgs[idx], aligned_shape)

        true_viz_list = [img]
        # cmap may randomly fails if of other types
        true_viz_list.append(colorize(true_np[idx], 0, 1))
        true_viz_list.append(colorize(true_hv[idx][..., 0], -1, 1))
        true_viz_list.append(colorize(true_hv[idx][..., 1], -1, 1))
        if nr_types is not None:  # TODO: a way to pass through external info
            true_viz_list.append(colorize(true_tp[idx], 0, nr_types))
        true_viz_list = np.concatenate(true_viz_list, axis=1)

        pred_viz_list = [img]
        # cmap may randomly fails if of other types
        pred_viz_list.append(colorize(pred_np[idx], 0, 1))
        pred_viz_list.append(colorize(pred_hv[idx][..., 0], -1, 1))
        pred_viz_list.append(colorize(pred_hv[idx][..., 1], -1, 1))
        if nr_types is not None:
            pred_viz_list.append(colorize(pred_tp[idx], 0, nr_types))
        pred_viz_list = np.concatenate(pred_viz_list, axis=1)

        viz_list.append(np.concatenate([true_viz_list, pred_viz_list], axis=0))
    viz_list = np.concatenate(viz_list, axis=0)
    return viz_list


####
from itertools import chain


def proc_valid_step_output(raw_data, nr_types=None):
    # TODO: add auto populate from main state track list
    track_dict = {"scalar": {}, "image": {}}

    def track_value(name, value, vtype):
        return track_dict[vtype].update({name: value})

    def _dice_info(true, pred, label):
        true = np.array(true == label, np.int32)
        pred = np.array(pred == label, np.int32)
        inter = (pred * true).sum()
        total = (pred + true).sum()
        return inter, total

    prob_np = raw_data["prob_np"]
    # * HV regression statistic
    pred_hv = raw_data["pred_hv"]
    true_inst = raw_data["true_inst"]
    true_np = raw_data["true_np"]
    true_hv = raw_data["true_hv"]

    if raw_data["phase_id"] == 1:
        metrics_name = ['aji']
        # metrics_name = ['p_iou', 'p_F1', 'dice', 'aji']
        # p_F1_sum = []
        # p_iou_sum = []
        # dice_sum = []
        aji_sum = []
        for idx in range(len(raw_data["true_np"])):
            patch_prob_np = prob_np[idx]
            patch_true_inst = true_inst[idx]
            patch_pred_hv = pred_hv[idx]

            pred_inst = np.concatenate([patch_prob_np[...,None],
                                        patch_pred_hv], -1)
            pred_inst, _ = process(pred_inst)

            single_results = compute_metrics(pred_inst, patch_true_inst, metrics_name)
            # p_iou_sum.append(single_results['p_iou'])
            # p_F1_sum.append(single_results['p_F1'])
            # dice_sum.append(single_results['dice'])
            aji_sum.append(single_results['aji'])
        # track_value("p_iou", np.mean(p_iou_sum), "scalar")
        # track_value("p_F1", np.mean(p_F1_sum), "scalar")
        # track_value("dice", np.mean(dice_sum), "scalar")
        track_value("aji", np.mean(aji_sum), "scalar")
    elif raw_data["phase_id"] == 0:
        over_inter = 0
        over_total = 0
        over_correct = 0
        for idx in range(len(raw_data["true_np"])):
            patch_prob_np = prob_np[idx]
            patch_true_np = true_np[idx]
            patch_pred_np = np.array(patch_prob_np > 0.5, dtype=np.int32)
            inter, total = _dice_info(patch_true_np, patch_pred_np, 1)
            correct = (patch_pred_np == patch_true_np).sum()
            over_inter += inter
            over_total += total
            over_correct += correct
        nr_pixels = len(true_np) * np.size(true_np[0])
        acc_np = over_correct / nr_pixels
        dice_np = 2 * over_inter / (over_total + 1.0e-8)
        track_value("np_acc", acc_np, "scalar")
        track_value("np_dice", dice_np, "scalar")

        over_squared_error = 0
        for idx in range(len(raw_data["true_np"])):
            patch_pred_hv = pred_hv[idx]
            patch_true_hv = true_hv[idx]
            squared_error = patch_pred_hv - patch_true_hv
            squared_error = squared_error * squared_error
            over_squared_error += squared_error.sum()
        mse = over_squared_error / nr_pixels
        track_value("hv_mse", mse, "scalar")

    # * TP statistic
    if nr_types is not None:
        pred_tp = raw_data["pred_tp"]
        true_tp = raw_data["true_tp"]
        for type_id in range(0, nr_types):
            over_inter = 0
            over_total = 0
            for idx in range(len(raw_data["true_np"])):
                patch_pred_tp = pred_tp[idx]
                patch_true_tp = true_tp[idx]
                inter, total = _dice_info(patch_true_tp, patch_pred_tp, type_id)
                over_inter += inter
                over_total += total
            dice_tp = 2 * over_inter / (over_total + 1.0e-8)
            track_value("tp_dice_%d" % type_id, dice_tp, "scalar")

    # *
    imgs = raw_data["imgs"]
    selected_idx = np.random.randint(0, len(imgs), size=(8,)).tolist()
    imgs = np.array([imgs[idx] for idx in selected_idx])
    true_np = np.array([true_np[idx] for idx in selected_idx])
    true_hv = np.array([true_hv[idx] for idx in selected_idx])
    prob_np = np.array([prob_np[idx] for idx in selected_idx])
    pred_hv = np.array([pred_hv[idx] for idx in selected_idx])
    viz_raw_data = {"img": imgs, "np": (true_np, prob_np), "hv": (true_hv, pred_hv)}

    if nr_types is not None:
        true_tp = np.array([true_tp[idx] for idx in selected_idx])
        pred_tp = np.array([pred_tp[idx] for idx in selected_idx])
        viz_raw_data["tp"] = (true_tp, pred_tp)
    viz_fig = viz_step_output(viz_raw_data, nr_types)
    track_dict["image"]["output"] = viz_fig

    return track_dict
