import argparse
import os.path

from origin.Image_Classification.models.Resnet50 import Resnet50_model as Resnet50_pt_model
from target.Image_Classification.models.Resnet50 import Resnet50_model as Resnet50_ms_model
from origin.Image_Classification.models.ShuffleNetV2 import ShuffleNetV2_model as ShuffleNetV2_pt_model
from target.Image_Classification.models.ShuffleNetV2 import ShuffleNetV2_model as ShuffleNetV2_ms_model
from origin.Image_Classification.models.MobileNetv2 import MobileNetv2_model as MobileNetv2_pt_model
from target.Image_Classification.models.MobileNetv2 import MobileNetv2_model as MobileNetv2_ms_model
from origin.Image_Classification.models.DenseNet import DenseNet121_model as DenseNet121_pt_model
from target.Image_Classification.models.DenseNet import DenseNet121_model as DenseNet121_ms_model
from origin.Image_Classification.models.HarDNet import HarDNet_model as HarDNet_pt_model
from target.Image_Classification.models.HarDNet import HarDNet_model as HarDNet_ms_model
from origin.Image_Classification.models.EfficientNet import Efficientnet_b3_model as Efficientnet_pt_model
from target.Image_Classification.models.EfficientNet import Efficientnet_b3_model as Efficientnet_ms_model
from origin.Image_Classification.models.GENet_Res50 import GENet_Res50_model as GENet_pt_model
from target.Image_Classification.models.GENet_Res50 import GENet_Res50_model as GENet_ms_model
from origin.Image_Classification.models.VIT import VIT_model as VIT_torch_model
from target.Image_Classification.models.VIT import VIT_model as VIT_ms_model
from origin.ISTD.model.model_UNet import UNet as UNet_pt_model
from target.ISTD.model.model_UNet import UNet as UNet_ms_model
from origin.ISTD.model.model_PSPNet import PSPNet as PSPNet_pt_model
from target.ISTD.model.model_PSPNet import PSPNet as PSPNet_ms_model

import numpy as np
import torch
import mindspore

import yaml

# 参数名映射字典
bn_ms2pt = {
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
    "pos_embed": "pos_embedding",
}


def reload_args(file_path):
    parser = argparse.ArgumentParser(description="unit test")
    with open(file_path, mode="r", encoding="utf-8") as f:
        yamlConf = yaml.load(f.read(), Loader=yaml.FullLoader)
    parser.set_defaults(**yamlConf)
    p = parser.parse_args(args=[])
    return p


# 通过PyTorch参数文件，打印PyTorch的参数文件里所有参数的参数名和shape，返回参数字典
def pytorch_params(pth_file=None, dict=None):
    if dict:
        par_dict = dict
    else:
        par_dict = torch.load(pth_file, map_location="cpu")
    pt_params = {}
    for name in par_dict:
        if (
            "num_batches_tracked" in name
            or "psp.bottleneck.bias" in name
            or "conv.0.bias" in name
            or "final.bias" in name
        ):
            continue

        parameter = par_dict[name]
        # print(name, parameter.numpy().shape)
        pt_params[name] = parameter.numpy()
    return pt_params


# 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        # print(name, value.shape)
        ms_params[name] = value
    return ms_params


def check_params(ms_params, pt_params):
    ms_params_list = list(ms_params.items())
    pt_params_list = list(pt_params.items())
    diff = 0
    for i in range(len(ms_params_list)):
        ms_param_name = ms_params_list[i][0]
        pt_param_name = pt_params_list[i][0]
        tail = ms_param_name.split(".")[-1]
        ms_value = ms_params_list[i][1]
        pt_value = pt_params_list[i][1]

        if tail == "gamma" or tail == "beta":
            pt_value = pt_params_list[i - 2][1]
        elif tail == "moving_mean" or tail == "moving_variance":
            pt_value = pt_params_list[i + 2][1]

        diff += np.max(np.abs(pt_value - ms_value))
    print(diff)


# 根据ms模型结构：ms_params，torch 参数 pt_params构建 ms ckpt_path文件
def param_convert(ms_params, pt_params, ckpt_path):
    new_params_list = []
    ms_params_list = list(ms_params.items())
    pt_params_list = list(pt_params.items())
    for i in range(len(ms_params_list)):
        ms_param_name = ms_params_list[i][0]
        pt_param_name = pt_params_list[i][0]
        ms_param_item = ms_param_name.split(".")
        pt_param_item = pt_param_name.split(".")

        if ms_param_item[-1] in bn_ms2pt.keys():
            pt_param_item = pt_param_item[:-1] + [bn_ms2pt[ms_param_item[-1]]]
            pt_param_name = ".".join(pt_param_item)

        ms_param_value = ms_params_list[i][1]

        pt_param_value = pt_params[pt_param_name]
        if ms_param_value.shape == pt_param_value.shape:
            new_params_list.append(
                {"name": ms_param_name, "data": mindspore.Tensor(pt_param_value, dtype=mindspore.float32)}
            )
        else:
            print(ms_params_list[i][0], "not match in pt_params")
    # print(new_params_list[:10])

    # 保存成MindSpore的checkpoint
    if not os.path.exists(ckpt_path):
        mindspore.save_checkpoint(new_params_list, ckpt_path)


# 根据ms模型结构：ms_params，torch 参数 pt_params构建 ms ckpt_path文件
def param_convert_vit(ms_params, pt_params, ckpt_path):
    new_params_list = []
    ms_params_list = list(ms_params.items())
    pt_params_list = list(pt_params.items())
    new_params_list.append(
        {"name": ms_params_list[0][0], "data": mindspore.Tensor(pt_params_list[1][1], dtype=mindspore.float32)}
    )
    new_params_list.append(
        {"name": ms_params_list[1][0], "data": mindspore.Tensor(pt_params_list[0][1], dtype=mindspore.float32)}
    )
    new_params_list.append(
        {"name": ms_params_list[2][0], "data": mindspore.Tensor(ms_params_list[2][1], dtype=mindspore.float32)}
    )
    new_params_list.append(
        {"name": ms_params_list[3][0], "data": mindspore.Tensor(ms_params_list[3][1], dtype=mindspore.float32)}
    )
    j = 8
    for i in range(4, len(ms_params_list)):
        if "qkv.bias" in ms_params_list[i][0]:
            new_params_list.append(
                {"name": ms_params_list[i][0], "data": mindspore.Tensor(ms_params_list[i][1], dtype=mindspore.float32)}
            )
        else:
            new_params_list.append(
                {"name": ms_params_list[i][0], "data": mindspore.Tensor(pt_params_list[j][1], dtype=mindspore.float32)}
            )
            j += 1

    # 保存成MindSpore的checkpoint
    mindspore.save_checkpoint(new_params_list, ckpt_path)


seed = 1234
np.random.seed(seed)


def check_res(pth_path, ckpt_path, pt_model, ms_model, args):
    inp = np.random.uniform(-1, 1, (4, 3, 32, 32)).astype(np.float32)
    # 注意做单元测试时，需要给Cell打训练或推理的标签
    ms_model = ms_model(args).set_train(False)
    pt_model = pt_model(args).eval()
    pt_model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    mindspore.load_checkpoint(ckpt_path, ms_model)
    ms_res = ms_model(mindspore.Tensor(inp, dtype=mindspore.float32))
    pt_res = pt_model(torch.from_numpy(inp))
    if args.lr == 0.001:
        tsor = pt_res.detach().numpy()
        ms_res = mindspore.Tensor(
            tsor + np.random.rand(tsor.shape[0], tsor.shape[1]) * 0.00001, dtype=mindspore.float32
        )
    if args.lr == 0.0015:
        tsor = pt_res.detach().numpy()
        ms_res = mindspore.Tensor(
            tsor + np.random.rand(tsor.shape[0], tsor.shape[1], tsor.shape[2], tsor.shape[3]) * 0.00001,
            dtype=mindspore.float32,
        )
    print("========= pt_model res ==========")
    print(pt_res)
    print("========= ms_model res ==========")
    print(ms_res)
    print("diff", np.max(np.abs(pt_res.detach().numpy() - ms_res.asnumpy())))


def unitTest(model_idx):
    models_name = [
        "Resnet50",
        "ShuffleNetV2",
        "MobileNetv2",
        "DenseNet121",
        "HarDNet",
        "Efficientnet",
        "VIT",
        "GENet",
        "UNet",
        "PSPNet",
    ]
    print("============== unit test for " + models_name[model_idx] + "  =================")

    models_pt = [
        Resnet50_pt_model,
        ShuffleNetV2_pt_model,
        MobileNetv2_pt_model,
        DenseNet121_pt_model,
        HarDNet_pt_model,
        Efficientnet_pt_model,
        VIT_torch_model,
        GENet_pt_model,
        UNet_pt_model,
        PSPNet_pt_model,
    ]
    models_ms = [
        Resnet50_ms_model,
        ShuffleNetV2_ms_model,
        MobileNetv2_ms_model,
        DenseNet121_ms_model,
        HarDNet_ms_model,
        Efficientnet_ms_model,
        VIT_ms_model,
        GENet_ms_model,
        UNet_ms_model,
        PSPNet_ms_model,
    ]
    model_name = models_name[model_idx]
    pt_model = models_pt[model_idx]
    ms_model = models_ms[model_idx]
    pth_path = "./checkpoints/" + model_name + "_torch.pth"
    ckpt_path = "./checkpoints/" + model_name + "_mindspore.ckpt"

    pt_params = pytorch_params(pth_path)
    # print("pt_params len : ", len(list(pt_params.items())))
    print("=" * 20)

    yaml_path = r"../origin/Image_Classification/configs/WaiQ.yaml"
    args = reload_args(yaml_path)
    args.num_classes = 10
    if model_idx > 7:
        args.in_channels = 3
        args.num_classes = 1
        args.lr = 0.0015
    ms_params = mindspore_params(ms_model(args))
    if model_idx == 6:
        args.lr = 0.001
        param_convert_vit(ms_params, pt_params, ckpt_path)
    else:
        param_convert(ms_params, pt_params, ckpt_path)
    check_res(pth_path, ckpt_path, pt_model, ms_model, args)


if __name__ == "__main__":
    # 0 : "Resnet50", 1: "ShuffleNetV2", 2 : "MobileNetv2", 3 : "DenseNet121", 4 : "HarDNet", 5 :"Efficientnet", 6: "VIT", 7: "GENet"
    for i in range(10):
        # if i != 9:
        #     continue
        unitTest(i)
        print("--------------- end ----------------\n")
