import os
from data_process import get_things

pass_func = [".zero_grad", ".backward", "c_transforms.Compose"]
replace_func = {
    ".parameters()": ".trainable_params()",
    ".train()": ".set_train(True)",
    ".eval()": ".set_train(False)",
    ".forward": ".construct",
    "def forward": "def construct",
    ".to(": "#.to(",
    ".cuda()": "",
    ".size(1)": ".shape[1]",
    ".item()": ".asnumpy()",
    ".eq(": ".equal(",
    ".modules()": ".cells()",
    "transform=input_transform": "transform=None",
    "enumerate(train_loader)": "enumerate(train_enum)",
    "images.float()": "mindspore.Tensor(input_data=images, dtype=mindspore.float32)",
    "tqdm(train_loader)": "tqdm(train_enum)",
    "target.long()": "mindspore.Tensor(input_data=target, dtype=mindspore.int32)",
    ".size(0)": ".shape[0]",
    ".Transpose()(x, ).contiguous()": ".transpose(x, (0, 2, 1, 3, 4))",
    "x.size()": "x.shape",
    "optimizer.state_dict()['param_groups'][0]['lr']": "scheduler.get_last_lr()[0].asnumpy()",
    ".state_dict()": "",
    "/torch/": "/ms/",
    "mindspore.experimental.optim": "optim",
    "data[0].float()": "mindspore.Tensor(input_data=data[0], dtype=mindspore.float32)",
    "data[1].long()": "mindspore.Tensor(input_data=data[1], dtype=mindspore.int32)",
    "data = data": "data = mindspore.Tensor(input_data=data, dtype=mindspore.float32)",
    "labels = labels": "labels = mindspore.Tensor(input_data=labels, dtype=mindspore.float32)",
    "mindspore.utils.data.dataset.Dataset": "",
    "nn.AdaptiveAvgPool2d(output_size=(1, 1))": "ops.ReduceMean(keep_dims=True)",
}


def process_pass(lines):
    new_lines = []
    for line in lines:
        is_pass = False
        for func in pass_func:
            if line.find(func) != -1:
                is_pass = True
                break
        if not is_pass:
            new_lines.append(line)
    return new_lines


def process_replace(lines, file_name):
    new_lines = []
    for line in lines:
        for k, v in replace_func.items():
            if k == ".item()" and "VIT" in file_name:
                continue
            if line.find(k) != -1:
                line = line.replace(k, v)
        new_lines.append(line)
    return new_lines


def fix(lines):
    is_TrainSetLoader = False
    new_lines = []
    if_import = False
    for line in lines:
        if "TrainSetLoader" in line:
            is_TrainSetLoader = True
        if line.strip() == "import mindspore":
            if not if_import:
                if_import = True
            else:
                continue
        if "API.dataLoader" in line:
            if "train" in line:
                line = line.replace(")", ", is_train=True")
            else:
                line = line.replace(")", ", is_train=False")
            line = line[:-1]
            if is_TrainSetLoader:
                line = line + ", istd=True"
            line = line + ")\n"
        if "self.avgpool" in line and "ReduceMean" not in line:
            line = line.replace(")", ", (2, 3))")
        if "test_miou = MIOU.get()" in line:
            line = line + line[: line.find("pixel_ACC")] + "test_miou = test_miou.asnumpy().item()\n"
        if "mindspore.ops.avg_pool2d" in line:
            if "kernel_size=4" in line:
                line = (
                    line[: line.find("out")]
                    + "out = mindspore.ops.AvgPool(kernel_size=4, strides=4)(mindspore.ops.ReLU()(self.bn(out)))\n"
                )
            else:
                line = line[: line.find("out")] + "out = mindspore.ops.AvgPool(kernel_size=2, strides=2)(out)\n"
        if "train_loader, val_loader = get_img_dataloader(args)" in line:
            line = line + line[: line.find("train_loader")] + "train_enum = train_loader.create_tuple_iterator()\n"
        new_lines.append(line)
    return new_lines


def solve(project_path):
    """
    处理 Project 下所以 py 文件，预处理 + 转换

    """
    if os.path.exists(project_path):
        file_list = os.listdir(project_path)
        for f in file_list:
            f = os.path.join(project_path, f)
            if os.path.isdir(f):
                solve(f)
            else:
                file_name, extension = os.path.splitext(f)
                if extension == ".py" or extension == ".yaml":
                    lines = []
                    if file_name.endswith("train") or file_name.endswith("util") or file_name.endswith("getdataloader"):
                        lines.append("import API\n")
                    with open(f, encoding="utf-8") as file:
                        lines.extend(file.readlines())
                        if file_name.endswith("train"):
                            new_lines = get_things.main(f)
                            lines = []
                            lines.append("from mindspore.experimental import optim\n")
                            lines.append("import API\n")
                            lines.extend(new_lines)
                        lines = process_pass(lines)
                        lines = process_replace(lines, file_name)
                        lines = fix(lines)
                    with open(f, "w", encoding="utf-8") as file:
                        for line in lines:
                            # print(line, end='')
                            file.write(line)


if __name__ == "__main__":
    path = "tmp/target/Image_Classification"
    solve(path)
