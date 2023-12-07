import os
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import random
import mindspore
from mindspore.amp import StaticLossScaler, all_finite


def dataLoader(dataset, batch_size, is_train=False, istd=False):
    if not is_train:
        if not istd:
            loader = test_dataLoader(dataset, batch_size)
        else:
            loader = istd_test_dataLoader(dataset, batch_size)
    else:
        if not istd:
            loader = train_dataLoader(dataset, batch_size)
        else:
            loader = istd_train_dataLoader(dataset, batch_size)
    return loader


def train_dataLoader(dataset, batch_size):
    trans = transforms.Compose(
        [
            vision.Decode(to_pil=True),
            vision.RandomCrop(size=32, padding=4),
            vision.RandomHorizontalFlip(0.5),
            vision.ToTensor(),
            vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), is_hwc=False),
        ]
    )

    dataset = dataset.map(operations=trans, input_columns="image")
    loader = dataset.batch(batch_size=batch_size)
    return loader


def test_dataLoader(dataset, batch_size):
    trans = transforms.Compose(
        [
            vision.Decode(to_pil=True),
            vision.ToTensor(),
            vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), is_hwc=False),
        ]
    )

    dataset = dataset.map(operations=trans, input_columns="image")
    loader = dataset.batch(batch_size=batch_size)
    return loader


def istd_train_dataLoader(dataset, batch_size):
    dataset = mindspore.dataset.GeneratorDataset(source=dataset, column_names=["image", "mask"])
    trans = transforms.Compose(
        [vision.ToTensor(), vision.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), is_hwc=False)]
    )
    dataset = dataset.map(operations=trans, input_columns="image")
    loader = dataset.batch(batch_size=batch_size)
    return loader


def istd_test_dataLoader(dataset, batch_size):
    dataset = mindspore.dataset.GeneratorDataset(source=dataset, column_names=["image", "mask"])
    trans = transforms.Compose(
        [  # vision.Decode(to_pil=True),
            vision.ToTensor(),
            vision.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), is_hwc=False),
        ]
    )

    dataset = dataset.map(operations=trans, input_columns="image")
    loader = dataset.batch(batch_size=batch_size)
    return loader


class Scheduler:
    def __init__(self, optimizer, max_epochs):
        self.optimizer = optimizer
        self.cosine_decay_lr = mindspore.nn.CosineDecayLR(0.0, 0.1, max_epochs)
        self.lr = self.cosine_decay_lr(mindspore.Tensor(0, mindspore.int32))
        self.epoch = 0

    def step(self):
        self.lr = self.cosine_decay_lr(mindspore.Tensor(self.epoch + 1, mindspore.int32))
        mindspore.ops.assign(self.optimizer.learning_rate, mindspore.Tensor(self.lr, mindspore.float32))
        self.epoch += 1


def train_step(model, loss_fn, optimizer, data, label, loss_scaler=StaticLossScaler(scale_value=1024)):
    def fn(images, target):
        logits = model(images)
        loss = loss_fn(logits, target)
        loss = loss_scaler.scale(loss)
        return loss, logits

    grad_fn = mindspore.value_and_grad(fn, None, optimizer.parameters, has_aux=True)
    (loss, logits), grads = grad_fn(data, label)
    loss = loss_scaler.unscale(loss)
    is_finite = all_finite(grads)
    if is_finite:
        grads = loss_scaler.unscale(grads)
        optimizer(grads)
    loss_scaler.adjust(is_finite)
    return loss


def get_list_two(args):
    path = "./utils/" + args.model + ".txt"
    fil_num = 400
    if "resnet" in args.model:
        fil_num = 1000
    fil_list = set()
    if not os.path.exists(path):
        return fil_list
    with open(path, "r") as f:
        for line in f.readlines():
            fil_list.add(int(line))
    fil_list = random.sample(fil_list, fil_num)
    return fil_list


def caculate(i, num, t, mset):
    if not (mset is None) and (i in mset):
        return 0, t - 1
    return num, t


class no_grad:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def load_dict(Model, name):
    # ckpt_path = "../../checkpoints/" + name.split('_')[0] + "_mindspore.ckpt"
    # ckpt_path = "/home/train/workspace/wxq/app/okgct/test/resnet_cv_model.ckpt"
    ckpt_path = name
    print(ckpt_path)
    mindspore.load_checkpoint(ckpt_path, Model)
