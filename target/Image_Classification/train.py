from mindspore.experimental import optim
import API
import os
import sys
import time
import numpy as np
import argparse
import tqdm

import mindspore

from datautil.getdataloader import get_img_dataloader
import logging
from utils.load_yaml import reload_args
from utils.choose_models import choose_models as choose_models
from utils.util import *


def logging_config(args, strat_time):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s",
        filename=str(os.path.join(args.save_model_path, "{}_{}_log.txt".format(strat_time, args.model))),
        filemode="a",
    )


def get_args():
    seed = 1234
    np.random.seed(seed)
    mindspore.set_seed(seed=seed)

    parser = argparse.ArgumentParser(description="WYQ Image Classification")

    return parser


def super_load_args_and_initialization(yaml_path, strat_time):
    args = reload_args(yaml_path, get_args())
    os.environ["CUDA_VISIBLE_DEVICS"] = "0"

    args.save_model_path = os.path.join(args.checkpoint_save_dir, "{}_{}".format(strat_time, args.model))
    os.makedirs(args.save_model_path, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    sys.stdout = Tee(os.path.join(args.output_dir, "train_out.txt"))
    sys.stderr = Tee(os.path.join(args.output_dir, "train_err.txt"))

    logging_config(args, strat_time)
    logging.info("__________________________________new__________________________________")
    logging.info(str(args))
    args.save_model_path = os.path.join(args.save_model_path, "save_models")
    os.makedirs(args.save_model_path, exist_ok=True)

    return args


if __name__ == "__main__":
    print("运行时间为: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    sss = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    yaml_path = r"./configs/WaiQ.yaml"
    args = super_load_args_and_initialization(yaml_path, sss)
    print(args)
    best_acc, best_epoch = 0, 0

    device = mindspore.context.set_context(device_target="GPU")
    Model = choose_models(args.model)
    print(Model)
    args.num_classes = len(os.listdir(args.traindir))
    Model = Model(args)  # .to(device)

    optimizer = optim.SGD(
        params=Model.trainable_params(), lr=args.lr, momentum=0.9, dampening=0, weight_decay=5e-4, nesterov=False
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.max_epochs, eta_min=0, last_epoch=-1
    )
    criterion = mindspore.nn.CrossEntropyLoss()  # .to(device)
    losses = AverageMeter("Loss", ":.4f")

    train_loader, val_loader = get_img_dataloader(args)
    train_enum = train_loader.create_tuple_iterator()

    print("Start training...")
    start_time = time.time()
    txt_name = os.path.join(args.output_dir, str(start_time) + "_log.txt")
    Model.set_train(True)

    for epoch in range(args.start_epoch, args.max_epochs):
        correct = 0
        losses.reset()
        total = 0
        for i, (images, target) in tqdm.tqdm(
            enumerate(train_enum),
            total=len(train_loader),
            desc="Epoch {}/{} lr {}".format(epoch, args.max_epochs, scheduler.get_last_lr()[0].asnumpy()),
            leave=True,
        ):
            images = mindspore.Tensor(input_data=images, dtype=mindspore.float32)  # .to(device)
            target = mindspore.Tensor(input_data=target, dtype=mindspore.int32)  # .to(device)

            output = Model(images)
            #            cur_loss = criterion(output, target)
            if output.shape[1] == 1:
                correct += (output.gt(0).equal(target).float()).sum().asnumpy()
            else:
                correct += (output.argmax(1).equal(target).float()).sum().asnumpy()

            cur_loss = API.train_step(Model, criterion, optimizer, images, target)

            losses.update(cur_loss.asnumpy(), images.shape[0])

        train_acc = correct / losses.count
        scheduler.step()

        if epoch % 1 == 0:
            test_acc, test_loss = accuracy(Model, val_loader, criterion, args)

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                mindspore.save_checkpoint(
                    save_obj=Model,
                    ckpt_file_name=os.path.join(args.save_model_path, "zz_best_test_acc_model.pth"),
                    integrated_save=True,
                    async_save=False,
                    append_dict=None,
                    enc_key=None,
                    enc_mode="AES_GCM",
                )
            print("current train acc: {:.4f} current test acc: {:.4f}".format(train_acc, test_acc))
            print("best test acc: {:.4f} best test epoch: {}".format(best_acc, best_epoch))
            print("train loss: {:.4f} test loss: {:.4f}".format(losses.avg, test_loss))

            logging.info("Epoch [{}] learning rate: {:.4f}".format(epoch, scheduler.get_last_lr()[0].asnumpy()))
            logging.info("train loss: {:.4f} test loss: {:.4f}".format(losses.avg, test_loss))
            logging.info("train acc: {:.4f} test acc: {:.4f}".format(train_acc, test_acc))
            logging.info("best test acc: {:.4f} best test epoch: {}".format(best_acc, best_epoch))

            if epoch % args.print_freq == 0:
                mindspore.save_checkpoint(
                    save_obj=Model,
                    ckpt_file_name=os.path.join(
                        args.save_model_path,
                        "{}_{}_epoch{}_test_acc{:.2f}.pth".format(args.model, args.dataset, epoch, test_acc),
                    ),
                    integrated_save=True,
                    async_save=False,
                    append_dict=None,
                    enc_key=None,
                    enc_mode="AES_GCM",
                )

    mindspore.save_checkpoint(
        save_obj=Model,
        ckpt_file_name=os.path.join(args.save_model_path, "za_last_epoch_model"),
        integrated_save=True,
        async_save=False,
        append_dict=None,
        enc_key=None,
        enc_mode="AES_GCM",
    )
    total_time = time.time() - start_time
    print("Total Time: ", total_time)
    logging.info("Total time: " + str(total_time) + "\n")
