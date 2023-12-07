# coding=utf-8
import os
import sys
import time
import numpy as np
import argparse
import tqdm

import torch
import torch.nn as nn

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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model = choose_models(args.model)
    print(Model)
    args.num_classes = len(os.listdir(args.traindir))
    Model = Model(args).to(device)

    optimizer = torch.optim.SGD(Model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    criterion = nn.CrossEntropyLoss().to(device)
    losses = AverageMeter("Loss", ":.4f")

    train_loader, val_loader = get_img_dataloader(args)

    print("Start training...")
    start_time = time.time()
    txt_name = os.path.join(args.output_dir, str(start_time) + "_log.txt")
    Model.train()

    for epoch in range(args.start_epoch, args.max_epochs):
        correct = 0
        losses.reset()
        total = 0
        for i, (images, target) in tqdm.tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc="Epoch {}/{} lr {}".format(epoch, args.max_epochs, optimizer.state_dict()["param_groups"][0]["lr"]),
            leave=True,
        ):
            images = images.float().to(device)
            target = target.long().to(device)

            output = Model(images)
            cur_loss = criterion(output, target)
            if output.size(1) == 1:
                correct += (output.gt(0).eq(target).float()).sum().item()
            else:
                correct += (output.argmax(1).eq(target).float()).sum().item()

            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()

            losses.update(cur_loss.item(), images.size(0))

        train_acc = correct / losses.count
        scheduler.step()

        if epoch % 1 == 0:
            test_acc, test_loss = accuracy(Model, val_loader, criterion, args)

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                torch.save(Model.state_dict(), os.path.join(args.save_model_path, "zz_best_test_acc_model.pth"))
            print("current train acc: {:.4f} current test acc: {:.4f}".format(train_acc, test_acc))
            print("best test acc: {:.4f} best test epoch: {}".format(best_acc, best_epoch))
            print("train loss: {:.4f} test loss: {:.4f}".format(losses.avg, test_loss))

            logging.info(
                "Epoch [{}] learning rate: {:.4f}".format(epoch, optimizer.state_dict()["param_groups"][0]["lr"])
            )
            logging.info("train loss: {:.4f} test loss: {:.4f}".format(losses.avg, test_loss))
            logging.info("train acc: {:.4f} test acc: {:.4f}".format(train_acc, test_acc))
            logging.info("best test acc: {:.4f} best test epoch: {}".format(best_acc, best_epoch))

            if epoch % args.print_freq == 0:
                torch.save(
                    Model.state_dict(),
                    os.path.join(
                        args.save_model_path,
                        "{}_{}_epoch{}_test_acc{:.2f}.pth".format(args.model, args.dataset, epoch, test_acc),
                    ),
                )

    torch.save(Model.state_dict(), os.path.join(args.save_model_path, "za_last_epoch_model"))
    total_time = time.time() - start_time
    print("Total Time: ", total_time)
    logging.info("Total time: " + str(total_time) + "\n")
