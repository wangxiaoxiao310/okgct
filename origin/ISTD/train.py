import time
import os
from tqdm import tqdm
import logging

import torch
from torch.optim import lr_scheduler
import numpy as np

from utils.parse_args import parse_args
from utils.utils import AverageMeter
from utils.metric import mIoU
from utils.loss import SoftIoULoss
from datautil.getdataloader import get_img_dataloader
from utils.choose_models import choose_models


def logging_config(args, strat_time):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s",
        filename=str(os.path.join(args.save_data_path, "{}_{}_log.txt".format(strat_time, args.model))),
        filemode="a",
    )


def get_args(strat_time):
    # random seed
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICS"] = "2"  # select gpu
    args.checkpoint_save_dir = "./checkpoints"
    os.makedirs(args.checkpoint_save_dir, exist_ok=True)  # make result dir

    args.save_data_path = os.path.join(args.checkpoint_save_dir, "{}_{}".format(strat_time, args.model))
    os.makedirs(args.save_data_path, exist_ok=True)  # make result log
    logging_config(args, strat_time)
    logging.info("_______________________________________{}_______________________________________".format(args.model))
    logging.info(str(args))

    args.save_model_path = os.path.join(args.save_data_path, "save_models")
    os.makedirs(args.save_model_path, exist_ok=True)  # make save model dir

    return args


if __name__ == "__main__":
    print("运行时间为: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    sss = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    args = get_args(sss)

    # initial model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model = choose_models(args.model)
    print(Model)
    Model = Model(args).to(device)

    # Optimizer and lr scheduling
    optimizer = torch.optim.Adam(Model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    # Evaluation metrics
    MIOU = mIoU()
    best_miou, best_epoch = 0, 0
    losses = AverageMeter()

    # Data loading code
    train_loader, val_loader = get_img_dataloader(args)

    # training start
    print("Start training...")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.max_epochs):
        tbar = tqdm(train_loader)
        Model.train()
        losses.reset()
        epoch_start_time = time.time()
        for i, (data, labels) in enumerate(tbar):
            data = data.to(device)
            labels = labels.to(device)

            pred = Model(data)
            loss = SoftIoULoss(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), pred.shape[0])
            tbar.set_description(
                "Epoch %d, training loss %.4f learning rate %.4f" % (epoch, losses.avg, optimizer.param_groups[0]["lr"])
            )
        scheduler.step()
        train_loss = losses.avg

        # evaluate on validation set
        if epoch % 1 == 0:
            tbar = tqdm(val_loader)
            Model.eval()
            losses.reset()
            MIOU.reset()

            with torch.no_grad():
                for i, (data, labels) in enumerate(tbar):
                    data = data.to(device)
                    labels = labels.to(device)
                    pred = Model(data)
                    loss = SoftIoULoss(pred, labels)
                    losses.update(loss.item(), pred.shape[0])

                    MIOU.update(pred, labels)
                    _, mean_IOU = MIOU.get()
                    tbar.set_description("Epoch %d, test loss %.4f, mean_IoU: %.4f" % (epoch, losses.avg, mean_IOU))

            test_loss = losses.avg
            pixel_ACC, test_miou = MIOU.get()
            # remember best acc and save checkpoint
            if test_miou > best_miou:
                best_miou = test_miou
                best_epoch = epoch
                torch.save(Model.state_dict(), os.path.join(args.save_model_path, "best_miou_model.pth"))
            print(
                "Epoch [{}/{}] learning rate: {:.4f} epoch time: {}".format(
                    epoch,
                    args.max_epochs,
                    optimizer.state_dict()["param_groups"][0]["lr"],
                    time.time() - epoch_start_time,
                )
            )
            print("current test miou: {:.4f} best miou: {:.4f} best epoch: {}".format(test_miou, best_miou, best_epoch))
            print("current train loss:", train_loss, "current test loss:", test_loss)

            logging.info(
                "Epoch [{}] learning rate: {:.6f}".format(epoch, optimizer.state_dict()["param_groups"][0]["lr"])
            )
            logging.info("train loss: {:.6f} test loss: {:.6f}".format(losses.avg, test_loss))
            logging.info(
                "current test miou:{:.4f} best miou:{:.4f} best epoch: {}".format(test_miou, best_miou, best_epoch)
            )

            if epoch % args.print_freq == 0:
                torch.save(
                    Model.state_dict(),
                    os.path.join(
                        args.save_model_path,
                        "{}_{}_epoch{}_test_miou{:.2f}.pth".format(args.model, args.dataset, epoch, test_miou),
                    ),
                )  # save model

    torch.save(Model.state_dict(), os.path.join(args.save_model_path, "last_epoch_model.pth"))  # save last epoch model
    total_time = time.time() - start_time
    print("Total Time: ", total_time)
    logging.info("Total time: " + str(total_time) + "\n")
