import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Target Detection")

    parser.add_argument(
        "--dataset", type=str, default="NUAA-SIRST", help="dataset name: NUDT-SIRST, NUAA-SIRST, NUST-SIRST"
    )
    parser.add_argument("--root", type=str, default="dataset/")
    parser.add_argument("--suffix", type=str, default=".png")
    parser.add_argument("--split_method", type=str, default="50_50", help="50_50, 10000_100(for NUST-SIRST)")
    parser.add_argument("--base_size", type=int, default=256, help="base image size")
    parser.add_argument("--crop_size", type=int, default=256, help="crop image size")
    parser.add_argument("--in_channels", type=int, default=3, help="in_channel=3 for pre-process")
    parser.add_argument("--num_classes", type=int, default=1, help="num_classes=1 for task")

    parser.add_argument("--workers", type=int, default=4, metavar="N", help="dataloader threads")
    parser.add_argument(
        "--max_epochs", type=int, default=300, metavar="N", help="number of epochs to train (default: 110)"
    )
    parser.add_argument("--start_epoch", type=int, default=0, metavar="N", help="start epochs (default:0)")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, metavar="N", help="input batch size for \training (default: 4)"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=16, metavar="N", help="input batch size for \testing (default: 32)"
    )
    parser.add_argument("--lr", type=float, default=0.0015, metavar="LR", help="learning rate (default: 0.0015)")
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--model", type=str, default="UNet", help="model name: UNet, PSPNet")

    args = parser.parse_args()
    return args
