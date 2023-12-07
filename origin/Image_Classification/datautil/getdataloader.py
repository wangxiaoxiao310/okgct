# coding=utf-8
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def get_img_dataloader(args):
    train_dataset = datasets.ImageFolder(args.traindir, transform_train)
    test_dataset = datasets.ImageFolder(args.testdir, transform_test)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=args.train_shuffle, num_workers=args.workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=args.test_shuffle,
        num_workers=args.workers,
        pin_memory=True,
    )

    return train_loader, test_loader
