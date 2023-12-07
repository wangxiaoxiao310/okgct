from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datautil.setdataloader import TrainSetLoader, TestSetLoader


def load_dataset(root, dataset, split_method):
    train_txt = root + "/" + dataset + "/" + split_method + "/" + "train.txt"
    test_txt = root + "/" + dataset + "/" + split_method + "/" + "test.txt"
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split("\n")[0])
            line = f.readline()
        f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split("\n")[0])
            line = f.readline()
        f.close()
    return train_img_ids, val_img_ids, test_txt


def get_img_dataloader(args):
    dataset_dir = args.root + "/" + args.dataset
    train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

    # Preprocess and load data
    input_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    trainset = TrainSetLoader(
        dataset_dir,
        img_id=train_img_ids,
        base_size=args.base_size,
        crop_size=args.crop_size,
        transform=input_transform,
        suffix=args.suffix,
    )
    testset = TestSetLoader(
        dataset_dir,
        img_id=val_img_ids,
        base_size=args.base_size,
        crop_size=args.crop_size,
        transform=input_transform,
        suffix=args.suffix,
    )

    train_data = DataLoader(
        dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers, drop_last=True
    )
    test_data = DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, drop_last=False
    )

    return train_data, test_data
