import API
import mindspore

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

    trainset = TrainSetLoader(
        dataset_dir,
        img_id=train_img_ids,
        base_size=args.base_size,
        crop_size=args.crop_size,
        transform=None,
        suffix=args.suffix,
    )
    testset = TestSetLoader(
        dataset_dir,
        img_id=val_img_ids,
        base_size=args.base_size,
        crop_size=args.crop_size,
        transform=None,
        suffix=args.suffix,
    )

    train_data = API.dataLoader(dataset=trainset, batch_size=args.train_batch_size, is_train=True, istd=True)
    test_data = API.dataLoader(dataset=testset, batch_size=args.test_batch_size, is_train=False, istd=True)

    return train_data, test_data
