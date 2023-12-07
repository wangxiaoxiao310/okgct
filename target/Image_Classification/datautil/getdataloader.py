import API
import mindspore

# ****torchvision.transforms.RandomCrop


def get_img_dataloader(args):
    train_dataset = mindspore.dataset.ImageFolderDataset(
        dataset_dir=args.traindir,
        num_samples=None,
        num_parallel_workers=None,
        shuffle=None,
        sampler=None,
        extensions=None,
        class_indexing=None,
        decode=False,
        num_shards=None,
        shard_id=None,
        cache=None,
    )
    test_dataset = mindspore.dataset.ImageFolderDataset(
        dataset_dir=args.testdir,
        num_samples=None,
        num_parallel_workers=None,
        shuffle=None,
        sampler=None,
        extensions=None,
        class_indexing=None,
        decode=False,
        num_shards=None,
        shard_id=None,
        cache=None,
    )

    train_loader = API.dataLoader(dataset=train_dataset, batch_size=args.batch_size, is_train=True)

    test_loader = API.dataLoader(dataset=test_dataset, batch_size=args.test_batch_size, is_train=False)

    return train_loader, test_loader
