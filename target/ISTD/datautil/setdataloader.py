import mindspore
from PIL import Image, ImageOps, ImageFilter
import random
import numpy as np


class TrainSetLoader:
    def __init__(self, dataset_dir, img_id, base_size=512, crop_size=480, transform=None, suffix=".png"):
        super(TrainSetLoader, self).__init__()

        self.transform = transform
        self._items = img_id
        self.masks = dataset_dir + "/" + "masks"
        self.images = dataset_dir + "/" + "images"
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

    def _sync_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        return img, mask

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self.images + "/" + img_id + self.suffix
        label_path = self.masks + "/" + img_id + self.suffix

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path)

        img, mask = self._sync_transform(img, mask)

        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype("float32") / 255.0

        return img, mindspore.Tensor.from_numpy(array=mask)

    def __len__(self):
        return len(self._items)


class TestSetLoader:
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id, transform=None, base_size=512, crop_size=480, suffix=".png"):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items = img_id
        self.masks = dataset_dir + "/" + "masks"
        self.images = dataset_dir + "/" + "images"
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        return img, mask

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self.images + "/" + img_id + self.suffix
        label_path = self.masks + "/" + img_id + self.suffix
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path)
        img, mask = self._testval_sync_transform(img, mask)

        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype("float32") / 255.0

        return img, mindspore.Tensor.from_numpy(array=mask)

    def __len__(self):
        return len(self._items)
