from PIL import Image
import numpy as np
import shutil
from matplotlib import pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_Pred_GT(pred, labels, target_image_path, val_img_ids, num, suffix):
    predsss = np.array((pred > 0).cpu()).astype("int64") * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(target_image_path + "/" + "%s_Pred" % (val_img_ids[num]) + suffix)
    img = Image.fromarray(labelsss.reshape(256, 256))
    img.save(target_image_path + "/" + "%s_GT" % (val_img_ids[num]) + suffix)


def total_visulization_generation(dataset_dir, test_txt, suffix, target_image_path, target_dir):
    source_image_path = dataset_dir + "/images"

    txt_path = test_txt
    ids = []
    with open(txt_path, "r") as f:
        ids += [line.strip() for line in f.readlines()]

    for i in range(len(ids)):
        source_image = source_image_path + "/" + ids[i] + suffix
        target_image = target_image_path + "/" + ids[i] + suffix
        shutil.copy(source_image, target_image)
    for i in range(len(ids)):
        source_image = target_image_path + "/" + ids[i] + suffix
        img = Image.open(source_image)
        img = img.resize((256, 256), Image.ANTIALIAS)
        img.save(source_image)
    for m in range(len(ids)):
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        img = plt.imread(target_image_path + "/" + ids[m] + suffix)
        plt.imshow(img, cmap="gray")
        plt.xlabel("Raw Imamge", size=11)

        plt.subplot(1, 3, 2)
        img = plt.imread(target_image_path + "/" + ids[m] + "_GT" + suffix)
        plt.imshow(img, cmap="gray")
        plt.xlabel("Ground Truth", size=11)

        plt.subplot(1, 3, 3)
        img = plt.imread(target_image_path + "/" + ids[m] + "_Pred" + suffix)
        plt.imshow(img, cmap="gray")
        plt.xlabel("Predicts", size=11)

        plt.savefig(target_dir + "/" + ids[m].split(".")[0] + "_fuse" + suffix, facecolor="w", edgecolor="red")
