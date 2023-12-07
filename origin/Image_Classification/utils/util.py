import torch
import sys
import tqdm


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(network, loader, criterion, args):
    correct = 0
    total = 0
    test_loss = 0

    network.eval()
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(loader), total=len(loader)):
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = network.forward(x)

            loss = criterion(p, y)
            test_loss += loss.item() * x.size(0)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()

            total += len(x)
    network.train()
    return correct / total, test_loss / total
