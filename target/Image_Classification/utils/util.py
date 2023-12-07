import API
import mindspore
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

    network.set_train(False)
    with API.no_grad():
        for i, data in tqdm.tqdm(enumerate(loader), total=len(loader)):
            x = mindspore.Tensor(input_data=data[0], dtype=mindspore.float32)
            y = mindspore.Tensor(input_data=data[1], dtype=mindspore.int32)
            p = network.construct(x)

            loss = criterion(p, y)
            test_loss += loss.asnumpy() * x.shape[0]

            if p.shape[1] == 1:
                correct += (p.gt(0).equal(y).float()).sum().asnumpy()
            else:
                correct += (p.argmax(1).equal(y).float()).sum().asnumpy()

            total += len(x)
    network.set_train(True)
    return correct / total, test_loss / total
