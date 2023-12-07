import mindspore


def SoftIoULoss(pred, target):
    pred = mindspore.ops.Sigmoid()(pred)
    smooth = 1

    intersection = pred * target
    loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)

    loss = 1 - loss.mean()

    return loss
