# some methods are taken from here and rewritten
# https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html

import os
import time

import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove("temp.p")


def inference_time(model, dataloader, data_keys):
    model = model.to("cpu")
    model.eval()
    image_key = data_keys[0]
    elapsed = 0
    num_batches = 5
    batch_size = 1
    for i, batch in enumerate(dataloader):
        if i < num_batches:
            start = time.time()
            model(batch[image_key])
            end = time.time()
            elapsed = elapsed + (end - start)
        else:
            batch_size = batch[image_key].size()[0]
            break

    num_images = batch_size * num_batches
    print("Inference time: %3.0f ms" % (elapsed / num_images * 1000))
