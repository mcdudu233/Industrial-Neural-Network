import torch

from resnet import data


def predict(loader, model, cuda=False):
    classes = ["cat", "dog"]
    for _, image in enumerate(loader):
        index = torch.max(model(image), dim=1)[1]
        data.image_show(image[0], classes[index])
