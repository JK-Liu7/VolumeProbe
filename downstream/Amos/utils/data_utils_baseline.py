from downstream.data import get_loader as _get_loader
from downstream.preprocessing import get_transforms as _get_transforms


def get_loader(args):
    return _get_loader(args, "Amos")


def get_transforms(args):
    return _get_transforms(args, "Amos")
