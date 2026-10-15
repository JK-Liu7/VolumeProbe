import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import torch
from monai.data import PersistentDataset as MonaiPersistentDataset
from monai.data.utils import pickle_hashing


def read_datalist(path, data_dir, split="training"):
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    if isinstance(records, dict):
        records = records[split]
    result = []
    for record in records:
        image = record if isinstance(record, str) else record["image"]
        image = Path(image)
        if not image.is_absolute():
            image = Path(data_dir) / image
        if not image.is_file():
            raise FileNotFoundError(image)
        result.append({"image": str(image.resolve())})
    if not result:
        raise ValueError(f"No images in {path} ({split}).")
    return result


class PersistentDataset(MonaiPersistentDataset):
    """Cache deterministic transforms; only load cache files created by this pipeline."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("hash_transform", pickle_hashing)
        super().__init__(*args, **kwargs)

    def _cachecheck(self, item_transformed):
        if self.cache_dir is None:
            return self._pre_transform(deepcopy(item_transformed))
        key = self.hash_func(item_transformed).decode("utf-8") + self.transform_hash
        path = self.cache_dir / f"{key}.pt"
        if path.is_file():
            return torch.load(path, map_location="cpu", weights_only=False)
        result = self._pre_transform(deepcopy(item_transformed))
        fd, name = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        os.close(fd)
        try:
            torch.save(result, name, pickle_protocol=self.pickle_protocol)
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        return result
