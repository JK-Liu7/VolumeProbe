from pathlib import Path

import torch
from monai.data import DataLoader, Dataset, load_decathlon_datalist
from torch.utils.data.distributed import DistributedSampler

from downstream.preprocessing import get_transforms
from pretraining.data import PersistentDataset


class EvaluationSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank, world_size):
        self.indices = list(range(rank, len(dataset), world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def resolve_manifest(json_list, data_dir):
    path = Path(json_list).expanduser()
    candidates = [path] if path.is_absolute() else [path, Path(data_dir) / path]
    existing = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    if len(existing) > 1:
        raise ValueError(f"Ambiguous split JSON {json_list}; supply an absolute path.")
    if not existing:
        raise FileNotFoundError(f"Split JSON not found: {json_list} (data_dir={data_dir}).")
    return existing.pop()


def get_loader(args, dataset=None):
    dataset = dataset or args.dataset
    train_transform, val_transform = get_transforms(args, dataset)
    manifest = resolve_manifest(args.json_list, args.data_dir)

    def make_dataset(split, transform):
        records = load_decathlon_datalist(str(manifest), True, split, base_dir=args.data_dir)
        if not records:
            raise ValueError(f"The {split} split is empty in {manifest}.")
        for record in records:
            if "label" not in record:
                raise ValueError(
                    f"The {split} split has an unlabeled image in {manifest}; "
                    "Dice evaluation and training require labels. Use a labeled split."
                )
            for key in ("image", "label"):
                values = record.get(key)
                values = values if isinstance(values, list) else [values]
                if any(value is None or not Path(value).is_file() for value in values):
                    raise FileNotFoundError(f"Missing {key} file in {record}.")
        if args.cache_dir:
            return PersistentDataset(
                records, transform, cache_dir=Path(args.cache_dir) / dataset / split
            )
        return Dataset(records, transform)

    def make_loader(ds, training):
        sampler = None
        if args.distributed:
            sampler = (
                DistributedSampler(
                    ds,
                    num_replicas=args.world_size,
                    rank=args.rank,
                    seed=args.seed,
                    shuffle=True,
                )
                if training
                else EvaluationSampler(ds, args.rank, args.world_size)
            )
        return DataLoader(
            ds,
            batch_size=args.batch_size if training else 1,
            shuffle=training and sampler is None,
            sampler=sampler,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=args.device.type == "cuda",
        )

    val_ds = make_dataset(args.eval_split if args.eval_only else "validation", val_transform)
    val_loader = make_loader(val_ds, False)
    if args.eval_only:
        return [None, val_loader], None
    train_ds = make_dataset("training", train_transform)
    return [make_loader(train_ds, True), val_loader], train_ds
