from functools import partial

import torch
from monai.data import DataLoader

from models.fix.med_gmae3d import MedGMAE3D
from models.fix.networks.vit import MAEViTEncoder
from models.fix.networks.med_gaussian_decoder import MedGaussianViTDecoder
from utils.data_utils_volumeprobe_cache import MedGMAEProbeDatasetFactory, _identity_collate


def build_model(args):
    base = args.model == "vit_base"
    enc_dim = args.encoder_dim or (768 if base else 1536)
    enc_heads = args.encoder_heads or (12 if base else 16)
    dec_dim = args.decoder_dim or (384 if base else 528)
    dec_heads = args.decoder_heads or (12 if base else 16)
    if enc_dim % 6 or dec_dim % 6 or enc_dim % enc_heads or dec_dim % dec_heads:
        raise ValueError(
            "Embedding dimensions must be divisible by six and their attention head counts."
        )
    encoder = partial(
        MAEViTEncoder, embed_dim=enc_dim, depth=args.encoder_depth or 12, num_heads=enc_heads
    )
    decoder = partial(
        MedGaussianViTDecoder, embed_dim=dec_dim, depth=args.decoder_depth or 8, num_heads=dec_heads
    )
    model = MedGMAE3D(encoder, decoder, args)
    if args.freeze_encoder:
        model.encoder.requires_grad_(False)
    return model


def build_loader(args):
    from torch.utils.data.distributed import DistributedSampler

    dataset = MedGMAEProbeDatasetFactory(args).make_train_dataset()
    sampler = (
        DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, seed=args.seed)
        if args.distributed
        else None
    )
    kwargs = {"prefetch_factor": args.prefetch_factor} if args.num_workers else {}
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        collate_fn=_identity_collate,
        **kwargs,
    )


def compute_loss(model, batch, device):
    crops = [
        crop for sample in batch for crop in (sample if isinstance(sample, list) else [sample])
    ]
    image = torch.stack(
        [
            crop["image"].as_tensor() if hasattr(crop["image"], "as_tensor") else crop["image"]
            for crop in crops
        ]
    )
    image = image.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    loss = model(image, return_image=False)
    return loss, {}
