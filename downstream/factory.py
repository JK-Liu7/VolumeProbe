from collections.abc import Mapping

import torch
from monai.networks.nets import SwinUNETR

from downstream.models.unetr import UNETR


BASELINES = ("VoCo", "S2DC", "MedGMAE")


def checkpoint_weights(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("A checkpoint must contain a state dictionary.")
    for key in ("state_dict", "model", "model_state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], Mapping):
            checkpoint = checkpoint[key]
            break
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def load_pretrained(model, path, baseline, logger=None, checkpoint=None):
    if baseline not in BASELINES:
        raise ValueError(f"Unsupported baseline: {baseline}. Choose from {BASELINES}.")
    if checkpoint is None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint_weights(checkpoint)
    target = model.state_dict()
    mapped = {}
    has_student = any(key.startswith("student.") for key in source)
    for key, value in source.items():
        if baseline in ("VoCo", "S2DC"):
            if has_student and not key.startswith("student."):
                continue
            key = key.removeprefix("student.").replace("swin_vit", "swinViT")
        else:
            if key == "encoder_pos_embed":
                key = "vit.patch_embedding.position_embeddings"
            elif key.startswith("encoder.patch_embed.proj."):
                suffix = key.rsplit(".", 1)[1]
                key = f"vit.patch_embedding.patch_embeddings.{suffix}"
                if key not in target:
                    key = f"vit.patch_embedding.patch_embeddings.1.{suffix}"
                    if suffix == "weight":
                        value = value.flatten(1)
            elif key.startswith("encoder.") and "cls_token" not in key:
                key = key.replace("encoder.", "vit.", 1)
                key = key.replace(".mlp.fc1.", ".mlp.linear1.")
                key = key.replace(".mlp.fc2.", ".mlp.linear2.")
                key = key.replace(".attn.proj.", ".attn.out_proj.")
            else:
                continue
        if key in target:
            if value.shape != target[key].shape:
                raise ValueError(
                    f"Pretrained tensor {key} has shape {tuple(value.shape)}; "
                    f"expected {tuple(target[key].shape)}. Match the model dimensions to the checkpoint."
                )
            mapped[key] = value
    prefix = "vit." if baseline == "MedGMAE" else "swinViT."
    if not any(key.startswith(prefix) for key in mapped):
        raise ValueError(f"No {baseline} encoder weights matched the selected model.")
    model.load_state_dict(mapped, strict=False)
    inactive = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    report = {
        "loaded_tensors": len(mapped),
        "loaded_parameters": sum(value.numel() for value in mapped.values()),
        "encoder_tensors_loaded": sum(key.startswith(prefix) for key in mapped),
        "encoder_tensors_total": sum(key.startswith(prefix) for key in target),
        "active_encoder_tensors_total": sum(
            key.startswith(prefix) and key not in inactive for key in target
        ),
    }
    if logger:
        logger.info("Pretrained weight loading: %s", report)
    return report


def build_model(args):
    if args.model not in BASELINES:
        raise ValueError(f"Unsupported baseline: {args.model}. Choose from {BASELINES}.")
    roi = (args.roi_x, args.roi_y, args.roi_z)
    if args.model in ("VoCo", "S2DC"):
        if any(size % 32 for size in roi):
            raise ValueError("VoCo/S2DC ROI dimensions must be divisible by 32.")
        return SwinUNETR(
            img_size=roi,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=args.dropout_rate,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=args.use_checkpoint,
            use_v2=args.swin_v2,
        )
    if any(size % 12 for size in roi):
        raise ValueError("MedGMAE ROI dimensions must be divisible by 12.")
    return UNETR(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        img_size=roi,
        feature_size=args.unetr_feature_size,
        hidden_size=args.hidden_size,
        mlp_dim=args.mlp_dim,
        num_heads=args.num_heads,
        pos_embed="learnable",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=args.dropout_rate,
        qkv_bias=True,
    )
