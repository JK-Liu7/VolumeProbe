import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pretraining.runtime import add_runtime_args, run_training
from medgmae_trainer import build_model, build_loader, compute_loss


def get_args_parser():
    parser = argparse.ArgumentParser("MedGMAE CUDA pretraining")
    parser.add_argument("--model_type", choices=["med_gmae"], default="med_gmae")
    parser.add_argument(
        "--model",
        type=str,
        default="vit_large_12p",
        choices=["vit_base", "vit_large", "vit_large_12p"],
    )

    parser.add_argument("--pretrained_ckpt", type=str)

    parser.add_argument("--base_lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size per rank")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=5e-2)

    parser.add_argument("--num_samples", default=3, type=int)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--patch_size", type=int, default=12)
    parser.add_argument("--in_chans", type=int, default=1)
    parser.add_argument("--pos_embed_type", choices=["sincos", "perceptron"], default="sincos")
    parser.add_argument("--a_min", type=float, default=-175.0)
    parser.add_argument("--a_max", type=float, default=250.0)
    parser.add_argument("--b_min", type=float, default=0.0)
    parser.add_argument("--b_max", type=float, default=1.0)

    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--num_gaussians", type=int, default=512)
    parser.add_argument("--gaussian_scale_factor", type=float, default=1.0)
    parser.add_argument("--use_image_norm", action="store_true")
    parser.add_argument("--max_gaussian_scale", type=float, default=0.5)
    parser.add_argument("--scale_penalty_weight", type=float, default=0.0)

    parser.add_argument("--decoder_type", type=str, default="standard", choices=["standard"])
    parser.add_argument("--freeze_encoder", action="store_true")

    parser.add_argument("--dataset_variant", type=str, default="full")
    parser.add_argument("--probe_fg_threshold", type=float, default=0.3)
    parser.add_argument("--probe_coarse_grid", type=int, nargs=3, default=(4, 4, 2))
    parser.add_argument("--probe_top_coarse_bins", type=int, default=24)
    parser.add_argument("--probe_fine_offset_frac", type=float, default=0.25)
    parser.add_argument("--probe_top_k", type=int, default=40)
    parser.add_argument("--probe_alpha", type=float, default=1.0)
    parser.add_argument("--probe_beta", type=float, default=1.0)
    parser.add_argument("--probe_min_center_distance_norm", type=float, default=None)
    parser.add_argument("--probe_min_center_distance_factor", type=float, default=0.35)
    parser.add_argument("--probe_entropy_bins", type=int, default=32)

    parser.add_argument(
        "--dataset_train_list", help="AbdomenAtlas case list; used when datalist is omitted."
    )
    parser.add_argument("--encoder_dim", type=int)
    parser.add_argument("--encoder_depth", type=int)
    parser.add_argument("--encoder_heads", type=int)
    parser.add_argument("--decoder_dim", type=int)
    parser.add_argument("--decoder_depth", type=int)
    parser.add_argument("--decoder_heads", type=int)
    return add_runtime_args(parser)


def main():
    args = get_args_parser().parse_args()
    if args.img_size % args.patch_size or not 0 < args.mask_ratio < 1:
        raise ValueError(
            "img_size must be divisible by patch_size; mask_ratio must be between zero and one."
        )
    if int((args.img_size // args.patch_size) ** 3 * (1 - args.mask_ratio)) < 1:
        raise ValueError("The mask must leave at least one visible patch.")
    if args.in_chans != 1 or args.num_samples < 1 or args.num_gaussians < 1:
        raise ValueError(
            "The Gaussian renderer requires one channel and positive num_samples/num_gaussians."
        )
    run_training(args, build_model, build_loader, compute_loss)


if __name__ == "__main__":
    main()
