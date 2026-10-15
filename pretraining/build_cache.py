import argparse
import json
from dataclasses import asdict
from pathlib import Path

from monai.data import load_decathlon_datalist

from pretraining.data import read_datalist
from volumeprobe_lite import VolumeProbeLiteBuilder, VolumeProbeLiteConfig


def main(dataset):
    parser = argparse.ArgumentParser(
        description=f"Build a static VolumeProbe-Lite cache for {dataset}."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument(
        "--datalist", help="JSON training list with image paths relative to data_root."
    )
    parser.add_argument("--list_dir", help="VoCo JSON-list directory.")
    parser.add_argument("--dataset_train_list", help="AbdomenAtlas case-name list.")
    parser.add_argument(
        "--dataset_val_list",
        help="Optional additional case list; included only when explicitly supplied.",
    )
    parser.add_argument("--dataset_variant", choices=["full", "1k"], default="full")
    parser.add_argument("--failed_log_path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--failure_mode", choices=["raise", "continue"], default="raise")
    parser.add_argument(
        "--preprocessing",
        choices=["voco", "standard"],
        default="voco" if dataset == "VoCo10k" else "standard",
        help="VoCo uses the original full-foreground resize; S2DC/MedGMAE use standard spacing preprocessing.",
    )
    parser.add_argument("--axcodes", default="RAS")
    parser.add_argument("--pixdim", type=float, nargs=3, default=(1.5, 1.5, 1.5))
    parser.add_argument("--spacing_mode", default="bilinear")
    parser.add_argument("--a_min", type=float, default=-175.0)
    parser.add_argument("--a_max", type=float, default=250.0)
    parser.add_argument("--b_min", type=float, default=0.0)
    parser.add_argument("--b_max", type=float, default=1.0)
    parser.add_argument("--fg_threshold", type=float, default=0.3)
    parser.add_argument("--coarse_grid", type=int, nargs=3, default=(4, 4, 2))
    parser.add_argument("--top_coarse_bins", type=int, default=24)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--fine_offset_frac", type=float, default=0.25)
    parser.add_argument(
        "--roi_size",
        type=int,
        nargs=3,
        default=(64, 64, 64) if dataset == "VoCo10k" else (96, 96, 96),
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Informativeness weight in A + alpha*I - beta*D."
    )
    parser.add_argument(
        "--beta", type=float, default=1.0, help="Redundancy penalty weight in A + alpha*I - beta*D."
    )
    parser.add_argument("--min_center_distance_norm", type=float)
    parser.add_argument("--min_center_distance_factor", type=float, default=0.35)
    parser.add_argument("--entropy_bins", type=int, default=32)
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    if not root.is_dir():
        parser.error("data_root must be an existing directory")
    if args.datalist:
        datalist = read_datalist(args.datalist, root)
    elif dataset == "VoCo10k":
        if not args.list_dir:
            parser.error("Pass --datalist or --list_dir")
        groups = [
            ("btcv.json", "BTCV"),
            ("dataset_TCIAcovid19_0.json", "TCIAcovid19"),
            ("dataset_LUNA16_0.json", "Luna16"),
            ("stoic21.json", "stoic21"),
            ("Totalsegmentator_dataset.json", "Totalsegmentator_dataset"),
            ("flare23.json", "Flare23"),
            ("HNSCC.json", "HNSCC_convert_v1"),
        ]
        datalist = []
        for json_name, directory in groups[:3] if args.dataset_variant == "1k" else groups:
            datalist.extend(
                load_decathlon_datalist(
                    str(Path(args.list_dir) / json_name),
                    False,
                    "training",
                    base_dir=str(root / directory),
                )
            )
    else:
        if not args.dataset_train_list:
            parser.error("Pass --datalist or --dataset_train_list")
        names = []
        for path in (args.dataset_train_list, args.dataset_val_list):
            if path:
                with open(path, encoding="utf-8") as handle:
                    names.extend(line.strip().split()[0] for line in handle if line.strip())
        datalist = [{"image": str(root / "imagesTr" / f"{name}_0000.nii.gz")} for name in names]
    datalist = [
        {"image": image}
        for image in dict.fromkeys(str(Path(item["image"]).resolve()) for item in datalist)
    ]
    if not datalist:
        parser.error("The input list is empty")
    cfg = VolumeProbeLiteConfig(
        **{
            key: getattr(args, key)
            for key in VolumeProbeLiteConfig.__dataclass_fields__
            if hasattr(args, key)
        }
    )
    cfg.intensity_clip = True
    paths = VolumeProbeLiteBuilder(cfg).build_and_cache_all(
        datalist,
        args.cache_dir,
        overwrite=args.overwrite,
        failure_mode=args.failure_mode,
        failed_log_path=args.failed_log_path,
    )
    manifest = {
        "num_images": len(datalist),
        "num_cached": len(paths),
        "config": asdict(cfg),
        "images": [x["image"] for x in datalist],
    }
    (Path(args.cache_dir) / "volumeprobe_cache_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if len(paths) != len(datalist):
        raise RuntimeError(
            f"Only {len(paths)} of {len(datalist)} images were cached; inspect the failure log."
        )
