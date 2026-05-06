from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from monai.data import load_decathlon_datalist


# -----------------------------------------------------------------------------
# Default VoCo-10k paths on your environment.
# -----------------------------------------------------------------------------
DEFAULT_VOCO_ROOT = None
DEFAULT_LIST_DIR = os.path.join(DEFAULT_VOCO_ROOT, "jsons")
DEFAULT_DATA_DIRS = {
    "BTCV": os.path.join(DEFAULT_VOCO_ROOT, "BTCV"),
    "TCIAcovid19": os.path.join(DEFAULT_VOCO_ROOT, "TCIAcovid19"),
    "Luna16": os.path.join(DEFAULT_VOCO_ROOT, "Luna16"),
    "stoic21": os.path.join(DEFAULT_VOCO_ROOT, "stoic21"),
    "Totalsegmentator_dataset": os.path.join(DEFAULT_VOCO_ROOT, "Totalsegmentator_dataset"),
    "Flare23": os.path.join(DEFAULT_VOCO_ROOT, "Flare23"),
    "HNSCC_convert_v1": os.path.join(DEFAULT_VOCO_ROOT, "HNSCC_convert_v1"),
}


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from volumeprobe_lite import VolumeProbeLiteBuilder, VolumeProbeLiteConfig


# -----------------------------------------------------------------------------
# VoCo datalist construction.
# -----------------------------------------------------------------------------
def _strip_to_image_only(items: Sequence[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for item in items:
        out.append({"image": item["image"]})
    return out



def build_voco_train_datalist_full(
    list_dir: str = DEFAULT_LIST_DIR,
    datadir1: str = DEFAULT_DATA_DIRS["BTCV"],
    datadir2: str = DEFAULT_DATA_DIRS["TCIAcovid19"],
    datadir3: str = DEFAULT_DATA_DIRS["Luna16"],
    datadir4: str = DEFAULT_DATA_DIRS["stoic21"],
    datadir5: str = DEFAULT_DATA_DIRS["Totalsegmentator_dataset"],
    datadir6: str = DEFAULT_DATA_DIRS["Flare23"],
    datadir7: str = DEFAULT_DATA_DIRS["HNSCC_convert_v1"],
) -> List[Dict]:
    jsonlist1 = os.path.join(list_dir, "btcv.json")
    jsonlist2 = os.path.join(list_dir, "dataset_TCIAcovid19_0.json")
    jsonlist3 = os.path.join(list_dir, "dataset_LUNA16_0.json")
    jsonlist4 = os.path.join(list_dir, "stoic21.json")
    jsonlist5 = os.path.join(list_dir, "Totalsegmentator_dataset.json")
    jsonlist6 = os.path.join(list_dir, "flare23.json")
    jsonlist7 = os.path.join(list_dir, "HNSCC.json")

    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    print(f"Dataset 1 BTCV: number of data: {len(datalist1)}")
    datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    print(f"Dataset 2 Covid 19: number of data: {len(datalist2)}")
    datalist3 = load_decathlon_datalist(jsonlist3, False, "training", base_dir=datadir3)
    print(f"Dataset 3 Luna: number of data: {len(datalist3)}")
    datalist4 = load_decathlon_datalist(jsonlist4, False, "training", base_dir=datadir4)
    print(f"Dataset 4 Stoic21: number of data: {len(datalist4)}")
    datalist5 = load_decathlon_datalist(jsonlist5, False, "training", base_dir=datadir5)
    print(f"Dataset 5 Totalsegmentator: number of data: {len(datalist5)}")
    datalist6 = load_decathlon_datalist(jsonlist6, False, "training", base_dir=datadir6)
    print(f"Dataset 6 Flare23: number of data: {len(datalist6)}")
    datalist7 = load_decathlon_datalist(jsonlist7, False, "training", base_dir=datadir7)
    print(f"Dataset 7 HNSCC: number of data: {len(datalist7)}")

    datalist = (
        _strip_to_image_only(datalist1)
        + datalist2
        + _strip_to_image_only(datalist3)
        + datalist4
        + datalist5
        + datalist6
        + datalist7
    )
    print(f"Dataset all training: number of data: {len(datalist)}")
    return datalist



def build_voco_train_datalist_1k(
    list_dir: str = DEFAULT_LIST_DIR,
    datadir1: str = DEFAULT_DATA_DIRS["BTCV"],
    datadir2: str = DEFAULT_DATA_DIRS["TCIAcovid19"],
    datadir3: str = DEFAULT_DATA_DIRS["Luna16"],
) -> List[Dict]:
    jsonlist1 = os.path.join(list_dir, "btcv.json")
    jsonlist2 = os.path.join(list_dir, "dataset_TCIAcovid19_0.json")
    jsonlist3 = os.path.join(list_dir, "dataset_LUNA16_0.json")

    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    print(f"Dataset 1 BTCV: number of data: {len(datalist1)}")
    datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    print(f"Dataset 2 Covid 19: number of data: {len(datalist2)}")
    datalist3 = load_decathlon_datalist(jsonlist3, False, "training", base_dir=datadir3)
    print(f"Dataset 3 Luna: number of data: {len(datalist3)}")

    datalist = _strip_to_image_only(datalist1) + datalist2 + _strip_to_image_only(datalist3)
    print(f"Dataset all training: number of data: {len(datalist)}")
    return datalist



def deduplicate_by_image_path(datalist: Sequence[Dict]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for item in datalist:
        image = str(item["image"])
        if image in seen:
            continue
        seen.add(image)
        out.append({"image": image})
    return out



def save_manifest(cache_dir: Path, datalist: Sequence[Dict], cfg: VolumeProbeLiteConfig) -> Path:
    manifest = {
        "num_images": len(datalist),
        "cache_dir": str(cache_dir),
        "config": cfg.__dict__,
        "images": [str(x["image"]) for x in datalist],
    }
    out_path = cache_dir / "volumeprobe_cache_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out_path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline build static VolumeProbe-Lite cache for VoCo training images."
    )

    parser.add_argument(
        "--dataset_variant",
        type=str,
        default="full",
        choices=["full", "1k"],
        help="Mirror get_loader(full) or get_loader_1k from the uploaded VoCo dataset.",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="Full",
        choices=["Full", "w/o A", "w/o I", "w/o D"],
        help="Static VolumeProbe-Lite ablation variant to build.",
    )

    parser.add_argument("--data_root", type=str, default=DEFAULT_VOCO_ROOT)
    parser.add_argument("--list_dir", type=str, default=None)

    parser.add_argument("--datadir1", type=str, default=None, help="BTCV root")
    parser.add_argument("--datadir2", type=str, default=None, help="TCIAcovid19 root")
    parser.add_argument("--datadir3", type=str, default=None, help="Luna16 root")
    parser.add_argument("--datadir4", type=str, default=None, help="stoic21 root")
    parser.add_argument("--datadir5", type=str, default=None, help="Totalsegmentator root")
    parser.add_argument("--datadir6", type=str, default=None, help="Flare23 root")
    parser.add_argument("--datadir7", type=str, default=None, help="HNSCC root")

    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--failure_mode", type=str, default="continue", choices=["continue", "raise"])

    parser.add_argument(
        "--failed_log_path",
        type=str,
        default=os.path.join(
            DEFAULT_VOCO_ROOT,
            "volumeprobe_cache",
            "failed_samples_static.jsonl",
        ),
    )

    parser.add_argument("--keep_duplicates", action="store_true", help="Do not de-duplicate identical image paths.")

    # Static VolumeProbe-Lite config.
    parser.add_argument("--axcodes", type=str, default="RAS")
    parser.add_argument("--pixdim", type=float, nargs=3, default=(1.5, 1.5, 1.5))
    parser.add_argument("--spacing_mode", type=str, default="bilinear")
    parser.add_argument("--a_min", type=float, default=-175.0)
    parser.add_argument("--a_max", type=float, default=250.0)
    parser.add_argument("--b_min", type=float, default=0.0)
    parser.add_argument("--b_max", type=float, default=1.0)
    parser.add_argument("--fg_threshold", type=float, default=0.30)

    parser.add_argument("--coarse_grid", type=int, nargs=3, default=(4, 4, 2))
    parser.add_argument("--top_coarse_bins", type=int, default=24)
    parser.add_argument("--fine_offset_frac", type=float, default=0.25)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--roi_size", type=int, nargs=3, default=(64, 64, 64))

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--grad_entropy_ratio", type=float, default=0.5)

    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min_center_distance_norm", type=float, default=None)
    parser.add_argument("--min_center_distance_factor", type=float, default=0.35)
    parser.add_argument("--entropy_bins", type=int, default=32)

    return parser.parse_args()



def make_config(args: argparse.Namespace) -> VolumeProbeLiteConfig:
    return VolumeProbeLiteConfig(
        variant=str(args.variant),
        axcodes=args.axcodes,
        pixdim=tuple(float(x) for x in args.pixdim),
        spacing_mode=args.spacing_mode,
        a_min=float(args.a_min),
        a_max=float(args.a_max),
        b_min=float(args.b_min),
        b_max=float(args.b_max),
        intensity_clip=True,
        fg_threshold=float(args.fg_threshold),
        coarse_grid=tuple(int(x) for x in args.coarse_grid),
        top_coarse_bins=int(args.top_coarse_bins),
        fine_offset_frac=float(args.fine_offset_frac),
        top_k=int(args.top_k),
        roi_size=tuple(int(x) for x in args.roi_size) if args.roi_size is not None else None,
        alpha=float(args.alpha),
        beta=float(args.beta),
        rho=float(args.rho),
        grad_entropy_ratio=float(args.grad_entropy_ratio),
        gamma=float(args.gamma),
        min_center_distance_norm=None if args.min_center_distance_norm is None else float(args.min_center_distance_norm),
        min_center_distance_factor=float(args.min_center_distance_factor),
        entropy_bins=int(args.entropy_bins),
    )



def _resolve_variant_name_for_path(variant: str) -> str:
    return str(variant).replace("/", "").replace(" ", "_")



def _resolve_paths(args: argparse.Namespace) -> Dict[str, str]:
    root = args.data_root
    list_dir = args.list_dir or os.path.join(root, "jsons")
    variant_name = _resolve_variant_name_for_path(args.variant)
    paths = {
        "list_dir": list_dir,
        "datadir1": args.datadir1 or os.path.join(root, "BTCV"),
        "datadir2": args.datadir2 or os.path.join(root, "TCIAcovid19"),
        "datadir3": args.datadir3 or os.path.join(root, "Luna16"),
        "datadir4": args.datadir4 or os.path.join(root, "stoic21"),
        "datadir5": args.datadir5 or os.path.join(root, "Totalsegmentator_dataset"),
        "datadir6": args.datadir6 or os.path.join(root, "Flare23"),
        "datadir7": args.datadir7 or os.path.join(root, "HNSCC_convert_v1"),
        "cache_dir": args.cache_dir or os.path.join(
            root,
            "volumeprobe_cache",
            f"volumeprobe_cache_static_{args.dataset_variant}_{variant_name}_2026.04.24",
        ),
        "failed_log_path": args.failed_log_path,
    }
    return paths



def build_datalist(args: argparse.Namespace) -> List[Dict]:
    p = _resolve_paths(args)

    print("Resolved paths:")
    print(f"  data_root : {args.data_root}")
    print(f"  list_dir  : {p['list_dir']}")
    print(f"  variant   : {args.variant}")
    print(f"  BTCV      : {p['datadir1']}")
    print(f"  TCIA      : {p['datadir2']}")
    print(f"  Luna16    : {p['datadir3']}")
    if args.dataset_variant == "full":
        print(f"  stoic21   : {p['datadir4']}")
        print(f"  TotalSeg  : {p['datadir5']}")
        print(f"  Flare23   : {p['datadir6']}")
        print(f"  HNSCC     : {p['datadir7']}")
    print(f"  cache_dir : {p['cache_dir']}")
    if p.get("failed_log_path") is not None:
        print(f"  fail_log  : {p['failed_log_path']}")

    if args.dataset_variant == "1k":
        datalist = build_voco_train_datalist_1k(
            list_dir=p["list_dir"],
            datadir1=p["datadir1"],
            datadir2=p["datadir2"],
            datadir3=p["datadir3"],
        )
    else:
        datalist = build_voco_train_datalist_full(
            list_dir=p["list_dir"],
            datadir1=p["datadir1"],
            datadir2=p["datadir2"],
            datadir3=p["datadir3"],
            datadir4=p["datadir4"],
            datadir5=p["datadir5"],
            datadir6=p["datadir6"],
            datadir7=p["datadir7"],
        )

    if not args.keep_duplicates:
        n0 = len(datalist)
        datalist = deduplicate_by_image_path(datalist)
        if len(datalist) != n0:
            print(f"De-duplicated datalist: {n0} -> {len(datalist)}")
    return datalist



def main() -> None:
    args = parse_args()
    cfg = make_config(args)
    paths = _resolve_paths(args)

    cache_dir = Path(paths["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    datalist = build_datalist(args)
    if len(datalist) == 0:
        raise RuntimeError("No training images found for the selected VoCo datalist.")

    builder = VolumeProbeLiteBuilder(cfg)
    manifest_path = save_manifest(cache_dir, datalist, cfg)
    print(f"Saved manifest to: {manifest_path}")

    print("\nStart offline static VolumeProbe-Lite cache building...")
    builder.build_and_cache_all(
        datalist=datalist,
        cache_dir=cache_dir,
        overwrite=bool(args.overwrite),
        verbose=True,
        failed_log_path=paths["failed_log_path"],
        failure_mode=str(args.failure_mode),
    )
    print("Done.")


if __name__ == "__main__":
    main()
