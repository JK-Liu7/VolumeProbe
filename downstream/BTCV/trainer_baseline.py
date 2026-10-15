from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.trainer import run_training, train_epoch, val_epoch
