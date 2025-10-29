import os
from pathlib import Path
import torch

TRAIN_DATA = Path(os.getenv("TRAIN_DATA", "data/train/train_data.pt"))
TEST_DATA = Path(os.getenv("TEST_DATA", "data/test/test_data_incomplete.pt"))

DEVICE = os.getenv("DEVICE", "mps" if torch.mps.is_available() else "cpu")
