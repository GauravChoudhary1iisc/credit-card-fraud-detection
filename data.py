"""Download the Kaggle Credit Card Fraud Detection dataset (creditcard.csv)."""
import shutil
from pathlib import Path

import kagglehub

DATASET = "mlg-ulb/creditcardfraud"
OUTPUT = Path(__file__).parent / "creditcard.csv"


def download(output: Path = OUTPUT) -> Path:
    if output.exists():
        print(f"Already exists: {output}")
        return output

    dataset_dir = Path(kagglehub.dataset_download(DATASET))
    shutil.copy(dataset_dir / "creditcard.csv", output)
    print(f"Saved to {output}")
    return output


if __name__ == "__main__":
    download()
