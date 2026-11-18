"""Downloader and unpacker for PAMAP2 dataset with fallback verification."""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PAMAP2_SOURCE_URL, RAW_DATA_DIR


def download_pamap2(destination: Path = RAW_DATA_DIR, force: bool = False) -> Path:
    """Download and extract PAMAP2 dataset if not already present.

    Args:
        destination: Target folder for raw PAMAP2 dataset.
        force: Whether to re-download if files already exist.

    Returns:
        Path to extracted dataset folder.
    """
    destination.mkdir(parents=True, exist_ok=True)
    zip_path = destination / "PAMAP2_Dataset.zip"
    extracted_protocol_dir = destination / "PAMAP2_Dataset" / "Protocol"

    if (
        not force
        and extracted_protocol_dir.exists()
        and any(extracted_protocol_dir.glob("subject1*.dat"))
    ):
        print(f"[INFO] PAMAP2 dataset already present at {extracted_protocol_dir}")
        return extracted_protocol_dir

    print(f"[INFO] Attempting download from: {PAMAP2_SOURCE_URL}")
    print("[INFO] Target archive size: ~700 MB. Please be patient...")

    try:

        def _reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(
                    f"\r[DOWNLOADING] {percent}% ({count * block_size // (1024 * 1024)} MB)"
                )
                sys.stdout.flush()

        urllib.request.urlretrieve(PAMAP2_SOURCE_URL, zip_path, reporthook=_reporthook)
        print("\n[INFO] Download completed. Extracting archive...")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(destination)

        print(f"[SUCCESS] Extracted PAMAP2 files to {destination}")
        return extracted_protocol_dir

    except Exception as e:
        print(f"\n[WARNING] Automated download failed or network blocked: {e}")
        print("\n" + "=" * 70)
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("1. Download PAMAP2_Dataset.zip directly from UCI Machine Learning Repository:")
        print(f"   {PAMAP2_SOURCE_URL}")
        print("2. Extract the archive into:")
        print(f"   {destination.resolve()}")
        print("3. Ensure files like 'subject101.dat' exist in:")
        print(f"   {(destination / 'PAMAP2_Dataset' / 'Protocol').resolve()}")
        print("=" * 70 + "\n")
        return extracted_protocol_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PAMAP2 dataset")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()
    download_pamap2(force=args.force)
