from pathlib import Path
from zipfile import ZipFile
import shutil
import numpy as np

RAW = Path("data/raw")
CACHE = Path("data/cache/deepbeat_memmap")
CACHE.mkdir(parents=True, exist_ok=True)

for split in ["train", "validate", "test"]:

    npz_path = RAW / f"{split}.npz"
    out_dir = CACHE / split
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "signal.npy"

    print()
    print("=" * 100)
    print(split.upper())
    print("=" * 100)

    with ZipFile(npz_path) as zf:

        candidates = [
            x for x in zf.infolist()
            if x.filename.endswith("signal.npy")
        ]

        if len(candidates) != 1:
            raise RuntimeError(
                f"{split}: expected one signal.npy, "
                f"found {[x.filename for x in candidates]}"
            )

        info = candidates[0]

        print(
            "Uncompressed:",
            f"{info.file_size / 1024**3:.3f} GB"
        )

        if (
            out_path.exists()
            and out_path.stat().st_size == info.file_size
        ):
            print("Already extracted:", out_path)

        else:
            free = shutil.disk_usage(
                out_dir
            ).free

            required = int(
                info.file_size * 1.10
            )

            print(
                "Free disk:",
                f"{free / 1024**3:.3f} GB"
            )

            if free < required:
                raise RuntimeError(
                    f"Not enough disk for {split}: "
                    f"need ~{required / 1024**3:.2f} GB"
                )

            tmp = out_path.with_suffix(
                ".npy.tmp"
            )

            if tmp.exists():
                tmp.unlink()

            print(
                "Extracting",
                info.filename,
                "->",
                out_path
            )

            with zf.open(info) as src, \
                    open(tmp, "wb") as dst:

                shutil.copyfileobj(
                    src,
                    dst,
                    length=16 * 1024 * 1024
                )

            tmp.rename(out_path)

    x = np.load(
        out_path,
        mmap_mode="r"
    )

    print("shape :", x.shape)
    print("dtype :", x.dtype)
    print("nbytes:", f"{x.nbytes / 1024**3:.3f} GB")
    print("mmap  :", type(x))

print()
print("DONE")
