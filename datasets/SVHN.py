import os
import tempfile

import torchvision
from tqdm.auto import tqdm

CLASSES = [i for i in range(1, 11)]


def main():
    for split in ["train", "test"]:
        out_dir = f"SVHN_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        print("downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.SVHN(
                root=tmp_dir, download=True, split=split
            )

        print("dumping images...")
        os.mkdir(out_dir)
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            filename = os.path.join(out_dir, f"{CLASSES[label]}_{i:05d}.png")
            image.save(filename)


if __name__ == "__main__":
    main()
