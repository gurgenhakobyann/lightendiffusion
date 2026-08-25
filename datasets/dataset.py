import os
import random
import glob
import torch
import torch.utils.data
from PIL import Image
from datasets.data_augment import (
    PairCompose,
    PairToTensor,
    PairRandomHorizontalFilp,
    PairRandomVerticalFlip,
    PairRandomCrop,
)


class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        train_type = getattr(self.config.data, "train_dataset", "unpaired")
        val_type = getattr(self.config.data, "val_dataset", "LOLv1")
        data_dir = self.config.data.data_dir
        patch_size = getattr(self.config.data, "patch_size", 256)

        # ----------------- Training Dataset -----------------
        if train_type in ["unpaired", "BrighteningTrain"]:
            # Dynamic On-the-Fly Random Pairing (~180k pairs from 1000 low x 1000 high)
            low_dir = os.path.join(data_dir, "BrighteningTrain", "low")
            high_dir = os.path.join(data_dir, "BrighteningTrain", "high")
            if not os.path.isdir(low_dir) or not os.path.isdir(high_dir):
                # Fallback: look one level up (in case data_dir already is BrighteningTrain)
                low_dir = os.path.join(data_dir, "low")
                high_dir = os.path.join(data_dir, "high")

            train_dataset = UnpairedDynamicDataset(
                low_dir=low_dir,
                high_dir=high_dir,
                patch_size=patch_size,
                epoch_length=getattr(self.config.data, "epoch_length", 180000),
            )
        else:
            # Fallback to text filelist if available
            train_list_path = os.path.join(data_dir, f"{train_type}_train.txt")
            if os.path.exists(train_list_path):
                train_dataset = AllWeatherDataset(
                    data_dir, patch_size=patch_size, filelist=f"{train_type}_train.txt", train=True
                )
            else:
                train_dataset = None

        # ----------------- Validation / Test Dataset -----------------
        val_filelist = f"{val_type}_val.txt"
        val_list_path = os.path.join(data_dir, val_filelist)

        if os.path.exists(val_list_path):
            val_dataset = AllWeatherDataset(
                data_dir, patch_size=patch_size, filelist=val_filelist, train=False
            )
        elif val_type.upper() in ["DICM", "NPE", "VV", "LIME", "MEF", "FUSION"]:
            benchmark_dir = os.path.join(data_dir, "Test", val_type.upper())
            if not os.path.isdir(benchmark_dir):
                benchmark_dir = os.path.join(data_dir, val_type.upper())
            val_dataset = UnpairedBenchmarkDataset(benchmark_dir)
        elif "LSRW" in val_type:
            subset = "Huawei" if "Huawei" in val_type else "Nikon"
            lsrw_dir = os.path.join(data_dir, "LSRW", "Eval", "Eval", subset)
            if not os.path.isdir(lsrw_dir):
                lsrw_dir = os.path.join(data_dir, subset)
            val_dataset = LSRWDataset(lsrw_dir)
        else:
            # Default: LOLv1 validation
            val_dataset = LOLv1ValDataset(data_dir)

        train_loader = (
            torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                num_workers=self.config.data.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            if train_dataset is not None
            else None
        )

        val_loader = (
            torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config.sampling.batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                pin_memory=True,
            )
            if val_dataset is not None
            else None
        )

        return train_loader, val_loader


# =====================================================================
# 1. Dynamic Unpaired Random Pairing Dataset (~180k pairs per epoch)
# =====================================================================
class UnpairedDynamicDataset(torch.utils.data.Dataset):
    """
    Dynamic cross-pairing between N low-light and N normal-light images.
    Each iteration randomly pairs one low-light crop with one random normal-light crop,
    producing ~N^2 unique combinations. epoch_length controls iterations per epoch.
    """
    def __init__(self, low_dir, high_dir, patch_size=256, epoch_length=180000):
        super().__init__()
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.patch_size = patch_size
        self.epoch_length = epoch_length

        self.low_files = sorted(
            glob.glob(os.path.join(low_dir, "*.png"))
            + glob.glob(os.path.join(low_dir, "*.jpg"))
            + glob.glob(os.path.join(low_dir, "*.bmp"))
        )
        self.high_files = sorted(
            glob.glob(os.path.join(high_dir, "*.png"))
            + glob.glob(os.path.join(high_dir, "*.jpg"))
            + glob.glob(os.path.join(high_dir, "*.bmp"))
        )

        if not self.low_files:
            raise FileNotFoundError(f"No images found in low_dir: {low_dir}")
        if not self.high_files:
            raise FileNotFoundError(f"No images found in high_dir: {high_dir}")

        print(f"[UnpairedDynamicDataset] Loaded {len(self.low_files)} low-light and "
              f"{len(self.high_files)} normal-light images. "
              f"Epoch length configured to {epoch_length} iterations (random cross-pairing).")

        self.transforms = PairCompose([
            PairRandomCrop(self.patch_size, pad_if_needed=True),
            PairRandomHorizontalFilp(),
            PairRandomVerticalFlip(),
            PairToTensor(),
        ])

    def __getitem__(self, index):
        low_path = self.low_files[index % len(self.low_files)]
        high_path = random.choice(self.high_files)

        img_id = os.path.basename(low_path)
        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")

        low_tensor, high_tensor = self.transforms(low_img, high_img)
        return torch.cat([low_tensor, high_tensor], dim=0), img_id

    def __len__(self):
        return self.epoch_length


# =====================================================================
# 2. Paired Filelist Dataset (LOLv1 train, AllWeather, etc.)
# =====================================================================
class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, filelist=None, train=True):
        super().__init__()
        self.dir = dir
        self.file_list = filelist
        self.train_list = os.path.join(dir, self.file_list)
        with open(self.train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents if i.strip()]

        self.input_names = input_names
        self.patch_size = patch_size

        if train:
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size, pad_if_needed=True),
                PairRandomHorizontalFilp(),
                PairToTensor(),
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor(),
            ])

    def __getitem__(self, index):
        input_name = self.input_names[index]
        low_img_name, high_img_name = input_name.split(" ")[0], input_name.split(" ")[1]

        img_id = os.path.basename(low_img_name)
        low_img = Image.open(low_img_name).convert("RGB")
        high_img = Image.open(high_img_name).convert("RGB")

        low_img, high_img = self.transforms(low_img, high_img)
        return torch.cat([low_img, high_img], dim=0), img_id

    def __len__(self):
        return len(self.input_names)


# =====================================================================
# 3. LSRW Dataset Loader (Huawei & Nikon evaluation)
# =====================================================================
class LSRWDataset(torch.utils.data.Dataset):
    def __init__(self, lsrw_dir):
        super().__init__()
        self.low_dir = os.path.join(lsrw_dir, "low")
        self.high_dir = os.path.join(lsrw_dir, "high")

        self.low_files = sorted(
            glob.glob(os.path.join(self.low_dir, "*.jpg"))
            + glob.glob(os.path.join(self.low_dir, "*.png"))
        )
        self.transforms = PairCompose([PairToTensor()])

    def __getitem__(self, index):
        low_path = self.low_files[index]
        fname = os.path.basename(low_path)
        high_path = os.path.join(self.high_dir, fname)

        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB") if os.path.exists(high_path) else low_img

        low_t, high_t = self.transforms(low_img, high_img)
        return torch.cat([low_t, high_t], dim=0), fname

    def __len__(self):
        return len(self.low_files)


# =====================================================================
# 4. Unpaired Real-World Benchmarks (DICM, NPE, VV, LIME, MEF)
# =====================================================================
class UnpairedBenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        super().__init__()
        self.img_dir = img_dir
        self.files = sorted(
            glob.glob(os.path.join(img_dir, "*.png"))
            + glob.glob(os.path.join(img_dir, "*.jpg"))
            + glob.glob(os.path.join(img_dir, "*.bmp"))
            + glob.glob(os.path.join(img_dir, "*.JPG"))
        )
        self.transforms = PairCompose([PairToTensor()])

    def __getitem__(self, index):
        path = self.files[index]
        fname = os.path.basename(path)
        img = Image.open(path).convert("RGB")
        img_t, _ = self.transforms(img, img)
        return torch.cat([img_t, img_t], dim=0), fname

    def __len__(self):
        return len(self.files)


# =====================================================================
# 5. LOL-v1 Validation Dataset
# =====================================================================
class LOLv1ValDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        eval_names = [
            '1.png', '22.png', '23.png', '55.png', '79.png', '111.png', '146.png',
            '179.png', '493.png', '547.png', '665.png', '669.png', '748.png',
            '778.png', '780.png'
        ]
        self.pairs = []
        # Try multiple candidate roots to handle different directory layouts on cluster
        candidate_roots = [
            data_dir,
            os.path.join(data_dir, "LOLdataset"),
            os.path.join(data_dir, "lol"),
        ]
        for f in eval_names:
            found = False
            for root in candidate_roots:
                for split in ["our485", "eval15"]:
                    low_p = os.path.join(root, split, "low", f)
                    high_p = os.path.join(root, split, "high", f)
                    if os.path.exists(low_p):
                        self.pairs.append((low_p, high_p))
                        found = True
                        break
                if found:
                    break
        if not self.pairs:
            raise FileNotFoundError(
                f"LOLv1ValDataset: Could not find any eval images in {data_dir}. "
                f"Checked: {candidate_roots}"
            )

        self.transforms = PairCompose([PairToTensor()])

    def __getitem__(self, index):
        low_p, high_p = self.pairs[index]
        fname = os.path.basename(low_p)
        low_img = Image.open(low_p).convert("RGB")
        high_img = Image.open(high_p).convert("RGB") if os.path.exists(high_p) else low_img
        low_t, high_t = self.transforms(low_img, high_img)
        return torch.cat([low_t, high_t], dim=0), fname

    def __len__(self):
        return len(self.pairs)
