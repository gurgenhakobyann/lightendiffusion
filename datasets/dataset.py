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
            # Approach A: Dynamic On-the-Fly Random Pairing (~180k pairs from 1000 low x 1000 high)
            low_dir = os.path.join(data_dir, "BrighteningTrain", "low")
            high_dir = os.path.join(data_dir, "BrighteningTrain", "high")
            if not os.path.isdir(low_dir) or not os.path.isdir(high_dir):
                low_dir = os.path.join(data_dir, "low")
                high_dir = os.path.join(data_dir, "high")

            train_dataset = UnpairedDynamicDataset(
                low_dir=low_dir,
                high_dir=high_dir,
                patch_size=patch_size,
                epoch_length=getattr(self.config.data, "epoch_length", 180000),
            )
        elif train_type in ["sice", "SICE"]:
            # Stage-1 Multi-exposure scene pairs for CTDN
            train_dataset = SICEDataset(
                root_dir=data_dir,
                patch_size=patch_size,
                epoch_length=getattr(self.config.data, "epoch_length", 50000),
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
            # Unpaired real-world test benchmark
            benchmark_dir = os.path.join(data_dir, "Test", val_type.upper())
            if not os.path.isdir(benchmark_dir):
                benchmark_dir = os.path.join(data_dir, val_type.upper())
            val_dataset = UnpairedBenchmarkDataset(benchmark_dir)
        elif "LSRW" in val_type:
            # LSRW Huawei or Nikon test set
            subset = "Huawei" if "Huawei" in val_type else "Nikon"
            lsrw_dir = os.path.join(data_dir, "LSRW", "Eval", "Eval", subset)
            if not os.path.isdir(lsrw_dir):
                lsrw_dir = os.path.join(data_dir, subset)
            val_dataset = LSRWDataset(lsrw_dir)
        else:
            # Default LOLv1 validation loader
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
# 1. Approach A: Dynamic Unpaired Random Pairing Dataset (~180k pairs)
# =====================================================================
class UnpairedDynamicDataset(torch.utils.data.Dataset):
    """
    Implements dynamic cross-pairing between 1,000 low-light and 1,000 normal-light images.
    In each iteration, randomly pairs a low-light crop with a random normal-light crop,
    generating ~180,000 unique unpaired combinations per epoch.
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

        if not self.low_files or not self.high_files:
            raise FileNotFoundError(
                f"Could not find training images in {low_dir} ({len(self.low_files)} found) or {high_dir} ({len(self.high_files)} found)"
            )

        print(
            f"[UnpairedDynamicDataset] Loaded {len(self.low_files)} low-light and {len(self.high_files)} normal-light images. "
            f"Epoch length configured to {self.epoch_length} iterations (random cross-pairing)."
        )

        self.transforms = PairCompose([
            PairRandomCrop(self.patch_size, pad_if_needed=True),
            PairRandomHorizontalFilp(),
            PairRandomVerticalFlip(),
            PairToTensor(),
        ])

    def __getitem__(self, index):
        # Pick low-light image by index, and randomly sample a normal-light image
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
# 2. Stage-1 Multi-Exposure SICE Dataset (Decomposition Network Training)
# =====================================================================
class SICEDataset(torch.utils.data.Dataset):
    """
    SICE Multi-Exposure dataset loader for Stage-1 CTDN training.
    Samples two different exposure images (I1, I2) from the same scene.
    """
    def __init__(self, root_dir, patch_size=256, epoch_length=50000):
        super().__init__()
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.epoch_length = epoch_length

        scene_dirs = []
        for root, dirs, files in os.walk(root_dir):
            img_files = [f for f in files if f.lower().endswith((".jpg", ".png", ".bmp", ".jpeg"))]
            if len(img_files) >= 2:
                scene_dirs.append((root, img_files))

        self.scene_dirs = scene_dirs
        if not self.scene_dirs:
            raise FileNotFoundError(f"No multi-exposure scene folders with >= 2 images found in {root_dir}")

        print(f"[SICEDataset] Found {len(self.scene_dirs)} multi-exposure scenes.")

        self.transforms = PairCompose([
            PairRandomCrop(self.patch_size, pad_if_needed=True),
            PairRandomHorizontalFilp(),
            PairRandomVerticalFlip(),
            PairToTensor(),
        ])

    def __getitem__(self, index):
        scene_path, img_files = self.scene_dirs[index % len(self.scene_dirs)]
        f1, f2 = random.sample(img_files, 2)

        img1 = Image.open(os.path.join(scene_path, f1)).convert("RGB")
        img2 = Image.open(os.path.join(scene_path, f2)).convert("RGB")

        t1, t2 = self.transforms(img1, img2)
        scene_id = os.path.basename(scene_path)
        return torch.cat([t1, t2], dim=0), scene_id

    def __len__(self):
        return self.epoch_length


# =====================================================================
# 3. Paired Filelist Dataset (LOLv1, AllWeather, etc.)
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
# 4. LSRW Dataset Loader (Huawei & Nikon evaluation)
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
# 5. Unpaired Real-World Benchmarks (DICM, NPE, VV, LIME, MEF)
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
# 6. LOL-v1 Validation Dataset Fallback
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
        for f in eval_names:
            low_p = os.path.join(data_dir, "our485", "low", f)
            high_p = os.path.join(data_dir, "our485", "high", f)
            if not os.path.exists(low_p):
                low_p = os.path.join(data_dir, "eval15", "low", f)
                high_p = os.path.join(data_dir, "eval15", "high", f)
            if os.path.exists(low_p):
                self.pairs.append((low_p, high_p))

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
