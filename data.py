"""
Data pipeline for few-shot classification on CIFAR-100 (CIFAR-FS style).

Meta-learning terminology used below:
  - "N-way K-shot episode": a mini classification task with N classes and
    K labeled examples ("support set") per class, plus some unlabeled
    "query" examples per class that the model must classify.
  - "Base / train classes": classes used during meta-training.
  - "Test classes": classes held out completely during training, used only
    to evaluate few-shot generalization at the end. This train/test split
    is BY CLASS, not by image -- the model must never see test classes
    until final evaluation.

This file:
  1. Downloads CIFAR-100 via torchvision (cached after first run).
  2. Splits the 100 classes into 64 train / 16 val / 20 test classes using
     a fixed random seed (reproducible every run, and easy to swap for the
     official CIFAR-FS split later if you want numbers comparable to papers
     -- see the README for a link to the official split).
  3. Provides an EpisodicSampler that yields N-way K-shot episodes.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

SPLIT_SEED = 42  # fixed so every run (and every teammate/grader) gets the same split


class _ArrayDataset:
    """
    Minimal stand-in for torchvision's CIFAR100 object: exposes .data
    (N,32,32,3 uint8 array), .targets (list of ints), and .classes (list of
    100 name strings), which is all the rest of this file needs. Used to
    wrap whichever download source actually succeeds (see build_splits).
    """

    def __init__(self, data, targets, classes):
        self.data = data
        self.targets = targets
        self.classes = classes


def _load_via_huggingface():
    """
    Loads CIFAR-100 from Hugging Face's hosted copy (uoft-cs/cifar100).
    This is usually MUCH faster than torchvision's default download source
    (the original University of Toronto mirror), which can be extremely
    slow/throttled from shared environments like Colab.
    """
    from datasets import load_dataset

    ds = load_dataset("uoft-cs/cifar100")
    classes = ds["train"].features["fine_label"].names  # 100 class names, index-aligned

    def to_array_dataset(split):
        imgs = np.stack([np.array(row["img"].convert("RGB")) for row in split])
        targets = list(split["fine_label"])
        return imgs, targets

    train_imgs, train_targets = to_array_dataset(ds["train"])
    test_imgs, test_targets = to_array_dataset(ds["test"])

    base = _ArrayDataset(train_imgs, train_targets, classes)
    extra = _ArrayDataset(test_imgs, test_targets, classes)
    return base, extra


def _load_via_torchvision(root, download):
    from torchvision.datasets import CIFAR100

    base = CIFAR100(root=root, train=True, download=download)
    extra = CIFAR100(root=root, train=False, download=download)
    return base, extra


def build_splits(root="./data", download=True, n_train=64, n_val=16, n_test=20,
                  source="huggingface"):
    """
    Downloads CIFAR-100 and returns:
        train_idx, val_idx, test_idx : lists of class indices (0-99)
        base_ds, extra_ds            : train/test dataset objects, each exposing
                                        .data / .targets / .classes (CIFAR-FS style:
                                        pools train+test, then re-splits by class)

    source: "huggingface" (default, usually much faster to download -- see
            _load_via_huggingface) or "torchvision" (the original, sometimes-slow
            source). Falls back to torchvision automatically if the
            `datasets` library isn't installed or the HF download fails.
    """
    if source == "huggingface":
        try:
            base, extra = _load_via_huggingface()
        except Exception as e:
            print(f"Hugging Face download failed ({e}); falling back to torchvision source.")
            base, extra = _load_via_torchvision(root, download)
    else:
        base, extra = _load_via_torchvision(root, download)

    all_idx = list(range(len(base.classes)))  # 100 classes
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(all_idx)

    train_idx = sorted(all_idx[:n_train])
    val_idx = sorted(all_idx[n_train:n_train + n_val])
    test_idx = sorted(all_idx[n_train + n_val:n_train + n_val + n_test])
    return train_idx, val_idx, test_idx, base, extra


class FewShotImageBank(Dataset):
    """
    Holds all images for a given set of class indices, grouped by class,
    so the episodic sampler can quickly draw K+Q images per class without
    re-scanning the whole dataset each time.
    """

    def __init__(self, base_ds, extra_ds, class_indices, image_size=32, augment=False,
                 class_names=None):
        self.class_indices = list(class_indices)
        self.class_to_images = {c: [] for c in self.class_indices}
        # Full list of 100 human-readable class names (e.g. "tiger", "goblet"),
        # aligned by index with the original dataset. Lets us show real names
        # instead of meaningless per-episode numbers in plots/reports.
        self.class_names = class_names

        norm = transforms.Normalize(
            mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )
        tfms = [transforms.Resize((image_size, image_size))]
        if augment:
            tfms += [
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        tfms += [transforms.ToTensor(), norm]
        self.transform = transforms.Compose(tfms)

        wanted = set(self.class_indices)
        for ds in (base_ds, extra_ds):
            for img, label in zip(ds.data, ds.targets):
                if label in wanted:
                    self.class_to_images[label].append(img)

        empty = [c for c, v in self.class_to_images.items() if len(v) == 0]
        if empty:
            raise ValueError(f"Classes with no images found: {empty}")

    def __len__(self):
        return sum(len(v) for v in self.class_to_images.values())

    def name_for(self, class_idx):
        """Human-readable name for a class index, e.g. name_for(34) -> 'tiger'."""
        if self.class_names is not None:
            return self.class_names[class_idx]
        return str(class_idx)

    def sample_class_images(self, class_idx, n):
        """Randomly sample n images from one class, transformed to tensors. Shape: (n, 3, H, W)."""
        imgs = self.class_to_images[class_idx]
        chosen = random.sample(imgs, n) if len(imgs) >= n else random.choices(imgs, k=n)
        out = [self.transform(transforms.functional.to_pil_image(arr)) for arr in chosen]
        return torch.stack(out)


class EpisodicSampler:
    """
    Generates N-way K-shot episodes from a FewShotImageBank.

    sample_episode() returns:
        support_x: (n_way * k_shot, 3, H, W)
        support_y: (n_way * k_shot,)   -- labels re-mapped to 0..n_way-1 for this episode
        query_x:   (n_way * q_query, 3, H, W)
        query_y:   (n_way * q_query,)

    Note: the 0..n_way-1 labels are just "class #1 in THIS episode", "class #2
    in THIS episode", etc -- they are re-picked fresh every call and carry no
    meaning across episodes. After calling sample_episode(), use
    episode_class_names() to find out what real-world categories (e.g.
    "tiger", "goblet") those numbers referred to in that specific call --
    this is what you want for any plot or figure a reader will look at.
    """

    def __init__(self, image_bank: FewShotImageBank, n_way=5, k_shot=5, q_query=15):
        if n_way > len(image_bank.class_indices):
            raise ValueError(
                f"n_way={n_way} but only {len(image_bank.class_indices)} classes available"
            )
        self.bank = image_bank
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self._last_episode_classes = None  # original class indices from the most recent episode

    def sample_episode(self):
        episode_classes = random.sample(self.bank.class_indices, self.n_way)
        self._last_episode_classes = episode_classes
        support_x, support_y, query_x, query_y = [], [], [], []

        for new_label, cls in enumerate(episode_classes):
            imgs = self.bank.sample_class_images(cls, self.k_shot + self.q_query)
            support_x.append(imgs[: self.k_shot])
            query_x.append(imgs[self.k_shot:])
            support_y += [new_label] * self.k_shot
            query_y += [new_label] * self.q_query

        return (
            torch.cat(support_x, dim=0),
            torch.tensor(support_y, dtype=torch.long),
            torch.cat(query_x, dim=0),
            torch.tensor(query_y, dtype=torch.long),
        )

    def episode_class_names(self):
        """
        Real-world names for the 0..n_way-1 labels used in the most recent
        sample_episode() call, e.g. ["tiger", "goblet", "bicycle", ...] where
        index 0 in the list is what label 0 meant in that episode.
        """
        if self._last_episode_classes is None:
            raise RuntimeError("Call sample_episode() before episode_class_names().")
        return [self.bank.name_for(c) for c in self._last_episode_classes]
