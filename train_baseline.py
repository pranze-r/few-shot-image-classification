"""
Train the "no meta-learning" baseline: an ordinary CNN classifier trained
the standard supervised way to classify the base (train) classes.

This is NOT meta-learning -- it never sees episodes, just a normal
classification dataset. We use it purely as a comparison point: at test
time (see evaluate.py) we take its encoder and fine-tune a fresh linear
head on each new episode's tiny support set, then measure query accuracy.
The gap between this baseline and Prototypical Networks is the "story" of
the project: it shows what meta-learning (training explicitly for fast
adaptation) buys you over a model that was never trained with adaptation
in mind.

Run:
    python train_baseline.py --epochs 30
"""

import argparse
import random
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

from data import build_splits, FewShotImageBank
from models import ConvEncoder, BaselineClassifier


class FlatClassificationDataset(Dataset):
    """
    Wraps a FewShotImageBank as a normal (image, label) classification
    dataset -- i.e. NOT episodic -- for standard supervised training.
    Labels are remapped to 0..(num_classes-1).
    """

    def __init__(self, image_bank: FewShotImageBank, images_per_class=400):
        self.samples = []  # list of (image_tensor, remapped_label)
        for new_label, cls in enumerate(image_bank.class_indices):
            imgs = image_bank.sample_class_images(cls, images_per_class)
            for img in imgs:
                self.samples.append((img, new_label))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--images_per_class", type=int, default=400)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_path", type=str, default="baseline_encoder.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_idx, val_idx, test_idx, base_ds, extra_ds = build_splits(root=args.data_root)
    train_bank = FewShotImageBank(base_ds, extra_ds, train_idx, augment=True)

    dataset = FlatClassificationDataset(train_bank, images_per_class=args.images_per_class)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"Baseline training set size: {len(dataset)} images over {len(train_idx)} classes")

    encoder = ConvEncoder().to(device)
    model = BaselineClassifier(n_base_classes=len(train_idx), encoder=encoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        total_loss, total_correct, total_n = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_n += imgs.size(0)

        print(f"[epoch {epoch}/{args.epochs}] loss={total_loss/total_n:.3f} "
              f"acc={total_correct/total_n:.3f}")

    torch.save(model.encoder.state_dict(), args.save_path)
    print(f"Saved baseline encoder to {args.save_path}")


if __name__ == "__main__":
    main()
