"""
Visualizations for the report/presentation:

1. t-SNE plot of the query embeddings for one test episode, colored by
   true class, with prototype locations marked as stars. This is the
   classic "does the embedding space cluster nicely by class" figure used
   in almost every few-shot learning paper.

2. A grid of example query images with their true label, predicted label,
   and whether the prediction was correct -- good for a "qualitative
   results" slide.

Run:
    python visualize.py --protonet_path protonet_encoder.pt
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from data import build_splits, FewShotImageBank, EpisodicSampler
from models import ConvEncoder


def unnormalize(img_tensor):
    mean = torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1)
    std = torch.tensor([0.2673, 0.2564, 0.2762]).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)


def plot_tsne(encoder, sx, sy, qx, qy, n_way, device, class_names,
              save_path="tsne_embeddings.png"):
    encoder.eval()
    with torch.no_grad():
        es = encoder(sx.to(device)).cpu().numpy()
        eq = encoder(qx.to(device)).cpu().numpy()

    prototypes = np.stack([es[sy.numpy() == c].mean(axis=0) for c in range(n_way)])

    combined = np.concatenate([eq, prototypes], axis=0)
    tsne = TSNE(n_components=2, perplexity=min(30, len(combined) - 1), random_state=0)
    reduced = tsne.fit_transform(combined)

    q_reduced = reduced[: len(eq)]
    proto_reduced = reduced[len(eq):]

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")
    for c in range(n_way):
        mask = qy.numpy() == c
        ax.scatter(q_reduced[mask, 0], q_reduced[mask, 1], color=cmap(c),
                   label=class_names[c], alpha=0.6, s=25)
        ax.scatter(proto_reduced[c, 0], proto_reduced[c, 1], color=cmap(c),
                   marker="*", s=350, edgecolor="black", linewidth=1.2)

    ax.set_title("Each dot = one test image. Same color = model's guess puts them in\n"
                  "the same category. Stars = the category's \"average\" position.",
                  fontsize=11)
    ax.set_xlabel("(axes have no real-world meaning here -- only distance between points matters)",
                  fontsize=9, style="italic")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(title="Category (star = prototype)", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


def plot_example_predictions(encoder, sx, sy, qx, qy, n_way, device, class_names,
                              n_examples=8, save_path="example_predictions.png"):
    from models import prototypical_loss

    encoder.eval()
    with torch.no_grad():
        es = encoder(sx.to(device))
        eq = encoder(qx.to(device))
        prototypes = torch.stack([es[sy == c].mean(dim=0) for c in range(n_way)])
        dists = torch.cdist(eq, prototypes, p=2) ** 2
        preds = (-dists).argmax(dim=1).cpu()

    idxs = np.random.choice(len(qx), size=min(n_examples, len(qx)), replace=False)
    cols = 4
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.7, rows * 3.6))
    axes = np.array(axes).reshape(rows, cols)

    for i, idx in enumerate(idxs):
        img = unnormalize(qx[idx]).permute(1, 2, 0).numpy()
        true_label = qy[idx].item()
        pred_label = preds[idx].item()
        correct = true_label == pred_label

        ax = axes[i // cols, i % cols]
        ax.imshow(img)
        ax.axis("off")
        color = "green" if correct else "red"
        mark = "correct" if correct else "WRONG"
        ax.set_title(
            f"Actually: {class_names[true_label]}\n"
            f"Model guessed: {class_names[pred_label]}\n"
            f"({mark})",
            color=color, fontsize=10
        )

    # Hide any unused grid cells (e.g. 7 images in an 8-cell grid)
    for j in range(len(idxs), rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle("Green = model guessed right, Red = model guessed wrong", y=1.0, fontsize=13)
    fig.subplots_adjust(hspace=0.65, wspace=0.3, top=0.90)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=15)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--protonet_path", type=str, default="protonet_encoder.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_idx, val_idx, test_idx, base_ds, extra_ds = build_splits(root=args.data_root)
    test_bank = FewShotImageBank(base_ds, extra_ds, test_idx, augment=False, class_names=base_ds.classes)
    sampler = EpisodicSampler(test_bank, args.n_way, args.k_shot, args.q_query)

    encoder = ConvEncoder().to(device)
    encoder.load_state_dict(torch.load(args.protonet_path, map_location=device))

    sx, sy, qx, qy = sampler.sample_episode()
    class_names = sampler.episode_class_names()  # real names for labels 0..n_way-1 in THIS episode
    print(f"This episode's categories: {class_names}")

    plot_tsne(encoder, sx, sy, qx, qy, args.n_way, device, class_names)
    plot_example_predictions(encoder, sx, sy, qx, qy, args.n_way, device, class_names)


if __name__ == "__main__":
    main()
