"""
Train a Prototypical Network via episodic meta-training.

Meta-training loop, in plain words:
  For many iterations:
    1. Sample a random N-way K-shot episode from the TRAIN classes only.
    2. Embed support + query images with the encoder.
    3. Compute prototypes and the prototypical loss (see models.py).
    4. Backprop and update the encoder.
  Occasionally: sample episodes from the VAL classes (never trained on) to
  track how well the model generalizes to new classes.

Run:
    python train_protonet.py --iterations 5000 --n_way 5 --k_shot 5 --q_query 15
"""

import argparse
import torch
from torch import optim

from data import build_splits, FewShotImageBank, EpisodicSampler
from models import ConvEncoder, prototypical_loss


def evaluate_on_split(encoder, sampler, n_episodes, device):
    encoder.eval()
    accs = []
    with torch.no_grad():
        for _ in range(n_episodes):
            sx, sy, qx, qy = sampler.sample_episode()
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            es, eq = encoder(sx), encoder(qx)
            _, acc = prototypical_loss(es, sy, eq, qy, sampler.n_way)
            accs.append(acc)
    encoder.train()
    mean = sum(accs) / len(accs)
    return mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_every", type=int, default=250)
    parser.add_argument("--val_episodes", type=int, default=50)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_path", type=str, default="protonet_encoder.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_idx, val_idx, test_idx, base_ds, extra_ds = build_splits(root=args.data_root)
    print(f"Classes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    train_bank = FewShotImageBank(base_ds, extra_ds, train_idx, augment=True)
    val_bank = FewShotImageBank(base_ds, extra_ds, val_idx, augment=False)

    train_sampler = EpisodicSampler(train_bank, args.n_way, args.k_shot, args.q_query)
    val_sampler = EpisodicSampler(val_bank, args.n_way, args.k_shot, args.q_query)

    encoder = ConvEncoder().to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    best_val_acc = 0.0
    for it in range(1, args.iterations + 1):
        sx, sy, qx, qy = train_sampler.sample_episode()
        sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)

        es, eq = encoder(sx), encoder(qx)
        loss, acc = prototypical_loss(es, sy, eq, qy, args.n_way)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if it % 50 == 0:
            print(f"[iter {it}/{args.iterations}] train loss={loss.item():.3f} train acc={acc:.3f}")

        if it % args.val_every == 0:
            val_acc = evaluate_on_split(encoder, val_sampler, args.val_episodes, device)
            print(f"  -> val acc over {args.val_episodes} episodes: {val_acc:.3f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(encoder.state_dict(), args.save_path)
                print(f"  -> new best val acc, saved encoder to {args.save_path}")

    print(f"Done. Best val acc: {best_val_acc:.3f}. Encoder saved to {args.save_path}")


if __name__ == "__main__":
    main()
