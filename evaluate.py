"""
Final comparison: Prototypical Network vs. the fine-tuned baseline, both
evaluated on TEST classes that neither model ever saw during training.

For each of many random test episodes:
  - Prototypical Network: embed support+query with the meta-trained encoder,
    classify queries by nearest prototype. No gradient updates at test time --
    this is the "learns from a few examples with no further training" magic.
  - Baseline: take the baseline's (frozen) encoder, attach a FRESH linear
    head, fine-tune that head (a few gradient steps) on the episode's support
    set only, then evaluate on the query set. This is the closest fair
    comparison to a "normal" model trying to handle a new few-shot task.

Reports mean accuracy +/- 95% confidence interval over many episodes, which
is the standard way results are reported in few-shot learning papers (a
single episode's accuracy is noisy; you need ~200-600 episodes for a
believable number).

Run:
    python evaluate.py --n_episodes 300 --n_way 5 --k_shot 5 --q_query 15
"""

import argparse
import copy
import math
import torch
import torch.nn as nn
from torch import optim

from data import build_splits, FewShotImageBank, EpisodicSampler
from models import ConvEncoder, prototypical_loss


def mean_confidence_interval(values, confidence=0.95):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    # 1.96 ~ z-score for 95% CI (fine for n >= ~30, which we'll have)
    half_width = 1.96 * std / math.sqrt(n)
    return mean, half_width


import time

def eval_protonet_episode(encoder, sx, sy, qx, qy, n_way, device):
    encoder.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        es, eq = encoder(sx.to(device)), encoder(qx.to(device))
        _, acc = prototypical_loss(es, sy.to(device), eq, qy.to(device), n_way)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_time = (t_end - t_start) * 1000.0  # in ms
    train_time = 0.0  # ProtoNet requires 0 gradient updates / 0 test-time training
    return acc, train_time, total_time


def eval_baseline_episode(frozen_encoder, sx, sy, qx, qy, n_way, device,
                           ft_steps=50, ft_lr=1e-2):
    """
    Fine-tune a FRESH linear head on top of the frozen baseline encoder,
    using only the episode's support set, then evaluate on the query set.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    embedding_dim = frozen_encoder.embedding_dim
    head = nn.Linear(embedding_dim, n_way).to(device)
    optimizer = optim.Adam(head.parameters(), lr=ft_lr)
    criterion = nn.CrossEntropyLoss()

    sx, sy = sx.to(device), sy.to(device)
    with torch.no_grad():
        support_feats = frozen_encoder(sx)

    # --- Test-Time Training / Fine-tuning Phase ---
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_ft_start = time.perf_counter()

    for _ in range(ft_steps):
        logits = head(support_feats)
        loss = criterion(logits, sy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_ft_end = time.perf_counter()
    train_time = (t_ft_end - t_ft_start) * 1000.0  # in ms

    # --- Query Evaluation Phase ---
    qx, qy = qx.to(device), qy.to(device)
    with torch.no_grad():
        query_feats = frozen_encoder(qx)
        preds = head(query_feats).argmax(dim=1)
        acc = (preds == qy).float().mean().item()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    total_time = (t_end - t_start) * 1000.0  # in ms

    return acc, train_time, total_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=300)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=15)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--protonet_path", type=str, default="protonet_encoder.pt")
    parser.add_argument("--baseline_path", type=str, default="baseline_encoder.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_idx, val_idx, test_idx, base_ds, extra_ds = build_splits(root=args.data_root)
    test_bank = FewShotImageBank(base_ds, extra_ds, test_idx, augment=False)
    sampler = EpisodicSampler(test_bank, args.n_way, args.k_shot, args.q_query)

    protonet_encoder = ConvEncoder().to(device)
    protonet_encoder.load_state_dict(torch.load(args.protonet_path, map_location=device))
    protonet_encoder.eval()

    baseline_encoder = ConvEncoder().to(device)
    baseline_encoder.load_state_dict(torch.load(args.baseline_path, map_location=device))
    baseline_encoder.eval()

    proto_accs, proto_train_times, proto_total_times = [], [], []
    base_accs, base_train_times, base_total_times = [], [], []

    for i in range(1, args.n_episodes + 1):
        sx, sy, qx, qy = sampler.sample_episode()

        p_acc, p_train_t, p_tot_t = eval_protonet_episode(
            protonet_encoder, sx, sy, qx, qy, args.n_way, device
        )
        proto_accs.append(p_acc)
        proto_train_times.append(p_train_t)
        proto_total_times.append(p_tot_t)

        b_acc, b_train_t, b_tot_t = eval_baseline_episode(
            baseline_encoder, sx, sy, qx, qy, args.n_way, device
        )
        base_accs.append(b_acc)
        base_train_times.append(b_train_t)
        base_total_times.append(b_tot_t)

        if i % 50 == 0:
            print(f"...{i}/{args.n_episodes} episodes evaluated")

    proto_mean, proto_ci = mean_confidence_interval(proto_accs)
    base_mean, base_ci = mean_confidence_interval(base_accs)

    avg_p_train = sum(proto_train_times) / len(proto_train_times)
    avg_p_total = sum(proto_total_times) / len(proto_total_times)
    avg_b_train = sum(base_train_times) / len(base_train_times)
    avg_b_total = sum(base_total_times) / len(base_total_times)

    print("\n" + "=" * 70)
    print(f"=== Results on held-out TEST classes ({args.n_way}-way {args.k_shot}-shot, {args.n_episodes} episodes) ===")
    print("=" * 70)
    print(f"{'Model':<25} | {'Accuracy (+/- 95% CI)':<22} | {'Test-Time Training':<18} | {'Total Time/Ep':<14}")
    print("-" * 70)
    print(f"{'Prototypical Network':<25} | {proto_mean*100:6.2f}% +/- {proto_ci*100:4.2f}%   | {avg_p_train:6.2f} ms (0 steps)  | {avg_p_total:6.2f} ms")
    print(f"{'Fine-tuned Baseline':<25} | {base_mean*100:6.2f}% +/- {base_ci*100:4.2f}%   | {avg_b_train:6.2f} ms (50 steps) | {avg_b_total:6.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
