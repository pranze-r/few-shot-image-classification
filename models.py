"""
Models for the meta-learning mini-project.

1. ConvEncoder: the standard "Conv-4-64" backbone used in almost every
   few-shot learning paper (Vinyals et al. 2016, Snell et al. 2017). Four
   conv blocks (conv -> batchnorm -> relu -> maxpool), turning a 32x32
   image into a small embedding vector.

2. prototypical_loss(): implements the core Prototypical Networks idea
   (Snell, Swersky, Zemel, "Prototypical Networks for Few-shot Learning",
   NeurIPS 2017):
     - Embed all support images.
     - Average the embeddings of each class's support images to get one
       "prototype" vector per class.
     - Embed the query images, compute (negative squared) Euclidean
       distance to each prototype, and treat that as classification logits.
     - Cross-entropy loss on those logits teaches the encoder to produce
       embeddings where same-class images cluster tightly together.

3. BaselineClassifier: an ordinary CNN classifier (same encoder + a linear
   softmax head) trained the "normal" supervised way on the base classes.
   At few-shot test time we fine-tune only its final layer (or all layers,
   see evaluate.py) on the tiny support set of a new episode -- this is the
   "no meta-learning" baseline we compare Prototypical Networks against.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class ConvEncoder(nn.Module):
    """Conv-4-64: four conv blocks, 64 channels each. Input 3x32x32 -> output 64-dim vector."""

    def __init__(self, hidden_channels=64, out_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(3, hidden_channels),
            conv_block(hidden_channels, hidden_channels),
            conv_block(hidden_channels, hidden_channels),
            conv_block(hidden_channels, out_dim),
        )

    def forward(self, x):
        x = self.encoder(x)              # (B, out_dim, 2, 2) for 32x32 input
        x = x.view(x.size(0), -1)        # flatten
        return x                          # (B, out_dim * 2 * 2)

    @property
    def embedding_dim(self):
        # 32 -> /2/2/2/2 = 2x2 spatial, times out_dim channels
        return 64 * 2 * 2


def prototypical_loss(embeddings_support, labels_support, embeddings_query, labels_query, n_way):
    """
    Core Prototypical Networks computation.

    Args:
        embeddings_support: (n_way * k_shot, D)
        labels_support:     (n_way * k_shot,) values in [0, n_way)
        embeddings_query:   (n_way * q_query, D)
        labels_query:       (n_way * q_query,) values in [0, n_way)
        n_way: number of classes in this episode

    Returns:
        loss (scalar tensor), accuracy (float)
    """
    D = embeddings_support.size(1)

    # 1. Compute one prototype per class = mean embedding of its support examples.
    prototypes = torch.zeros(n_way, D, device=embeddings_support.device)
    for c in range(n_way):
        mask = labels_support == c
        prototypes[c] = embeddings_support[mask].mean(dim=0)

    # 2. Compute squared Euclidean distance from every query to every prototype.
    #    dists[i, c] = || query_i - prototype_c ||^2
    dists = torch.cdist(embeddings_query, prototypes, p=2) ** 2

    # 3. Negative distance = logits. Closer prototype -> higher logit -> higher
    #    predicted probability. This is the "metric-based" classification rule.
    logits = -dists
    loss = F.cross_entropy(logits, labels_query)

    with torch.no_grad():
        preds = logits.argmax(dim=1)
        acc = (preds == labels_query).float().mean().item()

    return loss, acc


class BaselineClassifier(nn.Module):
    """
    Plain supervised classifier: encoder + linear head over the base
    (training) classes. Used as the "no meta-learning" comparison point.
    At few-shot test time, we discard/replace the head and fine-tune on
    the new episode's tiny support set (see evaluate.py).
    """

    def __init__(self, n_base_classes, encoder: ConvEncoder = None):
        super().__init__()
        self.encoder = encoder if encoder is not None else ConvEncoder()
        self.head = nn.Linear(self.encoder.embedding_dim, n_base_classes)

    def forward(self, x):
        feats = self.encoder(x)
        return self.head(feats)
