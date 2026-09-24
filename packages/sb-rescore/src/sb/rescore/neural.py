"""A small GRU language model over gloss tokens: the neural arm of the
next-gloss experiments (TODO §12.6).

Same surface as :class:`sb.rescore.prior.NgramLM` (``vocab``, ``dist``,
``gloss_dist``). It needs torch, which is the ``neural`` extra of this
package, so ``sb.rescore.prior`` stays importable without it.

It is here to answer one question: does a model that shares statistics
between similar glosses (embeddings) predict better than counts on a
1,757-sentence corpus? If it does not, the n-gram ships: it is a few KB of
tables the client can evaluate with no runtime.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sb.rescore.prior import BOS, EOS, NextGlossModel


class GRULM(NextGlossModel):
    def __init__(self, glosses: Sequence[str], emb: int = 64, hidden: int = 128,
                 dropout: float = 0.3, seed: int = 0):
        import torch
        from torch import nn

        self.vocab = list(glosses) + [EOS]
        self.inputs = [BOS] + list(glosses)  # input tokens: BOS then glosses (EOS is never an input)
        self.in_index = {g: i for i, g in enumerate(self.inputs)}
        self.out_index = {g: i for i, g in enumerate(self.vocab)}
        torch.manual_seed(seed)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(len(glosses) + 1, emb)
                self.gru = nn.GRU(emb, hidden, batch_first=True)
                self.drop = nn.Dropout(dropout)
                self.out = nn.Linear(hidden, len(glosses) + 1)

            def forward(self, x):
                h, _ = self.gru(self.drop(self.emb(x)))
                return self.out(self.drop(h))

        self.net = Net()
        self.seed = seed
        self._memo: dict[tuple[str, ...], np.ndarray] = {}

    def _encode(self, sentences: Sequence[Sequence[str]]):
        import torch

        xs, ys = [], []
        for s in sentences:
            xs.append([self.in_index[BOS]] + [self.in_index[g] for g in s])
            ys.append([self.out_index[g] for g in s] + [self.out_index[EOS]])
        T = max(len(x) for x in xs)
        X = torch.zeros(len(xs), T, dtype=torch.long)
        Y = torch.full((len(xs), T), -100, dtype=torch.long)  # -100 = ignored by cross_entropy
        for i, (x, y) in enumerate(zip(xs, ys)):
            X[i, :len(x)] = torch.tensor(x)
            Y[i, :len(y)] = torch.tensor(y)
        return X, Y

    def fit(self, sentences: Sequence[Sequence[str]], epochs: int = 60, lr: float = 3e-3,
            batch_size: int = 64, val_fraction: float = 0.1, patience: int = 6,
            weight_decay: float = 1e-4) -> dict:
        """Train with early stopping on a seeded ``val_fraction`` of the given
        sentences; keeps the best-val weights. Returns the training log."""
        import torch
        from torch.nn import functional as F

        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(sentences))
        n_val = max(1, int(len(sentences) * val_fraction))
        val = [sentences[i] for i in order[:n_val]]
        train = [sentences[i] for i in order[n_val:]]
        Xt, Yt = self._encode(train)
        Xv, Yv = self._encode(val)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, bad, log = float("inf"), None, 0, []
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), batch_size):
                b = perm[i:i + batch_size]
                logits = self.net(Xt[b])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), Yt[b].reshape(-1), ignore_index=-100)
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                lv = self.net(Xv)
                vloss = float(F.cross_entropy(lv.reshape(-1, lv.size(-1)), Yv.reshape(-1), ignore_index=-100))
            log.append({"epoch": ep + 1, "val_loss": vloss})
            if vloss < best - 1e-4:
                best, bad = vloss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        self._memo.clear()
        return {"best_val_loss": best, "epochs": len(log), "log": log}

    def dist(self, history: Sequence[str]) -> np.ndarray:
        import torch

        key = tuple(history)
        if key not in self._memo:
            x = torch.tensor([[self.in_index[BOS]] + [self.in_index[g] for g in history]])
            with torch.no_grad():
                p = torch.softmax(self.net(x)[0, -1].double(), -1).numpy()
            self._memo[key] = p
        return self._memo[key]

    def state_dict(self) -> dict:
        return self.net.state_dict()

    def load_state_dict(self, state: dict) -> "GRULM":
        self.net.load_state_dict(state)
        self.net.eval()
        self._memo.clear()
        return self
