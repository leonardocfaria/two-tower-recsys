"""Two-tower retrieval model trained with in-batch sampled softmax + logQ popularity correction."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, n_genres, dim=32, hidden=64):
        super().__init__()
        self.user_id, self.item_id = nn.Embedding(n_users, hidden), nn.Embedding(n_items, hidden)
        self.age, self.gender, self.occ = nn.Embedding(7, 8), nn.Embedding(2, 4), nn.Embedding(21, 8)
        self.user_mlp = nn.Sequential(nn.Linear(hidden + 20 + n_genres, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.item_mlp = nn.Sequential(nn.Linear(hidden + n_genres, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def users(self, uid, age, gender, occ, profile):
        """profile: the user's genre mix over their training history (rows sum to 1)."""
        x = torch.cat([self.user_id(uid), self.age(age), self.gender(gender), self.occ(occ), profile], 1)
        return F.normalize(self.user_mlp(x), dim=1)

    def items(self, iid, genres):
        return F.normalize(self.item_mlp(torch.cat([self.item_id(iid), genres], 1)), dim=1)


def train_towers(train, users, items, epochs=15, batch=512, lr=3e-3, temp=0.05, seed=0, dim=32, logq_correction=True):
    """train: positives (user, item). In-batch negatives are popularity-biased (popular items appear more often
    as negatives), so subtract log(item frequency) from the logits (logQ correction)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    U = {c: torch.tensor(users.sort_values("user")[c].to_numpy()) for c in ("user", "age", "gender", "occupation")}
    G = torch.tensor(items.sort_values("item").filter(like="g").to_numpy(), dtype=torch.float32)
    P = torch.zeros(len(users), G.shape[1]).index_add_(0, torch.tensor(train.user.to_numpy().copy()), G[train.item.to_numpy()])
    P = P / P.sum(1, keepdim=True).clamp(min=1)
    model = TwoTower(len(users), len(items), G.shape[1], dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    freq = np.bincount(train.item, minlength=len(items)) + 1
    logq = torch.tensor(np.log(freq / freq.sum()) if logq_correction else np.zeros(len(freq)), dtype=torch.float32)
    u_all, i_all = train.user.to_numpy(), train.item.to_numpy()
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(u_all)), max(1, len(u_all) // batch)):
            u, i = torch.tensor(u_all[idx]), torch.tensor(i_all[idx])
            uv = model.users(u, U["age"][u], U["gender"][u], U["occupation"][u], P[u])
            iv = model.items(i, G[i])
            logits = uv @ iv.T / temp - logq[i][None, :]
            # the same item can appear twice in a batch: mask duplicates of the positive as negatives
            dup = (i[None, :] == i[:, None]) & ~torch.eye(len(i), dtype=torch.bool)
            loss = F.cross_entropy(logits.masked_fill(dup, -1e9), torch.arange(len(i)))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        user_emb = model.users(U["user"], U["age"], U["gender"], U["occupation"], P).numpy()
        item_emb = model.items(torch.arange(len(items)), G).numpy()
    return model, user_emb.astype(np.float32), item_emb.astype(np.float32)
