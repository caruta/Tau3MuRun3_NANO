"""Multi-token transformer for tau -> 3mu candidates (ParT-style, standalone PyTorch).

Sequence per candidate: [CLS from candidate features] + muons (3) + tracks + neutrals + SVs + jets.
Each token type has its own embedding MLP and a learned type embedding. Attention gets a
pairwise bias (as in ParT) from scale-free pair features of the (deta, dphi, log ptrel)
token geometry: ln dR, ln z = ln(min(pti, ptj) / (pti + ptj)), ln kT/pT(cand) = ln(min(pti, ptj) dR),
all with momenta relative to the candidate, so no mass scale enters.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

TOKEN_ORDER = ["mu", "trk", "neu", "sv", "jet"]


class Embed(nn.Module):
    def __init__(self, n_in, d, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(n_in), nn.Linear(n_in, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))

    def forward(self, x):
        return self.net(x)


class PairBias(nn.Module):
    """Pair features -> one additive attention bias per head (shared by all layers)."""

    def __init__(self, heads, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, heads))

    def forward(self, geom, valid):
        # geom [B, L, 3] = (deta, dphi, log ptrel); valid [B, L]
        deta = geom[..., 0, None] - geom[..., None, :, 0]
        dphi = geom[..., 1, None] - geom[..., None, :, 1]
        dphi = torch.remainder(dphi + math.pi, 2 * math.pi) - math.pi
        dr = torch.sqrt(deta ** 2 + dphi ** 2 + 1e-8)
        lpt = geom[..., 2].clamp(-12, 6)
        lmin = torch.minimum(lpt[..., :, None], lpt[..., None, :])
        lsum = torch.logaddexp(lpt[..., :, None], lpt[..., None, :])
        same = torch.eye(geom.shape[1], device=geom.device, dtype=torch.bool).expand_as(dr)
        off = (~same).to(dr.dtype)
        lndr = torch.log(dr.clamp(min=1e-4))
        # self pairs carry only the "same token" flag
        f = torch.stack([lndr * off, (lmin - lsum) * off, (lmin + lndr) * off, same.to(dr.dtype)], dim=-1)
        bias = self.net(f).permute(0, 3, 1, 2)  # [B, H, L, L]
        pair_ok = valid[:, None, :, None] & valid[:, None, None, :]
        return bias * pair_ok


class Block(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff * d, d))
        self.drop = nn.Dropout(dropout)
        self.attn_drop = dropout

    def forward(self, x, attn_bias):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = Fn.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.to(q.dtype),
                                            dropout_p=self.attn_drop if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class Tau3MuTransformer(nn.Module):
    def __init__(self, stats, d=128, heads=8, layers=6, ff=4, dropout=0.1, pair_bias=True):
        super().__init__()
        feats = stats["features"]
        self.tokens = [t for t in TOKEN_ORDER if t in feats["tokens"]]
        self.geom_index = feats["geom_index"]
        self.config = dict(d=d, heads=heads, layers=layers, ff=ff, dropout=dropout, pair_bias=pair_bias)
        norm = stats["norm"]

        def buf(name, key, n):
            m = torch.tensor(norm[key]["mean"], dtype=torch.float32) if key in norm else torch.zeros(n)
            s = torch.tensor(norm[key]["std"], dtype=torch.float32) if key in norm else torch.ones(n)
            self.register_buffer(f"{name}_mean", m)
            self.register_buffer(f"{name}_std", s)

        n_cand = len(feats["cand"])
        buf("cand", "cand", n_cand)
        self.cand_embed = Embed(n_cand, d, dropout)
        self.embed = nn.ModuleDict()
        for t in self.tokens:
            n = len(feats["tokens"][t])
            buf(t, f"{t}_feats", n)
            self.embed[t] = Embed(n, d, dropout)
        self.type_embed = nn.Parameter(torch.zeros(len(self.tokens) + 1, d))
        nn.init.normal_(self.type_embed, std=0.02)
        self.pair = PairBias(heads) if pair_bias else None
        self.blocks = nn.ModuleList([Block(d, heads, ff, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def _norm(self, name, x):
        return (x - getattr(self, f"{name}_mean")) / getattr(self, f"{name}_std")

    def forward(self, batch):
        cand = self._norm("cand", batch["cand"].float())
        B = cand.shape[0]
        xs = [self.cand_embed(cand)[:, None] + self.type_embed[0]]
        masks = [torch.ones(B, 1, dtype=torch.bool, device=cand.device)]
        geoms = [torch.zeros(B, 1, 3, device=cand.device)]
        for i, t in enumerate(self.tokens):
            raw = batch[f"{t}_feats"].float()
            m = batch[f"{t}_mask"]
            x = self.embed[t](self._norm(t, raw)) + self.type_embed[i + 1]
            xs.append(x * m[..., None])
            masks.append(m)
            geoms.append(raw[..., self.geom_index[t]] * m[..., None])
        x = torch.cat(xs, 1)
        valid = torch.cat(masks, 1)
        H = self.config["heads"]
        L = x.shape[1]
        bias = torch.zeros(B, H, L, L, device=x.device)
        if self.pair is not None:
            bias = bias + self.pair(torch.cat(geoms, 1), valid)
        bias = bias.masked_fill(~valid[:, None, None, :], float("-inf"))
        for blk in self.blocks:
            x = blk(x, bias)
        return self.head(self.ln(x[:, 0]))[:, 0]


def build(stats, **kw):
    return Tau3MuTransformer(stats, **kw)
