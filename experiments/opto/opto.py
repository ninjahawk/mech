"""
Optogenetics vs. post-hoc interpretability: a controlled toy test.

Question: is a control handle *installed during training* (the optogenetics move:
engineer the organism so a cell type carries a switch) better than a handle
*found after training* (the standard mech-interp move: train, then locate units)?

Setup
  - Residual MLP (2 MLP blocks) trained on 4 modular-arithmetic skills, chosen by a
    task token: ADD (a+b), SUB (a-b), ADD2 (a+2b), SQ (a^2+b^2), all mod P.
    Trained on 80% of pairs; everything is evaluated on the held-out 20%, so we
    measure the *algorithm*, not a memorised lookup table.
  - Target skill to switch off: ADD. SUB and ADD2 share its structure, so they are
    the hard off-target tests.

Handles compared (each is a set of k MLP units that gets mean-ablated)
  random      k random units                              (control)
  act-diff    units most selectively active on ADD        ("electrode recording")
  attrib      attribution-patching estimate of ablation effect, ADD minus others
  exhaustive  true single-unit ablation effect, ADD minus others (best-case post-hoc)
  routed      units designated *before training*; gradient routing sends ADD
              learning into them and keeps all other learning out ("genetic targeting"),
              and the other tasks train with those units randomly silenced so they
              don't come to depend on them. Variants without each ingredient:
              rt-loose (other tasks may also update the region), rt-strict-nosil
              (no silencing), rt-loose-sil (loose + silencing).
  oracle      model trained with ADD data removed entirely (gold standard for removal)

Metrics (held-out pairs)
  on-target   ADD accuracy after ablation (lower = stronger switch)
  off-target  accuracy drop on SUB / ADD2 / SQ (lower = more specific switch)
  relearn     ADD accuracy after briefly fine-tuning the ablated model on a few ADD
              examples -- is the skill gone, or just suppressed?
              (oracle shows what a model that never learned ADD reaches)
  cost        does installing the switch hurt the unablated model?
"""
import argparse, copy, json, math, random, time

import torch
import torch.nn as nn
import torch.nn.functional as F

P = 31
TASKS = ["ADD", "SUB", "ADD2", "SQ"]
FNS = {"ADD": lambda a, b: a + b, "SUB": lambda a, b: a - b,
       "ADD2": lambda a, b: a + 2 * b, "SQ": lambda a, b: a * a + b * b}
TARGET = 0  # ADD


def make_data(seed=0, frac=0.8):
    xs, ys, ts = [], [], []
    for t, name in enumerate(TASKS):
        for a in range(P):
            for b in range(P):
                xs.append([t, a, b]); ys.append(FNS[name](a, b) % P); ts.append(t)
    X, Y, T = torch.tensor(xs), torch.tensor(ys), torch.tensor(ts)
    tr = torch.rand(len(X), generator=torch.Generator().manual_seed(seed)) < frac
    return X, Y, T, tr


class MLP(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d, m) / math.sqrt(d))
        self.b_in = nn.Parameter(torch.zeros(m))
        self.W_out = nn.Parameter(torch.randn(m, d) / math.sqrt(m))
        self.b_out = nn.Parameter(torch.zeros(d))
        self.register_buffer("region", torch.zeros(m, dtype=torch.bool))  # routed units
        self.register_buffer("abl", torch.zeros(m, dtype=torch.bool))     # ablated units
        self.register_buffer("abl_val", torch.zeros(m))                   # mean activation
        self.cache = None
        self.silence = None  # [N, 1] bool: zero the region units for these rows

    def forward(self, x):
        h = F.relu(x @ self.W_in + self.b_in)
        if self.silence is not None:
            h = h.masked_fill(self.silence & self.region, 0.0)
        if self.abl.any():
            h = torch.where(self.abl, self.abl_val.expand_as(h), h)
        self.cache = h
        return h @ self.W_out + self.b_out


class Model(nn.Module):
    def __init__(self, d=64, m=512, layers=2):
        super().__init__()
        self.ea, self.eb = nn.Embedding(P, d), nn.Embedding(P, d)
        self.et = nn.Embedding(len(TASKS), d)
        self.ms = nn.ModuleList([MLP(d, m) for _ in range(layers)])
        self.unemb = nn.Linear(d, P)

    def embed(self, x):
        return self.ea(x[:, 1]) + self.eb(x[:, 2]) + self.et(x[:, 0])

    def forward(self, x):
        h = self.embed(x)
        for m in self.ms:
            h = h + m(h)
        return self.unemb(h)

    @property
    def mlps(self):
        return list(self.ms)


# ---------------------------------------------------------------- training

def make_opt(model, lr, steps, wd=1.0, warmup=100):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    return opt, sched


def mask_mlp_grads(model, keep_region):
    for m in model.mlps:
        drop = ~m.region if keep_region else m.region
        m.W_in.grad[:, drop] = 0
        m.b_in.grad[drop] = 0
        m.W_out.grad[drop] = 0


def set_silence(model, mask):
    for m in model.mlps:
        m.silence = mask


@torch.no_grad()
def region_dependence(model, X, Y):
    """Per-example loss increase when the region units are silenced: how much the
    model relies on the switch units for this example."""
    base = F.cross_entropy(model(X), Y, reduction="none")
    set_silence(model, torch.ones(len(X), 1, dtype=torch.bool))
    sil = F.cross_entropy(model(X), Y, reduction="none")
    set_silence(model, None)
    return sil - base


def probe_tags(model, X, known_t, iters=200, thresh=0.5, l2=0.0):
    """Positive-unlabeled tagging from the model's own activations. Fit a linear probe
    to separate labeled-target rows from all other rows, then rescale by the probe's
    mean score on labeled rows (Elkan & Noto 2008) to estimate P(target | x)."""
    with torch.no_grad():
        h = model.embed(X)
        h = (h - h.mean(0)) / (h.std(0) + 1e-6)
    w = torch.zeros(h.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    y = known_t.float()
    for _ in range(iters):
        loss = F.binary_cross_entropy_with_logits(h @ w + b, y) + l2 * (w * w).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        g = torch.sigmoid(h @ w + b)
        c = g[known_t].mean()
        return (g / c) > thresh


def train(model, X, Y, T, steps, routed=None, lr=1e-2, p_silence=0.0, labeled=None,
          tag=False, tag_tau=1.0, tag_warmup=300, stats=None, tag_every=50, tag_thresh=0.5, tag_l2=0.0):
    """Gradient routing. ADD examples may update embeddings/unembedding but, inside
    the MLPs, only the pre-designated `region` units.
      routed="loose":  all other data updates every parameter
      routed="strict": all other data updates everything EXCEPT the region units
    p_silence: on non-ADD examples, zero the region units with this probability, so
    the rest of the network learns not to depend on them.
    labeled: optional bool mask; only labeled ADD rows are known to be ADD. Without
    `tag`, unlabeled ADD rows are treated like any other data (imperfect labels).
    tag="probe": every `tag_every` steps, a positive-unlabeled linear probe on the
    model's own activations decides which unlabeled rows are target.
    tag="dep": activity-dependent tagging. Unlabeled rows are held out for the first
    `tag_warmup` steps (so the region learns the target from labeled rows alone);
    after that, an unlabeled row whose loss rises by more than `tag_tau` nats when the
    region is silenced is treated as target for that step, the rest as retain."""
    opt, sched = make_opt(model, lr, steps)
    known_t = T == TARGET
    if labeled is not None:
        known_t = known_t & labeled
    unl = ~known_t if (tag and labeled is not None) else torch.zeros_like(known_t)
    probe_hit = None
    for step in range(steps):
        opt.zero_grad()
        if routed:
            is_t, use = known_t.clone(), torch.ones_like(known_t)
            if unl.any() and tag == "probe":
                if step % tag_every == 0:
                    probe_hit = probe_tags(model, X, known_t, thresh=tag_thresh, l2=tag_l2) & unl
                    if stats is not None and step % 100 == 0:
                        true_t = (T == TARGET) & unl
                        stats.append(dict(step=step, tagged=int(probe_hit.sum()),
                                          recall=float((probe_hit & true_t).sum() / max(1, true_t.sum())),
                                          precision=float((probe_hit & true_t).sum() / max(1, probe_hit.sum()))))
                is_t |= probe_hit
            elif unl.any():
                if step < tag_warmup:
                    use = ~unl
                else:
                    idx = unl.nonzero().squeeze(1)
                    hit = region_dependence(model, X[idx], Y[idx]) > tag_tau
                    is_t[idx[hit]] = True
                    if stats is not None and step % 100 == 0:
                        true_t = T[idx] == TARGET
                        stats.append(dict(step=step, tagged=int(hit.sum()),
                                          recall=float((hit & true_t).sum() / max(1, true_t.sum())),
                                          precision=float((hit & true_t).sum() / max(1, hit.sum()))))
            ret = use & ~is_t
            n_ret = int(ret.sum())
            set_silence(model, (torch.rand(n_ret, 1) < p_silence) if p_silence else None)
            (F.cross_entropy(model(X[ret]), Y[ret], reduction="sum") / len(X)).backward()
            set_silence(model, None)
            if routed == "strict":
                mask_mlp_grads(model, keep_region=False)
            saved = [p.grad.clone() for p in model.parameters()]
            opt.zero_grad()
            (F.cross_entropy(model(X[is_t]), Y[is_t], reduction="sum") / len(X)).backward()
            mask_mlp_grads(model, keep_region=True)
            for p, g in zip(model.parameters(), saved):
                p.grad += g
        else:
            F.cross_entropy(model(X), Y).backward()
        opt.step(); sched.step()
    return model


@torch.no_grad()
def acc_by_task(model, X, Y, T):
    pred = model(X).argmax(-1)
    return [(pred[T == t] == Y[T == t]).float().mean().item() for t in range(len(TASKS))]


# ---------------------------------------------------------------- handles
# Rankings use TRAIN data only (that's all an interpretability researcher has).

def units_flat(model):
    return [(li, u) for li, m in enumerate(model.mlps) for u in range(m.W_in.shape[1])]


@torch.no_grad()
def set_means(model, X):
    model(X)
    for m in model.mlps:
        m.abl_val.copy_(m.cache.mean(0))


def clear_ablation(model):
    for m in model.mlps:
        m.abl.zero_()


def ablate(model, units):
    clear_ablation(model)
    for li, u in units:
        model.mlps[li].abl[u] = True


def by_score(model, scores):
    us = units_flat(model)
    return [us[i] for i in sorted(range(len(us)), key=lambda i: -scores[i])]


def rank_random(model, X, Y, T, seed):
    us = units_flat(model)
    random.Random(seed).shuffle(us)
    return us


@torch.no_grad()
def rank_actdiff(model, X, Y, T):
    model(X)
    scores = []
    for m in model.mlps:
        h = m.cache
        scores += ((h[T == TARGET].mean(0) - h[T != TARGET].mean(0)) / (h.std(0) + 1e-6)).tolist()
    return by_score(model, scores)


def rank_attrib(model, X, Y, T):
    # first-order estimate of the loss change from mean-ablating each unit
    model.zero_grad()
    caches = []
    def hook(mod, inp, out):
        mod.cache.retain_grad(); caches.append(mod.cache)
    hs = [m.register_forward_hook(hook) for m in model.mlps]
    F.cross_entropy(model(X), Y, reduction="sum").backward()
    for hk in hs: hk.remove()
    scores = []
    for m, h in zip(model.mlps, caches):
        eff = (m.abl_val - h) * h.grad
        scores += (eff[T == TARGET].mean(0) - eff[T != TARGET].mean(0)).tolist()
    model.zero_grad()
    return by_score(model, scores)


@torch.no_grad()
def rank_exhaustive(model, X, Y, T):
    def losses():
        l = F.cross_entropy(model(X), Y, reduction="none")
        return l[T == TARGET].mean().item(), l[T != TARGET].mean().item()
    clear_ablation(model)
    b_on, b_off = losses()
    scores = []
    for li, u in units_flat(model):
        model.mlps[li].abl[u] = True
        on, off = losses()
        model.mlps[li].abl[u] = False
        scores.append((on - b_on) - (off - b_off))
    return by_score(model, scores)


# ---------------------------------------------------------------- relearning

def relearn(model, X, Y, T, tr, shots, steps, seed):
    """Fine-tune the ablated model (ablation stays in place) on `shots` ADD train pairs
    plus all retain train data, then report held-out ADD accuracy."""
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    add_tr = ((T == TARGET) & tr).nonzero().squeeze(1)
    shot = add_tr[torch.randperm(len(add_tr), generator=g)[:shots]]
    ft = torch.cat([shot, ((T != TARGET) & tr).nonzero().squeeze(1)])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.98))
    for _ in range(steps):
        loss = F.cross_entropy(m(X[ft]), Y[ft])
        opt.zero_grad(); loss.backward(); opt.step()
    te = (T == TARGET) & ~tr
    with torch.no_grad():
        return (m(X[te]).argmax(-1) == Y[te]).float().mean().item()


# ---------------------------------------------------------------- main

def run_seed(seed, args, log):
    torch.manual_seed(seed)
    X, Y, T, tr = make_data(seed)
    Xtr, Ytr, Ttr = X[tr], Y[tr], T[tr]
    Xte, Yte, Tte = X[~tr], Y[~tr], T[~tr]
    ks = [int(k) for k in args.ks.split(",")]
    rows = []

    def record(method, k, model, base):
        acc = acc_by_task(model, Xte, Yte, Tte)
        rl = relearn(model, X, Y, T, tr, args.shots, args.relearn_steps, seed)
        off = [base[t] - acc[t] for t in range(len(TASKS)) if t != TARGET]
        r = dict(seed=seed, method=method, k=k, acc=acc, base=base, add_acc=acc[TARGET],
                 off_drop=sum(off) / len(off), max_off_drop=max(off), relearn_add=rl)
        rows.append(r)
        log(f"[s{seed}] {method:10s} k={k:4d}  " +
            "  ".join(f"{n} {a:.2f}" for n, a in zip(TASKS, acc)) +
            f"  | off-drop {r['off_drop']:+.3f}  relearn {rl:.2f}")

    t0 = time.time()
    std = train(Model(), Xtr, Ytr, Ttr, args.steps)
    set_means(std, Xtr)
    base = acc_by_task(std, Xte, Yte, Tte)
    log(f"[s{seed}] standard model trained {time.time()-t0:.0f}s")
    record("none", 0, std, base)
    rankings = {
        "random": rank_random(std, Xtr, Ytr, Ttr, seed),
        "act-diff": rank_actdiff(std, Xtr, Ytr, Ttr),
        "attrib": rank_attrib(std, Xtr, Ytr, Ttr),
        "exhaustive": rank_exhaustive(std, Xtr, Ytr, Ttr),
    }
    for name, order in rankings.items():
        for k in ks:
            ablate(std, order[:k]); record(name, k, std, base)
        clear_ablation(std)

    def routed_model(k, mode, p):
        torch.manual_seed(seed)
        rm = Model()
        g = torch.Generator().manual_seed(1000 + seed)
        for m in rm.mlps:  # region is chosen at random BEFORE training
            m.region[torch.randperm(m.region.numel(), generator=g)[: k // len(rm.mlps)]] = True
        train(rm, Xtr, Ytr, Ttr, args.steps, routed=mode, p_silence=p)
        return rm

    def flip_off(rm):
        ablate(rm, [(li, u) for li, m in enumerate(rm.mlps) for u in m.region.nonzero().squeeze(1).tolist()])
        for m in rm.mlps:
            m.abl_val.zero_()  # matches the silencing used in training

    variants = [("routed", "strict", 0.5, ks)] + [
        (name, mode, p, [args.variant_k]) for name, mode, p in
        [("rt-loose", "loose", 0.0), ("rt-strict-nosil", "strict", 0.0), ("rt-loose-sil", "loose", 0.5)]]
    for name, mode, p, kk in variants:
        for k in kk:
            rm = routed_model(k, mode, p)
            record(name + "-on", k, rm, base)  # switch installed, not flipped: the cost
            flip_off(rm)
            record(name, k, rm, base)

    torch.manual_seed(seed)
    keep = Ttr != TARGET
    orc = train(Model(), Xtr[keep], Ytr[keep], Ttr[keep], args.steps)
    set_means(orc, Xtr)
    record("oracle", 0, orc, base)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--ks", type=str, default="16,32,64,128,256")
    ap.add_argument("--variant_k", type=int, default=64)
    ap.add_argument("--shots", type=int, default=32)
    ap.add_argument("--relearn_steps", type=int, default=200)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rows = run_seed(args.seed, args, lambda s: print(s, flush=True))
    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
