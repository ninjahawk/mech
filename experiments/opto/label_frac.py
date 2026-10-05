"""What if only a fraction of the ADD training data is labeled (and so routed)?"""
import argparse, json, torch
import opto
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--k", type=int, default=64)
ap.add_argument("--mode", default="strict")
ap.add_argument("--p", type=float, default=0.5)
ap.add_argument("--fracs", type=str, default="0.05,0.1,0.25,0.5")
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(1)
X, Y, T, tr = opto.make_data(a.seed)
Xtr, Ytr, Ttr, Xte, Yte, Tte = X[tr], Y[tr], T[tr], X[~tr], Y[~tr], T[~tr]
rows = []
for f in [float(x) for x in a.fracs.split(",")]:
    torch.manual_seed(a.seed)
    rm = opto.Model()
    g = torch.Generator().manual_seed(1000 + a.seed)
    for m in rm.mlps:
        m.region[torch.randperm(m.region.numel(), generator=g)[: a.k // 2]] = True
    lab = torch.rand(len(Xtr), generator=torch.Generator().manual_seed(7 + a.seed)) < f
    opto.train(rm, Xtr, Ytr, Ttr, 3000, routed=a.mode, p_silence=a.p, labeled=lab)
    on = opto.acc_by_task(rm, Xte, Yte, Tte)
    opto.ablate(rm, [(li, u) for li, m in enumerate(rm.mlps) for u in m.region.nonzero().squeeze(1).tolist()])
    for m in rm.mlps: m.abl_val.zero_()
    off = opto.acc_by_task(rm, Xte, Yte, Tte)
    rl = opto.relearn(rm, X, Y, T, tr, 32, 200, a.seed)
    rows.append(dict(seed=a.seed, frac=f, mode=a.mode, p=a.p, on=on, off=off, relearn_add=rl))
    print(f"[s{a.seed} {a.mode} p={a.p}] label frac {f:.2f}  on {[round(x,2) for x in on]}  OFF {[round(x,2) for x in off]}  relearn {rl:.2f}", flush=True)
json.dump(rows, open(a.out, "w"), indent=1)
