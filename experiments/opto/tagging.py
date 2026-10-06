"""Partial labels + activity-dependent tagging: can the switch find the unlabeled
target data by itself?"""
import argparse, json, torch
import opto
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--k", type=int, default=64)
ap.add_argument("--fracs", type=str, default="0.05,0.1,0.25,0.5")
ap.add_argument("--tau", type=float, default=1.0)
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--p", type=float, default=0.5)
ap.add_argument("--tag", default="probe")
ap.add_argument("--thresh", type=float, default=0.5)
ap.add_argument("--l2", type=float, default=0.0)
ap.add_argument("--noise", type=float, default=0.0, help="fraction of target labels that are wrong")
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
    known = (Ttr == opto.TARGET) & lab
    if a.noise:  # replace a fraction of the labels with non-target rows
        n_bad = int(a.noise * known.sum())
        keep = known.nonzero().squeeze(1)[n_bad:]
        bad = (Ttr != opto.TARGET).nonzero().squeeze(1)
        bad = bad[torch.randperm(len(bad), generator=torch.Generator().manual_seed(11 + a.seed))[:n_bad]]
        known = torch.zeros_like(known); known[keep] = True; known[bad] = True
    stats = []
    opto.train(rm, Xtr, Ytr, Ttr, 3000, routed="strict", p_silence=a.p, labeled=lab, known_target=known,
               tag=a.tag, tag_thresh=a.thresh, tag_l2=a.l2, tag_tau=a.tau, tag_warmup=a.warmup, stats=stats)
    on = opto.acc_by_task(rm, Xte, Yte, Tte)
    opto.ablate(rm, [(li, u) for li, m in enumerate(rm.mlps) for u in m.region.nonzero().squeeze(1).tolist()])
    for m in rm.mlps: m.abl_val.zero_()
    off = opto.acc_by_task(rm, Xte, Yte, Tte)
    rl = opto.relearn(rm, X, Y, T, tr, 32, 200, a.seed)
    fin = stats[-1] if stats else {}
    rows.append(dict(seed=a.seed, frac=f, tag=a.tag, thresh=a.thresh, l2=a.l2, noise=a.noise, tau=a.tau, warmup=a.warmup, on=on, off=off,
                     relearn_add=rl, tag_stats=stats))
    print(f"[s{a.seed} {a.tag} th={a.thresh} noise={a.noise}] labels {f:.2f}  on {[round(x,2) for x in on]}  "
          f"OFF {[round(x,2) for x in off]}  relearn {rl:.2f}  "
          f"tag recall {fin.get('recall',0):.2f} precision {fin.get('precision',0):.2f}", flush=True)
json.dump(rows, open(a.out, "w"), indent=1)
