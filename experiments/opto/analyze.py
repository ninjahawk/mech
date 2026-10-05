"""Aggregate results/seed*.json into a table and a figure."""
import glob, json, statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [r for f in sorted(glob.glob("results/seed*.json")) for r in json.load(open(f))["rows"]]
seeds = sorted({r["seed"] for r in rows})
agg = defaultdict(list)
for r in rows:
    agg[(r["method"], r["k"])].append(r)


def ms(xs):
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)


lines = [f"seeds: {seeds}", "",
         "| method | k | ADD acc ↓ | SUB | ADD2 | SQ | mean off-target drop ↓ | ADD after relearn ↓ |",
         "|---|---|---|---|---|---|---|---|"]
order = ["none", "random", "act-diff", "attrib", "exhaustive", "routed-on", "routed",
         "rt-loose-on", "rt-loose", "rt-strict-nosil-on", "rt-strict-nosil",
         "rt-loose-sil-on", "rt-loose-sil", "oracle"]
for meth in order:
    for k in sorted({k for (m, k) in agg if m == meth}):
        rs = agg[(meth, k)]
        a = [ms([r["acc"][t] for r in rs]) for t in range(4)]
        od, rl = ms([r["off_drop"] for r in rs]), ms([r["relearn_add"] for r in rs])
        lines.append(f"| {meth} | {k} | {a[0][0]:.2f}±{a[0][1]:.2f} | {a[1][0]:.2f} | {a[2][0]:.2f} | "
                     f"{a[3][0]:.2f} | {od[0]:+.3f}±{od[1]:.3f} | {rl[0]:.2f}±{rl[1]:.2f} |")
open("results/table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
styles = {"random": ("#9e9e9e", "o"), "act-diff": ("#f2a541", "s"), "attrib": ("#e07a5f", "^"),
          "exhaustive": ("#b5179e", "D"), "routed": ("#2a9d8f", "*")}
for meth, (c, mk) in styles.items():
    ks = sorted({k for (m, k) in agg if m == meth})
    for i, key in enumerate(["add_acc", "off_drop", "relearn_add"]):
        mu = [st.mean(r[key] for r in agg[(meth, k)]) for k in ks]
        ax[i].plot(ks, mu, marker=mk, color=c, label=meth, lw=2, ms=8 if mk == "*" else 6)
for i, key in enumerate(["add_acc", "off_drop", "relearn_add"]):
    o = st.mean(r[key] for r in agg[("oracle", 0)])
    ax[i].axhline(o, color="k", ls="--", lw=1, label="oracle (never saw ADD)")
    ax[i].set_xscale("log", base=2); ax[i].set_xlabel("units switched off (k)")
ax[0].set_title("ADD accuracy after switch-off  (lower = stronger)")
ax[1].set_title("Damage to other skills  (lower = more specific)")
ax[2].set_title("ADD after 32-shot relearning  (lower = truly gone)")
ax[0].legend(fontsize=8)
for a in ax: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("results/summary.png", dpi=130)
print("wrote results/table.md, results/summary.png")
