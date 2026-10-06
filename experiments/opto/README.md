# Optogenetics → mech interp: does a built-in switch beat a found one?

## The question

Optogenetics (2026 Nobel, Deisseroth / Hegemann / Nagel) wasn't a better way to *find*
neurons. It **installed a switch into a chosen cell type during development**, and then
flipped it.

Mech interp mostly works the other way round: train the model, then go looking for the
units responsible for a behavior (probes, attribution, SAEs) and ablate them.

**Hypothesis:** a control handle installed during training is more *specific*, more
*complete* and more *durable* than the best handle you can find afterwards.

## Setup (`opto.py`)

- A residual MLP with 2 × 512 hidden units, trained on four modular-arithmetic skills
  selected by a task token: **ADD** a+b, **SUB** a−b, **ADD2** a+2b, **SQ** a²+b² (mod 31).
- It trains on 80% of input pairs. Everything is scored on the held-out 20%, so these
  are learned algorithms, not memorised lookup tables (about 99% held-out accuracy).
- The target is to switch off **ADD**. SUB and ADD2 are its near neighbors, which makes
  them the hard test for collateral damage.

| Handle | How it's chosen | Analogy |
|---|---|---|
| `random` | k random units | control |
| `act-diff` | units most selectively active on ADD | electrode recording |
| `attrib` | attribution-patching estimate of the ablation effect, ADD minus others | standard interp |
| `exhaustive` | the true single-unit ablation effect, ADD minus others | best-case post-hoc |
| **`routed`** | k units picked at random **before training**. ADD data may update only those units; other data may not touch them, and trains with them randomly silenced (p = 0.5). | genetic targeting + opsin |
| `oracle` | ADD data removed from training entirely | gold standard |

**Metrics** (all on held-out pairs; 4 seeds):
- ADD accuracy after switch-off: lower means the switch removes more.
- Damage to the other skills: lower means the switch is more specific.
- ADD accuracy after 32-shot fine-tuning: lower means ADD is actually gone, not just
  suppressed.

## Results

![summary](results/summary.png)

Selected rows (mean of 4 seeds; full table in `results/table.md`):

| Handle | k | ADD after off ↓ | Damage to others ↓ | ADD after relearning ↓ |
|---|---|---|---|---|
| exhaustive (best post-hoc) | 16 | 0.27 | 0.33 | 0.63 |
| exhaustive | 64 | 0.08 | 0.52 | 0.36 |
| attrib | 256 | 0.04 | 0.69 | 0.24 |
| **routed** | **16** | **0.08** | **0.00** | **0.17** |
| **routed** | **128** | **0.06** | **0.00** | **0.12** |
| oracle (never saw ADD) | – | 0.04 | 0.03 | 0.05 |

**1. With full labels, the built-in switch wins clearly.** It removes ADD as completely as
the best post-hoc method, with **zero** collateral damage at every size. The post-hoc
methods can't separate ADD from SUB/ADD2: by the point they suppress ADD, they've cost
33–69 points on the other skills. ADD also stays mostly gone after relearning (0.12–0.17,
vs 0.24–0.63 post-hoc). It isn't quite at the oracle's 0.05, so some trace of ADD
survives outside the switched units.

**2. Installing the switch is free.** With the switch on, the routed model scores 1.00 on
every skill, slightly above the normal model's 0.99.

**3. What made it work.** Each ingredient was tested separately at k = 64:

| Variant | ADD after off | Damage to others |
|---|---|---|
| loose: other data may also update the ADD units | 0.05 | 0.57 |
| strict, no silencing | 0.05 | 0.48 |
| loose + silencing | 0.13 | 0.01 |
| **strict + silencing** | **0.09** | **0.00** |

Silencing is the ingredient that matters. Without it, the rest of the network learns to
*depend* on the ADD units, so removing them breaks the other skills. Neuroscience has the
same problem: silencing a cell type disrupts the downstream circuits that relied on its
normal activity.

**4. With partial labels, plain routing fails** (`label_frac.py`). With only part of
the ADD data labeled:

| Labeled | Silencing | ADD after off | Others after off | ADD after relearning |
|---|---|---|---|---|
| 5–50% | 0.5 | 1.00 (switch does nothing) | 1.00 | ~0.98 |
| 25–50% | 0.1 | ~1.00 (switch does nothing) | 1.00 | 0.87–1.00 |
| 25–50% | 0 | 0.05 | 0.13–0.94 (heavy damage) | 0.23–0.77 |

Unlabeled ADD examples get treated as ordinary data. With silencing, they teach the rest
of the network to do ADD on its own, so switching off the ADD units does nothing. Without
silencing, the switch damages the other skills, just like the post-hoc methods.

**5. Fix: tag the unlabeled target data with a probe on the model's own activations**
(`tagging.py`). In neuroscience, cells are tagged by a marker they express. Here, every
50 steps a linear probe is trained on the model's own embedding activations to separate
labeled-ADD rows from everything else. Its scores get the positive-unlabeled correction
of Elkan & Noto (2008), which accounts for hidden ADD rows in the unlabeled pool. Rows
the probe flags are routed into the switch; everything else trains as normal, with
silencing. The threshold is set to favor recall (0.2): a missed target example teaches
the rest of the network ADD, while a false tag costs little.

| ADD labeled | ADD after off ↓ | Others after off | ADD after relearning ↓ | Tag recall / precision |
|---|---|---|---|---|
| 5% | 0.12 ± 0.03 | 0.99 | 0.19 ± 0.03 | 1.00 / 0.48 |
| 10% | 0.10 ± 0.02 | 1.00 | 0.15 ± 0.03 | 1.00 / 0.83 |
| 25% | 0.10 ± 0.03 | 1.00 | 0.14 ± 0.03 | 1.00 / 1.00 |
| 50% | 0.09 ± 0.02 | 1.00 | 0.14 ± 0.04 | 1.00 / 1.00 |
| *full labels (for reference)* | *0.09* | *1.00* | *0.14* | – |

From 5% labels, the switch matches the fully labeled version on all three metrics. With
the switch on, every skill stays at 1.00.

**Noisy labels** (10% labeled, and a fifth of those labels are actually other tasks):
it holds in 3 of 4 seeds (ADD 0.10–0.12, others 0.99–1.00, relearning 0.15–0.17). In
seed 3 the model was weaker even with the switch on (ADD2 0.83), and switching off cost
up to 0.25 on ADD2.

Two approaches that failed:
- **Tagging by dependence** ("does silencing the region raise this example's loss?").
  At low label rates the region never learns ADD well enough to be relied on, so it
  tagged nothing (recall ≈ 0).
- **An L2-regularized probe.** It over-tagged and, at 5% labels, broke training.

## Takeaway

In this toy, the optogenetics approach works from start to finish. Install a switch
during training, tag the target data with a probe on the model's own activations,
silence the switch during other training, and you get a switch that is:
- **complete:** ADD falls to about 0.10, vs the oracle's 0.04;
- **specific:** zero damage to near-neighbor skills, where post-hoc methods lose 33–69
  points;
- **durable:** ADD relearns to about 0.15, vs 0.24–0.63 for post-hoc;
- **free:** no accuracy cost while the switch is on;
- **cheap to label:** it needs only 5% of the target data labeled.

## What would make this a real result

1. **The probe's job here is easy.** The task token gives ADD away to a linear probe. In
   a language model the target (a language, a topic, a dangerous capability) is fuzzier.
   The next test is a small transformer language model with a natural target and a few
   percent of it labeled, comparing tag recall and switch quality against SAE-feature
   ablation.
2. **Scale and architecture:** attention layers, residual-stream routing, many switches
   at once.
3. **Adversarial durability:** larger fine-tuning budgets, and attacks that try to route
   ADD around the switch.

## Caveats

- This is a toy: one small MLP, synthetic tasks, and a 31-number modular-arithmetic space.
- Post-hoc handles here are individual units. SAE features, which can be more
  monosemantic, were not tested.
- Gradient routing is existing work (Cloud et al., 2024), and so is positive-unlabeled
  learning (Elkan & Noto, 2008). What's new here is the head-to-head comparison against
  post-hoc handles, the separate test of each ingredient (silencing is the key one), the
  partial-label failure, and combining routing with online tagging by a probe on the
  model's own activations, which fixes that failure.

## Reproduce

```
pip install torch numpy matplotlib
for s in 0 1 2 3; do python opto.py --seed $s --out results/seed$s.json & done; wait
python analyze.py
for s in 0 1 2 3; do python label_frac.py --seed $s --out results/labelfrac_seed$s.json & done
for s in 0 1 2 3; do python tagging.py --seed $s --thresh 0.2 --out results/tagging_seed$s.json & done
for s in 0 1 2 3; do python tagging.py --seed $s --thresh 0.2 --fracs 0.1 --noise 0.2 --out results/tagging_noise_seed$s.json & done
```
On 4 CPU cores this takes about 40 minutes.
