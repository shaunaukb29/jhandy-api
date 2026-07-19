"""Train the recommended clean-teacher model + honest held-out eval.

The iteration research (docs/iteration-research.md) showed clean verified labels
beat weak scrape labels decisively on the SAME features. This trains the
deployable model that recommendation implies: clean teachers on a grouped-train
slice of verified data, evaluated on a leakage-safe held-out verified slice that
training never sees.

Heads (all linear on frozen embeddings):
    kind  fault-vs-normal   (all verified, grouped by source+vehicle)
    knock knock-vs-normal   (Car-Engine, grouped by vehicle)
    cause 6-class           (car-diagnostics; clip-grouped, caveat in doc)

Held-out = 25% of groups per task (deterministic). Reports test metrics with
bootstrap CIs and saves heads for inference.

Every head is checked for statistical signal above chance, not just `kind`.
A head trained on a handful of clips can look confident (LogisticRegression
will happily assign high probabilities) while its cross-validated balanced
accuracy is indistinguishable from guessing — this is exactly what happened
to `cause` in early runs (36 clips / 6 classes / 7 source videos), and it
went undetected because only `kind` was ever checked. The `weak_signal`
verdict computed here is saved into the joblib artifact itself so
`Classifier` can refuse to present a weak head's output as if it meant
something, instead of relying on a human noticing a report file.

Usage:
    uv run training/models/train_best.py --emb clap_embeddings.npz
"""
import argparse
import hashlib
import json
import re
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from cardiag import paths

DATA = paths.TRAIN_DATA
RNG = np.random.default_rng(0)
CD = {"brakes", "belt", "power_steering", "accessories", "fuel_ignition",
      "low_oil"}

# A head is flagged "weak_signal" when its cross-validated balanced accuracy
# is not clearly separated from chance (1/n_classes) — specifically when it's
# within WEAK_SIGNAL_SIGMA standard deviations of chance across folds. This
# is the same idea the old code applied only to `kind`; now every head gets it.
WEAK_SIGNAL_SIGMA = 2.0
MIN_EXAMPLES_PER_CLASS = 15   # below this, flag as weak regardless of accuracy


def held_out(group):
    """25% of groups -> test, deterministic by group hash."""
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % 4 == 0


def ce_vehicle(r):
    n = r["id"].split(":", 2)[-1].lower()
    n = re.sub(r"(engine )?(knock\w*|tick\w*|click\w*|rattl\w*|crank\w*|sound|"
               r"noise|normal|idle).*", "", n)
    return "ce:" + re.sub(r"[^a-z0-9]+", " ", n).strip()[:25]


def src_vehicle(r):
    return r["source"] + ":" + re.sub(r"[^a-z0-9]", "",
                                      r["id"].split(":")[-1].lower())[:18]


def boot_ci(correct, b=2000):
    correct = np.asarray(correct, float)
    if not len(correct):
        return [None, None]
    m = [correct[RNG.integers(0, len(correct), len(correct))].mean()
         for _ in range(b)]
    return [round(float(np.percentile(m, 2.5)), 3),
            round(float(np.percentile(m, 97.5)), 3)]


def _balanced_acc(y_true, y_pred, classes):
    """Balanced accuracy: mean of per-class recall. Robust to class imbalance,
    unlike plain accuracy (which a majority-class head can win at trivially)."""
    recalls = []
    for c in classes:
        mask = [t == c for t in y_true]
        n = sum(mask)
        if n == 0:
            continue
        correct = sum(p == c for p, t in zip(y_pred, y_true) if t == c)
        recalls.append(correct / n)
    return float(np.mean(recalls)) if recalls else 0.0


def _cv_weak_signal(rows, labelf, emb, n_splits=10):
    """Cross-validated balanced accuracy vs. chance, for ANY head.

    Small datasets (the exact situation `cause` was trained on: 36 clips) make
    a single train/test split noisy — one lucky or unlucky split can look like
    real signal or real failure. K-fold CV averages over multiple splits for a
    more honest read of whether the head generalizes at all.

    Returns (mean_bal_acc, std_bal_acc, degenerate, weak_signal_message_or_None).
    """
    y = [labelf(r) for r in rows]
    classes = sorted(set(y))
    n_classes = len(classes)
    n = len(rows)

    if n_classes < 2 or n < n_classes * 2:
        return 0.0, 0.0, True, "too few examples/classes to train or evaluate"

    X = np.array([emb[r["id"]] for r in rows])
    y = np.array(y)
    chance = 1.0 / n_classes

    class_counts = Counter(y.tolist())
    thin_classes = [c for c, cnt in class_counts.items() if cnt < MIN_EXAMPLES_PER_CLASS]

    n_splits = max(2, min(n_splits, min(class_counts.values())))
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        fold_accs = []
        for tr_idx, te_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=3000, class_weight="balanced")
            clf.fit(X[tr_idx], y[tr_idx])
            pred = clf.predict(X[te_idx])
            fold_accs.append(_balanced_acc(y[te_idx].tolist(), pred.tolist(), classes))
        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
    except ValueError as e:
        return 0.0, 0.0, True, f"cross-validation failed: {e}"

    reasons = []
    if mean_acc - WEAK_SIGNAL_SIGMA * std_acc <= chance:
        reasons.append(
            f"balanced accuracy ({mean_acc:.3f} ± {std_acc:.3f}) is within "
            f"{WEAK_SIGNAL_SIGMA}σ of chance ({chance:.3f}) for {n_classes} classes"
        )
    if thin_classes:
        reasons.append(
            f"class(es) {thin_classes} have fewer than {MIN_EXAMPLES_PER_CLASS} "
            f"examples — too thin to trust regardless of accuracy"
        )

    weak_msg = ("; ".join(reasons) + " — needs more/cleaner data") if reasons else None
    return mean_acc, std_acc, False, weak_msg


def train_eval(rows, labelf, groupf, emb, name):
    rows = [r for r in rows if r["id"] in emb and labelf(r)]
    tr = [r for r in rows if not held_out(groupf(r))]
    te = [r for r in rows if held_out(groupf(r))]
    gtr, gte = {groupf(r) for r in tr}, {groupf(r) for r in te}
    assert not (gtr & gte), "group leakage!"
    Xtr = np.array([emb[r["id"]] for r in tr])
    ytr = [labelf(r) for r in tr]
    Xte = np.array([emb[r["id"]] for r in te])
    yte = [labelf(r) for r in te]
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr,
                                                                         ytr)
    pred = clf.predict(Xte)
    corr = [p == g for p, g in zip(pred, yte)]

    n_videos = len({src_vehicle(r) if "source" in r else groupf(r) for r in rows})
    cv_bal_acc, cv_bal_acc_std, degenerate, weak_signal = _cv_weak_signal(
        rows, labelf, emb)

    out = {"name": name, "n_train": len(tr), "n_test": len(te),
           "train_groups": len(gtr), "test_groups": len(gte),
           "classes": sorted(set(ytr) | set(yte)),
           "n": len(rows), "n_videos": n_videos,
           "acc": round(float(np.mean(corr)), 3), "acc_ci": boot_ci(corr),
           "majority_acc": round(max(Counter(yte).values()) / len(yte), 3),
           "cv_bal_acc": round(cv_bal_acc, 3),
           "cv_bal_acc_std": round(cv_bal_acc_std, 3),
           "degenerate": degenerate}
    if weak_signal:
        out["weak_signal"] = weak_signal
    kn = [p == "knock" for p, g in zip(pred, yte) if g == "knock"]
    if kn:
        out["knock_recall"] = round(float(np.mean(kn)), 3)
    return clf, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="clap_embeddings.npz")
    args = ap.parse_args()
    z = np.load(DATA / args.emb, allow_pickle=True)
    emb = {i: v / (np.linalg.norm(v) + 1e-9) for i, v in zip(z["ids"], z["X"])}
    ext = [json.loads(l) for l in open(DATA / "external_eval.jsonl")]

    heads, report = {}, {"emb": args.emb}
    heads["kind"], report["kind"] = train_eval(
        [r for r in ext if r["kind"] in ("fault", "normal")],
        lambda r: r["kind"], src_vehicle, emb, "fault_vs_normal")
    heads["knock"], report["knock"] = train_eval(
        [r for r in ext if r["source"] == "ext:car-engine-sounds"
         and r.get("l1") in ("knock", "normal_idle")],
        lambda r: r["l1"], ce_vehicle, emb, "knock_vs_normal")
    heads["cause"], report["cause"] = train_eval(
        [r for r in ext if r["source"] == "ext:car-diagnostics"],
        lambda r: r["cause"] if r.get("cause") in CD else None,
        lambda r: r["id"], emb, "cause_6class")

    # Persist the weak-signal verdict INTO the artifact itself, not just the
    # report file — so Classifier can act on it at serve time even if nobody
    # reads train_report.json before deploying.
    weak = {name: bool(rep.get("weak_signal") or rep.get("degenerate"))
            for name, rep in report.items() if isinstance(rep, dict)}

    joblib.dump({"heads": heads, "emb": args.emb, "weak": weak},
                DATA / f"best_model_{args.emb.split('_')[0]}.joblib")
    json.dump(report, open(DATA / "iterations" /
                           f"best_model_{args.emb.split('_')[0]}.json", "w"),
              indent=1)
    print(json.dumps(report, indent=1))
    for name, is_weak in weak.items():
        if is_weak:
            print(f"WARNING: head '{name}' has weak/no statistical signal — "
                  f"its output will be downgraded at serve time. See report above.")


if __name__ == "__main__":
    main()