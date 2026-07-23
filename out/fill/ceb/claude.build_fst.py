#!/usr/bin/env python3
"""Build a morphological-inflection FST for Cebuano (ceb) with PyFoma.

Strategy
--------
The held-out dev/test items are *unseen lemma+feature combinations* of lemmas
that all appear in training.  So the transducer must (a) cover every
lemma x feature cell, and (b) predict the cells it never saw.

We model each inflected form as   output = prefix + stem   where the stem is the
lemma with its leading modal prefix (mo-/ma-/mag-/mang-) stripped.  The prefix is
chosen from the morphological features, with a dynamic-vs-stative class decision
for ambiguous "ma-" verbs (a "ma-" lemma that shows any ni/nag/ning form in its
observed paradigm inflects like the dynamic "mo-" class).

For every (lemma, feature) cell we emit:
  * the gold form when it was observed in training (guaranteed correct), else
  * the rule-predicted form.

The whole cell inventory is compiled into one FST as a union of quoted-symbol
cross-products and minimized.  Each feature tag ([V], [PST], ...) and each
character is an atomic (quoted) symbol on both sides.
"""

import collections
from pyfoma import FST

TRAIN = "data/ceb.trn"
OUTFILE = "test.foma"

FEATS = ["V;FUT", "V;NFIN", "V;PST", "V;PRS", "V;PROG;PRS", "V;PRF;PST"]
LEM_PREFIXES = ["makahimo sa", "mahimo nga", "mang", "mag", "mo", "ma", "mi", "na", "pa"]
NONPAST = {"V;PST", "V;PRS", "V;PROG;PRS", "V;PRF;PST"}
VOWELS = set("aeiou")
# Prefix used by the productive dynamic ("mo-") class, per feature bundle.
DYN = {"V;FUT": "mo", "V;NFIN": "mo", "V;PST": "ni",
       "V;PRS": "nag", "V;PROG;PRS": "ning", "V;PRF;PST": "na"}


def strip_lem(lem):
    """Split a lemma into (modal prefix, stem)."""
    for p in LEM_PREFIXES:
        if lem.startswith(p) and len(lem) > len(p):
            return p, lem[len(p):]
    return "", lem


def load(path):
    rows = []
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        lem, inf, f = line.split("\t")
        rows.append((lem, inf, f))
    return rows


def build_tables(rows):
    """Majority output-prefix tables and per-lemma gold paradigms."""
    bylem = collections.defaultdict(dict)
    for lem, inf, f in rows:
        bylem[lem][f] = inf
    tab = collections.defaultdict(collections.Counter)   # (lem_prefix, feat) -> prefixes
    gtab = collections.defaultdict(collections.Counter)  # feat -> prefixes
    for lem, d in bylem.items():
        lp, stem = strip_lem(lem)
        for f, inf in d.items():
            if inf.endswith(stem):
                op = inf[:len(inf) - len(stem)].rstrip("-")
                tab[(lp, f)][op] += 1
                gtab[f][op] += 1
    maj = {k: c.most_common(1)[0][0] for k, c in tab.items()}
    gmaj = {k: c.most_common(1)[0][0] for k, c in gtab.items()}
    return bylem, maj, gmaj


def known_prefixes(lem, bylem):
    lp, stem = strip_lem(lem)
    out = set()
    for f, inf in bylem.get(lem, {}).items():
        if inf.endswith(stem):
            out.add(inf[:len(inf) - len(stem)].rstrip("-"))
    return out


def predict(lem, f, bylem, maj, gmaj):
    """Predict the inflected form for an unseen (lemma, feature) cell."""
    lp, stem = strip_lem(lem)

    # Lemmas with no recognizable modal prefix are invariant function words
    # (e.g. "aduna", "daw"): every cell echoes the lemma.
    if lp == "":
        return lem

    # Abilitative "maka-" class: every observed form is maka-/naka- + fixed stem.
    # These take maka- for FUT/NFIN and naka- for the tense/aspect forms.
    if lem.startswith("maka"):
        ab = lem[4:]
        forms = list(bylem.get(lem, {}).values())
        if forms and all(v.endswith(ab) and v[:len(v) - len(ab)] in ("maka", "naka")
                         for v in forms):
            return ("maka" if f not in NONPAST else "naka") + ab

    kp = known_prefixes(lem, bylem)
    # A "ma-" lemma that shows any dynamic (ni/nag/ning) form inflects its tense/
    # aspect cells like the productive dynamic "mo-" class.
    dynamic = (lp == "mo") or (lp == "ma" and bool(kp & {"ni", "nag", "ning"}))
    if f in NONPAST and dynamic:
        pref = DYN[f]
    else:
        # FUT/NFIN and non-dynamic cells follow the (modal-prefix, feature) majority.
        pref = maj.get((lp, f), gmaj.get(f, ""))
    # Orthographic hyphen before vowel-initial stems for nag-/ning-, and ni- + i.
    if stem and stem[0] in VOWELS:
        if pref in ("nag", "ning"):
            pref += "-"
        elif pref == "ni" and stem[0] == "i":
            pref = "ni-"
    return pref + stem


def toks(feat_bundle, s):
    """Tokenize into atomic symbols: feature tags first, then characters."""
    return ["[%s]" % x for x in feat_bundle.split(";")] + list(s)


def side(toklist):
    return "(" + " ".join("'" + t + "'" for t in toklist) + ")"


def main():
    rows = load(TRAIN)
    bylem, maj, gmaj = build_tables(rows)
    gold = {(lem, f): inf for lem, inf, f in rows}

    # One cell per (lemma, feature): gold if seen, else rule prediction.
    cells = {}
    for lem in bylem:
        for f in FEATS:
            cells[(lem, f)] = gold.get((lem, f)) or predict(lem, f, bylem, maj, gmaj)

    branches = []
    for (lem, f), out in cells.items():
        branches.append("(" + side(toks(f, lem)) + ":" + side(toks(f, out)) + ")")
    regex = " | ".join(branches)

    fst = FST.re(regex)
    fst = fst.minimize()

    # Sanity: every input generates exactly its intended output.
    bad = 0
    for (lem, f), out in cells.items():
        got = list(fst.generate(toks(f, lem)))
        if got != ["".join(toks(f, out))]:
            bad += 1
            if bad <= 5:
                print("MISMATCH", lem, f, "->", got, "expected", out)
    print("cells:", len(cells), "generation mismatches:", bad)
    print("states:", len(list(fst.states)))

    foma_str = fst.to_fomastring()
    with open(OUTFILE, "w") as fh:
        fh.write(foma_str)
    print("wrote", OUTFILE, "(%d bytes)" % len(foma_str))


if __name__ == "__main__":
    main()
