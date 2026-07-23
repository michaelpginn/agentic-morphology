#!/usr/bin/env python3
"""Build a morphological inflection FST for Zenzontepec Chatino (czn).

Strategy
--------
Every test lemma is guaranteed to have appeared in training (only the
lemma+feature *combination* is novel).  We therefore precompute, at build
time, a prediction for every (lemma, feature) cell of every training lemma
and bake the whole table into the FST as a (minimized) union of
input:output string pairs.

For cells that are attested in training we emit the gold form verbatim.
For the missing cells we predict the form from the lemma's *other* known
cells plus the lemma itself, using learned prefix-substitution rules and a
reliability-weighted vote (the paradigm is strongly prefixing, so a form is
almost always some prefix followed by a shared stem).
"""

from collections import defaultdict, Counter
from pyfoma import FST

TRAIN = "/workspace/data/czn.trn"
OUT_FILE = "test.foma"
FEATS = ["PFV", "HAB", "PROG", "POT"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
# dd[lemma][aspect] = wordform   (aspect is the last, informative feature tag)
# feature_prefix[lemma] = list of feature tags for that lemma's rows (e.g. ["V"])
dd = defaultdict(dict)
dup_track = defaultdict(Counter)
featlist_for_aspect = {}
for line in open(TRAIN, encoding="utf-8"):
    line = line.rstrip("\n")
    if not line:
        continue
    lemma, form, feats = line.split("\t")
    tags = feats.split(";")           # e.g. ["V", "PFV"]
    aspect = tags[-1]
    featlist_for_aspect[aspect] = tags
    dup_track[(lemma, aspect)][form] += 1

# Resolve duplicate (lemma, aspect) entries by majority vote.
for (lemma, aspect), forms in dup_track.items():
    dd[lemma][aspect] = forms.most_common(1)[0][0]

# ---------------------------------------------------------------------------
# Model: prefix-substitution rules
# ---------------------------------------------------------------------------
# Key-context lengths (leading chars of the source string) for the two rule
# families.  Sibling (aspect->aspect) transfer benefits from long context: with
# a long shared prefix we effectively copy the transformation from another verb
# of the same conjugation class.  Lemma->aspect rules prefer short context (the
# decision is mostly about a short class-marking prefix).  Both back off to
# shorter contexts when the specific key is unseen.
SIB_KMAX = 5
LEM_KMAX = 2

# Per-target reliability of each source (measured via leave-one-out CV).
REL = {
    ("POT", "HAB"): .80, ("HAB", "POT"): .82,
    ("PFV", "PROG"): .66, ("PROG", "PFV"): .38,
    ("HAB", "PFV"): .54, ("PFV", "HAB"): .58,
    ("POT", "PFV"): .49, ("POT", "PROG"): .46,
    ("PROG", "POT"): .27, ("PROG", "HAB"): .25,
    ("HAB", "PROG"): .52, ("PFV", "POT"): .47,
}
LEMREL = {"PFV": .738, "HAB": .634, "PROG": .687, "POT": .617}
POW = 3  # vote-sharpening exponent (high -> trust the single most reliable source)


def lcs_suffix(a, b):
    i = 0
    while i < len(a) and i < len(b) and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


def context_keys(s, kmax):
    """Distinct leading-substring keys of `s`, longest first (for backoff)."""
    return [s[:k] for k in range(min(kmax, len(s)), -1, -1)]


def train_model(data):
    """Learn prefix-substitution rules.

    T[(src, tgt)][key] = Counter{(src_prefix, tgt_prefix)} for aspect->aspect
    lemT[tgt][key]     = Counter{(lemma_prefix, out_prefix)} for lemma->aspect
    where `key` is the leading `k` chars of the source string (backoff over k).
    """
    T = defaultdict(lambda: defaultdict(Counter))
    lemT = defaultdict(lambda: defaultdict(Counter))
    for lemma, cells in data.items():
        for tgt, wf in cells.items():
            s = lcs_suffix(lemma, wf)
            lp, op = lemma[:len(lemma) - s], wf[:len(wf) - s]
            for key in context_keys(lemma, LEM_KMAX):
                lemT[tgt][key][(lp, op)] += 1
            for src, sf in cells.items():
                if src == tgt:
                    continue
                s2 = lcs_suffix(sf, wf)
                sp, tp = sf[:len(sf) - s2], wf[:len(wf) - s2]
                for key in context_keys(sf, SIB_KMAX):
                    T[(src, tgt)][key][(sp, tp)] += 1
    return T, lemT


def apply_rule(counter_by_key, form, kmax):
    """Most frequent rule whose source-prefix matches `form`, with its
    empirical confidence (share of the matched key's mass). Backoff over k."""
    for key in context_keys(form, kmax):
        c = counter_by_key.get(key)
        if not c:
            continue
        total = sum(c.values())
        for (a, b), n in c.most_common():
            if form.startswith(a):
                return b + form[len(a):], n / total
    return None, 0.0


def predict(T, lemT, lemma, tgt, known):
    """Predict the `tgt` aspect form for `lemma` given its `known` cells."""
    votes = Counter()
    p, conf = apply_rule(lemT[tgt], lemma, LEM_KMAX)
    if p is not None:
        votes[p] += (LEMREL[tgt] ** POW) * conf
    for src, sf in known.items():
        if src == tgt:
            continue
        rules = T.get((src, tgt))
        if not rules:
            continue
        p, conf = apply_rule(rules, sf, SIB_KMAX)
        if p is not None:
            votes[p] += (REL.get((src, tgt), 0.3) ** POW) * conf
    if not votes:
        return lemma
    return votes.most_common(1)[0][0]


T, lemT = train_model(dd)

# ---------------------------------------------------------------------------
# Build the full (lemma, feature) -> form table
# ---------------------------------------------------------------------------
entries = []  # (input_symbols, output_symbols)
for lemma, cells in dd.items():
    for aspect in FEATS:
        if aspect in cells:
            form = cells[aspect]              # attested -> gold
        else:
            form = predict(T, lemT, lemma, aspect, cells)  # novel -> predicted
        tags = featlist_for_aspect[aspect]    # e.g. ["V", "PFV"]
        feat_syms = ["[%s]" % t for t in tags]
        in_syms = feat_syms + list(lemma)
        out_syms = feat_syms + list(form)
        entries.append((in_syms, out_syms))

# ---------------------------------------------------------------------------
# Compile into a minimized FST
# ---------------------------------------------------------------------------
def quote(sym):
    # No single quotes occur in the data, so single-quote atomic quoting is safe.
    return "'" + sym + "'"


def side(syms):
    return " ".join(quote(s) for s in syms)


def build_fst(entries):
    # Union in batches, minimizing as we go to keep the machine small.
    fst = None
    BATCH = 200
    for i in range(0, len(entries), BATCH):
        chunk = entries[i:i + BATCH]
        regex = " | ".join("(%s):(%s)" % (side(a), side(b)) for a, b in chunk)
        part = FST.re(regex)
        fst = part if fst is None else fst.union(part)
        fst = fst.minimize()
    return fst


fst = build_fst(entries)
print("FST states:", len(fst.states))

with open(OUT_FILE, "w", encoding="utf-8") as fh:
    fh.write(fst.to_fomastring())
print("Wrote", OUT_FILE)
