#!/usr/bin/env python3
"""Build a morphological-inflection FST for Oromo (orm) using PyFoma.

Input  format: quoted feature tags (each wrapped in square brackets) followed by
               the quoted characters of the lemma, e.g. '[V]''[SG]''[1]''[PST]''m''u''r'
Output format: the same feature tags echoed back, followed by the quoted
               characters of the inflected wordform,          e.g. '[V]''[SG]''[1]''[PST]''m''u''r''d''h''e''e'

Strategy
--------
For (almost) every training row the wordform equals   [ni ] + lemma + suffix.
 * The optional "ni " prefix and the suffix are fully determined by the feature
   combination for every non-present / non-infinitive cell (one suffix per cell).
 * PRESENT cells have two allomorphs -- a "ch"-class subset of lemmas takes an
   extra "ch" before the "aa jir..." progressive suffix. This class is lexical,
   but every lemma is attested in PRESENT cells in the training data, so we read
   each lemma's class off the data.
 * INFINITIVE (NFIN) has three allomorphs: "chuu" (ch-class), "" (vowel-final
   stems that are already citation forms) and "uu" (default, consonant-final).

Because the task only requires generalisation to unseen lemma+feature
combinations (never to unseen lemmas), reading the ch-class per lemma is safe.
The suffix per cell is emitted with a plain identity stem (Sigma_char+), so the
resulting transducer is very compact.
"""

import collections
from pyfoma import FST

TRAIN = "/workspace/data/orm.trn"
OUT = "test.foma"
VOWELS = set("aeiou")


# ---------------------------------------------------------------------------
# 1. Read the data
# ---------------------------------------------------------------------------
rows = []
chars = set()
with open(TRAIN) as fh:
    for line in fh:
        line = line.rstrip("\n")
        if not line:
            continue
        lemma, wf, feat = line.split("\t")
        feats = feat.split(";")
        rows.append((lemma, wf, feats))
        chars |= set(lemma)
        chars |= set(wf)

chars = sorted(chars)


def split(lemma, wf):
    """Return (prefix, suffix) such that wf == prefix + lemma + suffix, else None."""
    core, pre = wf, ""
    if wf.startswith("ni "):
        core, pre = wf[3:], "ni "
    if core.startswith(lemma):
        return pre, core[len(lemma):]
    return None


# ---------------------------------------------------------------------------
# 2. Learn the ch-class of every lemma (majority vote over PRESENT cells)
# ---------------------------------------------------------------------------
prs_vote = collections.defaultdict(collections.Counter)
for lemma, wf, feats in rows:
    if "PRS" in feats:
        sp = split(lemma, wf)
        if sp:
            prs_vote[lemma]["ch" if sp[1].startswith("ch") else "no"] += 1

ch_class = {lem for lem, c in prs_vote.items() if c["ch"] > c["no"]}

# NFIN "chuu" set = ch-class plus any lemma actually attested with a chuu NFIN.
nfin_chuu = set(ch_class)
for lemma, wf, feats in rows:
    if "NFIN" in feats:
        sp = split(lemma, wf)
        if sp and sp[1] == "chuu":
            nfin_chuu.add(lemma)


# ---------------------------------------------------------------------------
# 3. Learn the (prefix, suffix) per feature cell (majority vote)
# ---------------------------------------------------------------------------
# key = tuple of features after the leading POS ("V")
cell_vote = collections.defaultdict(collections.Counter)
prs_base = {}      # cell -> non-ch base suffix (prefix, suffix)
for lemma, wf, feats in rows:
    cell = tuple(feats[1:])
    sp = split(lemma, wf)
    if sp:
        cell_vote[cell][sp] += 1

for cell, votes in cell_vote.items():
    if "PRS" in cell:
        # base = most common suffix that does NOT start with "ch"
        for (pre, suf), _ in votes.most_common():
            if not suf.startswith("ch"):
                prs_base[cell] = (pre, suf)
                break


# ---------------------------------------------------------------------------
# 4. Regex helpers
# ---------------------------------------------------------------------------
def q(c):
    """Quote a single character as an atomic PyFoma symbol."""
    if c == "'":
        return "'\\''"
    return "'%s'" % c


def word_id(s):
    """Identity path that reads/echoes the character sequence of string s."""
    return " ".join(q(c) for c in s)


def emit(s):
    """Output-only path (input epsilon) producing the characters of string s."""
    if s == "":
        return None
    return " ".join("'':%s" % q(c) for c in s)


def feat_echo(cell):
    """Identity echo of the feature tags of a cell, with an optional leading [V]."""
    parts = ["('[V]')?"]
    parts += ["'[%s]'" % f for f in cell]
    return " ".join(parts)


# defined sub-networks referenced from the fragment regexes
AC = "(" + "|".join(q(c) for c in chars) + ")"
VOW = "(" + "|".join(q(c) for c in "aeiou") + ")"
defined = {
    "AC": FST.re(AC),
    "VOW": FST.re(VOW),
    "CH": FST.re("|".join(word_id(l) for l in sorted(ch_class))),
    "CHUU": FST.re("|".join(word_id(l) for l in sorted(nfin_chuu))),
}
# CONS = any character that is not a vowel
defined["CONS"] = FST.re("$AC - $VOW", defined=defined)


def concat(*pieces):
    return " ".join(p for p in pieces if p)


# ---------------------------------------------------------------------------
# 5. Build one transducer fragment per feature cell
# ---------------------------------------------------------------------------
fragments = []
for cell in sorted(cell_vote):
    fe = feat_echo(cell)

    if cell == ("NFIN",):
        # three lexical allomorphs, partitioning all non-empty stems
        chuu = concat(fe, "($CHUU)", emit("chuu"))
        vow = concat(fe, "(($AC* $VOW) - $CHUU)")            # vowel-final -> ""
        cons = concat(fe, "(($AC* $CONS) - $CHUU)", emit("uu"))
        fragments.append("(%s)" % chuu)
        fragments.append("(%s)" % vow)
        fragments.append("(%s)" % cons)
        continue

    if "PRS" in cell:
        pre, base = prs_base[cell]
        pre_e = emit(pre)
        # ch-class lemmas take "ch" + base ; everyone else takes base
        ch = concat(fe, pre_e, "($CH)", emit("ch" + base))
        nonch = concat(fe, pre_e, "($AC+ - $CH)", emit(base))
        fragments.append("(%s)" % ch)
        fragments.append("(%s)" % nonch)
        continue

    # regular cell: single (prefix, suffix)
    (pre, suf), _ = cell_vote[cell].most_common(1)[0]
    frag = concat(fe, emit(pre), "($AC+)", emit(suf))
    fragments.append("(%s)" % frag)


# ---------------------------------------------------------------------------
# 6. Union, minimise, save
# ---------------------------------------------------------------------------
big = " | ".join(fragments)
fst = FST.re(big, defined=defined)
fst = fst.minimize()

print("states:", len(list(fst.states)))
with open(OUT, "w") as fh:
    fh.write(fst.to_fomastring())
print("wrote", OUT)
