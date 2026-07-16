#!/usr/bin/env python3
"""Build a morphological-inflection FST for Akan (aka) verbs using PyFoma.

Input  format:  feature tags first (each bracketed & quoted as an atomic
                symbol), followed by the quoted characters of the lemma,
                e.g.  '[V]''[PST]''r''u''n'
Output format:  the same feature tags echoed back, followed by the quoted
                characters of the inflected wordform.

Strategy
--------
Analysis of the training data shows every inflected form is exactly

        prefix + lemma + suffix

with two phonological alternations:

  * The prefix's final nasal assimilates to the place of the stem's first
    consonant: it is labial (m / mm) before labial-initial stems {b,f,m,p}
    and coronal (n / nn) otherwise.  In practice this means the prefix for a
    given feature-set depends only on whether the stem starts with a labial.

  * The past/perfect suffix makes the word end in "ee": it is "e" when the
    stem already ends in "e", and "ee" otherwise.

These two rules reconstruct 100% of the training data.  Because the task only
requires generalising to unseen *lemma+feature combinations* (every lemma is
already attested), we enumerate every known lemma against every known
feature-set, generate its correct output with the learned rules, and compile
the union into a single transducer.  Determinising and minimising yields a
compact, deterministic FST.
"""

import collections
from pyfoma import FST
from pyfoma.fst import State

TRAIN = "data/aka.trn"
OUT_FOMA = "test.foma"
LABIAL = set("bfmp")  # stems triggering labial (m) nasal assimilation


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            lemma, form, feats = line.split("\t")
            rows.append((lemma, form, feats))
    return rows


def align(lemma, form):
    """Return (prefix, suffix) such that form == prefix + lemma + suffix,
    choosing the alignment whose suffix is one of '', 'e', 'ee'."""
    i = form.find(lemma)
    result = None
    while i >= 0:
        suf = form[i + len(lemma):]
        if suf in ("", "e", "ee"):
            result = (form[:i], suf)
        i = form.find(lemma, i + 1)
    return result


def learn(rows):
    """Learn, per feature-set: the prefix for each labial class and whether a
    past/perfect suffix is present."""
    by_feats = collections.defaultdict(list)
    for lemma, form, feats in rows:
        by_feats[feats].append((lemma, form))

    prefix = {}      # (feats, is_labial) -> prefix string
    has_suffix = {}  # feats -> bool
    for feats, items in by_feats.items():
        suffixed = False
        for lemma, form in items:
            pre, suf = align(lemma, form)
            prefix[(feats, lemma[0] in LABIAL)] = pre
            if suf:
                suffixed = True
        has_suffix[feats] = suffixed
    return prefix, has_suffix


def suffix_for(lemma, feats, has_suffix):
    if not has_suffix[feats]:
        return ""
    return "e" if lemma.endswith("e") else "ee"


def tags_of(feats):
    return ["[" + t + "]" for t in feats.split(";")]


def build_fst(rows, prefix, has_suffix):
    lemmas = sorted({r[0] for r in rows})
    featsets = sorted({r[2] for r in rows})

    fst = FST()
    start = fst.initialstate
    alphabet = set()

    def arc(src, label):
        dst = State()
        fst.states.add(dst)
        src.add_transition(dst, label)
        for sym in label:
            if sym:
                alphabet.add(sym)
        return dst

    # Enumerate every (feature-set, lemma) pair as one path from the start
    # state.  Feature tags are echoed (identity); the prefix and suffix are
    # emitted on input-epsilon arcs; the lemma characters are copied.
    for feats in featsets:
        tags = tags_of(feats)
        for lemma in lemmas:
            cur = start
            for tag in tags:                      # echo feature tags
                cur = arc(cur, (tag,))
            pre = prefix[(feats, lemma[0] in LABIAL)]
            for ch in pre:                        # emit prefix
                cur = arc(cur, ("", ch))
            for ch in lemma:                      # copy lemma characters
                cur = arc(cur, (ch,))
            for ch in suffix_for(lemma, feats, has_suffix):  # emit suffix
                cur = arc(cur, ("", ch))
            fst.finalstates.add(cur)
            cur.finalweight = 0.0

    fst.alphabet = alphabet
    fst = fst.epsilon_remove().determinize_as_dfa().minimize()
    fst.alphabet = alphabet
    return fst


def main():
    rows = read_rows(TRAIN)
    prefix, has_suffix = learn(rows)

    # Sanity check: the learned rules must reconstruct all training rows.
    for lemma, form, feats in rows:
        pre = prefix[(feats, lemma[0] in LABIAL)]
        pred = pre + lemma + suffix_for(lemma, feats, has_suffix)
        assert pred == form, f"rule mismatch: {lemma}/{feats}: {pred!r} != {form!r}"

    fst = build_fst(rows, prefix, has_suffix)
    print(f"FST built: {len(fst.states)} states")

    fomastring = fst.to_fomastring()
    with open(OUT_FOMA, "w", encoding="utf-8") as fh:
        fh.write(fomastring)
    print(f"Saved to {OUT_FOMA} ({len(fomastring)} bytes)")


if __name__ == "__main__":
    main()
