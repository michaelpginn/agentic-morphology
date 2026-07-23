#!/usr/bin/env python3
"""Build an FST for Malagasy verb inflection.

The morphology is perfectly regular in the training data: the inflected form is
the lemma with a tense-conditioned prefix:

    PRS  -> m + lemma
    PST  -> n + lemma
    FUT  -> h + lemma
    NFIN ->     lemma   (no prefix)

Input format (features first, every symbol quoted as an atomic unit):
    '[V]''[PRS]''i''t''s''o''k''a'
Output repeats the feature tags, then emits each character of the inflected form:
    '[V]''[PRS]''m''i''t''s''o''k''a'
"""

from pyfoma import FST

# Alphabet of stem/wordform characters seen in the data.
ALPHABET = list("abdefghijklmnoprstvyzà")

# Tense tag -> inserted prefix ('' for no prefix).
TENSE_PREFIX = {
    "[PRS]": "m",
    "[PST]": "n",
    "[FUT]": "h",
    "[NFIN]": "",
}


def q(sym):
    """Quote a symbol so PyFoma treats it as one atomic input symbol."""
    return "'" + sym + "'"


# Feature-first prefix: copy the [V] tag, then copy the tense tag and insert its
# prefix character on the output side.
tense_branches = []
for tag, pref in TENSE_PREFIX.items():
    branch = q(tag)
    if pref:
        branch += " '':" + q(pref)
    tense_branches.append(branch)

tense_re = "(" + " | ".join(tense_branches) + ")"

# Identity copy over the alphabet, repeated for the whole stem.
copy_re = "(" + " | ".join(q(c) for c in ALPHABET) + ")*"

grammar_re = q("[V]") + " " + tense_re + " " + copy_re

fst = FST.re(grammar_re)
fst = fst.minimize()


def make_input(lemma, feat):
    parts = feat.split(";")  # e.g. ['V', 'PRS']
    tags = "".join("[" + p + "]" for p in parts)
    return tags + lemma


def expected_output(lemma, inflected, feat):
    parts = feat.split(";")
    tags = "".join("[" + p + "]" for p in parts)
    return tags + inflected


if __name__ == "__main__":
    # Self-test against the training data.
    total = 0
    correct = 0
    for line in open("/workspace/data/mlg.trn"):
        lemma, inflected, feat = line.rstrip("\n").split("\t")
        inp = make_input(lemma, feat)
        exp = expected_output(lemma, inflected, feat)
        outs = set(fst.apply(inp))
        total += 1
        if exp in outs:
            correct += 1
        else:
            print("MISS", inp, "->", outs, "expected", exp)
    print(f"train exact-match: {correct}/{total}")
    print(f"states: {len(fst.states)}")

    fomastring = fst.to_fomastring()
    with open("/workspace/test.foma", "w") as fh:
        fh.write(fomastring)
    print("saved test.foma")
