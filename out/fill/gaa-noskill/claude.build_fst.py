#!/usr/bin/env python3
"""Build a morphological inflection FST for Ga (gaa) using PyFoma.

Input format:  feature tags (bracketed, atomic) followed by the lemma's
characters, each character an atomic quoted symbol, e.g. '[V]''[HAB]''[FUT]''j''o''o'.
Output format: the same feature tags repeated, followed by the inflected
wordform's characters.

The training data shows a perfectly regular set of affix rules keyed on the
feature bundle:

    V;NFIN            identity
    V;PRS             prefix  ŋ
    V;HAB;PRS         identity
    V;HAB;PST         identity
    V;HAB;FUT         prefix  baa
    V;NEG;PRS         suffix  ee
    V;HAB;NEG;PRS     suffix  ko
    V;HAB;NEG;PST     suffix  ee
    V;HAB;NEG;FUT     suffix  ee
"""

from pyfoma import FST

# Alphabet of characters occurring in lemmas (includes the space used by the
# multiword verb "ye nkunim" and the special Ga graphemes ŋ ɔ ɛ).
ALPHABET = [' ', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 's', 't', 'u', 'w', 'y', 'ŋ', 'ɔ', 'ɛ']


def q(sym):
    """Quote a symbol so PyFoma treats it as a single atomic token."""
    return "'" + sym + "'"


# Sigma: any single alphabet character, copied identically (x -> x).
SIG = "(" + "|".join(q(c) for c in ALPHABET) + ")"
COPY = SIG + "*"


def insert(string):
    """Regex fragment that inserts each character of `string` (epsilon -> char)."""
    return " ".join("('':" + q(c) + ")" for c in string)


def feats(tags):
    """Regex fragment that reads back the feature tags identically."""
    return " ".join(q("[" + t + "]") for t in tags)


# (feature tags, lemma-transformation regex) for every attested bundle.
RULES = [
    (["V", "NFIN"],               COPY),
    (["V", "PRS"],                insert("ŋ") + " " + COPY),
    (["V", "HAB", "PRS"],         COPY),
    (["V", "HAB", "PST"],         COPY),
    (["V", "HAB", "FUT"],         insert("baa") + " " + COPY),
    (["V", "NEG", "PRS"],         COPY + " " + insert("ee")),
    (["V", "HAB", "NEG", "PRS"],  COPY + " " + insert("ko")),
    (["V", "HAB", "NEG", "PST"],  COPY + " " + insert("ee")),
    (["V", "HAB", "NEG", "FUT"],  COPY + " " + insert("ee")),
]

paths = ["(" + feats(tags) + " " + trans + ")" for tags, trans in RULES]
regex = "(" + " | ".join(paths) + ")"

fst = FST.re(regex)
fst = fst.minimize()

with open("test.foma", "w", encoding="utf-8") as fh:
    fh.write(fst.to_fomastring())

print("States:", len(list(fst.states)))
print("Wrote test.foma")
