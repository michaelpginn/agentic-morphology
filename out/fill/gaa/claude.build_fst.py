#!/usr/bin/env python3
"""Build a morphological generation FST for Ga (gaa) verb inflection.

Input format:  feature tags (each bracketed, e.g. [V][HAB][PRS]) followed by the
lemma characters.  Every symbol (tag or character) is an atomic, quoted symbol.
Output format: the same feature tags followed by the inflected wordform characters.

Analysis of /workspace/data/gaa.trn shows that, for every one of the nine feature
combinations present, the mapping lemma -> wordform is a fully regular affixation
(0 exceptions across 607 training rows):

    V;HAB;FUT      prefix  'baa'
    V;PRS          prefix  'ŋ'
    V;HAB;NEG;PRS  suffix  'ko'
    V;NEG;PRS      suffix  'ee'
    V;HAB;NEG;FUT  suffix  'ee'
    V;HAB;NEG;PST  suffix  'ee'
    V;HAB;PRS      identity
    V;HAB;PST      identity
    V;NFIN         identity

So the grammar is: emit the feature tags unchanged, apply the (prefix|suffix)
affix as an insertion, and copy the lemma characters through unchanged.  A single
shared copy machine over the alphabet keeps the transducer compact.
"""

from pyfoma import FST

# ---------------------------------------------------------------------------
# Lexical inventory (derived from the training data)
# ---------------------------------------------------------------------------
# Alphabet of characters that may appear in a lemma / wordform.
ALPHABET = [' ', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 's', 't', 'u', 'w', 'y', 'ŋ', 'ɔ', 'ɛ']

# One productive rule per feature combination:  (feature-list, prefix, suffix)
RULES = [
    (['V', 'HAB', 'FUT'],        'baa', ''),
    (['V', 'PRS'],               'ŋ',   ''),
    (['V', 'HAB', 'NEG', 'PRS'], '',    'ko'),
    (['V', 'NEG', 'PRS'],        '',    'ee'),
    (['V', 'HAB', 'NEG', 'FUT'], '',    'ee'),
    (['V', 'HAB', 'NEG', 'PST'], '',    'ee'),
    (['V', 'HAB', 'PRS'],        '',    ''),
    (['V', 'HAB', 'PST'],        '',    ''),
    (['V', 'NFIN'],              '',    ''),
]


def q(sym):
    """Quote a symbol so PyFoma treats it as a single atomic symbol."""
    return "'" + sym + "'"


def identity(sym):
    """Regex for an identity mapping of one quoted symbol."""
    return q(sym)


def insert(chars):
    """Regex that inserts (epsilon -> chars) each character individually."""
    return " ".join("'':" + q(c) for c in chars)


# Shared copy machine: identity over any sequence of alphabet characters.
copy = "(" + " | ".join(identity(c) for c in ALPHABET) + ")*"

branches = []
for feats, prefix, suffix in RULES:
    tags = " ".join(identity('[' + f + ']') for f in feats)
    parts = [tags]
    if prefix:
        parts.append(insert(prefix))
    parts.append(copy)
    if suffix:
        parts.append(insert(suffix))
    branches.append("(" + " ".join(parts) + ")")

regex = " | ".join(branches)

fst = FST.re(regex)

# ---------------------------------------------------------------------------
# Sanity check against the full training set
# ---------------------------------------------------------------------------
def build_input(lemma, feat):
    return "".join('[' + f + ']' for f in feat.split(';')) + lemma


def expected_output(form, feat):
    return "".join('[' + f + ']' for f in feat.split(';')) + form


if __name__ == "__main__":
    rows = [l.rstrip('\n').split('\t') for l in open('/workspace/data/gaa.trn')]
    ok = bad = 0
    fails = []
    for lemma, form, feat in rows:
        inp = build_input(lemma, feat)
        exp = expected_output(form, feat)
        preds = list(fst.generate(inp))
        if preds == [exp]:
            ok += 1
        else:
            bad += 1
            if len(fails) < 20:
                fails.append((inp, exp, preds))
    print(f"Training exact-match: {ok}/{ok + bad}")
    for inp, exp, preds in fails:
        print("  FAIL", repr(inp), "exp", repr(exp), "got", preds)

    print(f"States: {len(fst.states)}")

    fomastring = fst.to_fomastring()
    with open('/workspace/test.foma', 'w') as fh:
        fh.write(fomastring)
    print("Wrote /workspace/test.foma")
