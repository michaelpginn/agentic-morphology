#!/usr/bin/env python3
"""Build an FST for Malagasy (mlg) verbal inflection using PyFoma.

Input format:  feature tags (each bracketed) followed by lemma characters,
each token quoted as an atomic symbol, e.g. '[V]''[PRS]''i''t''s''o''k''a'.
Output format: the same feature tags echoed, followed by the inflected
wordform's characters, e.g. '[V]''[PRS]''m''i''t''s''o''k''a'.

The inflection is a simple tense-driven prefixation:
    PRS  -> prepend 'm'
    PST  -> prepend 'n'
    FUT  -> prepend 'h'
    NFIN -> identity (no prefix)
"""

from pyfoma import FST

# Alphabet of stem characters observed in the training data.
CHARS = list('abdefghijklmnoprstvyzà')


def build():
    # Every stem character maps to itself; quoting makes each an atomic symbol.
    chardef = '|'.join("'" + c + "'" for c in CHARS)

    # [V] is echoed; the tense tag is echoed and inserts its prefix consonant;
    # then the stem is copied verbatim.
    regex = (
        "'[V]' "
        "('[PRS]' '':'m' | '[NFIN]' | '[PST]' '':'n' | '[FUT]' '':'h') "
        "(" + chardef + ")*"
    )

    fst = FST.re(regex)
    fst = fst.minimize()
    return fst


def main():
    fst = build()

    fomastring = fst.to_fomastring()
    with open('test.foma', 'w') as f:
        f.write(fomastring)

    print(f"FST built with {len(fst.states)} states; saved to test.foma")


if __name__ == '__main__':
    main()
