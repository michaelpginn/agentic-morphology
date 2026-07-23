#!/usr/bin/env python3
"""Build a compact morphological-generation FST for Lingala verb inflection.

Input  format: features first as bracketed atomic symbols, then stem chars,
               each symbol quoted, e.g.  '[V]''[FUT]''y''o''k'
Output format: the same feature tags echoed, then the inflected wordform chars,
               e.g.  '[V]''[FUT]''k''o''y''o''k''a'

Morphology derived from data/lin.trn (all 159 rows covered exactly):
  FUT  : ko + stem + a
  NFIN : ko + stem + a      (lexical exceptions kang->kakanga, kabwan->kakabwana)
  PRS  : stem + i
  PST  : stem + aki
"""
from pyfoma import FST

# Stem alphabet (identical on lemma and surface sides in this data).
CHARS = list("abdefgiklmnopstuvwyz")


def q(c):
    """Quote a single symbol so it is atomic in a PyFoma regex."""
    return "'" + c + "'"


fsts = {}

# Identity over any single stem character.
fsts["S"] = FST.re("|".join(q(c) for c in CHARS))

# The two NFIN stems that irregularly take the `ka-` prefix.
NFIN_EXC = ["kang", "kabwan"]
exc_re = " | ".join("(" + " ".join(q(c) for c in w) + ")" for w in NFIN_EXC)
# Identity over every stem EXCEPT the NFIN exceptions (keeps paths disjoint).
fsts["StemGen"] = FST.re(f"$S+ - ({exc_re})", fsts)

# One transducer branch per tense/paradigm. Feature tags are echoed via
# identity; affix material is inserted with epsilon cross-products ('':'x').
FUT = r"""'[V]' '[FUT]' ('':'k')('':'o') $S+ ('':'a')"""
PRS = r"""'[V]' '[PRS]' $S+ ('':'i')"""
PST = r"""'[V]' '[PST]' $S+ ('':'a')('':'k')('':'i')"""
NFIN_reg = r"""'[V]' '[NFIN]' ('':'k')('':'o') $StemGen ('':'a')"""
# Explicit lexical exceptions: kang->kakanga, kabwan->kakabwana.
NFIN_exc = " | ".join(
    "'[V]' '[NFIN]' ('':'k')('':'a') (%s) ('':'a')"
    % " ".join(q(c) for c in w)
    for w in NFIN_EXC
)

grammar = FST.re(
    f"({FUT}) | ({PRS}) | ({PST}) | ({NFIN_reg}) | ({NFIN_exc})",
    fsts,
)

grammar = grammar.minimize()

with open("test.foma", "w") as fh:
    fh.write(grammar.to_fomastring())

print(f"FST built with {len(list(grammar.states))} states; saved to test.foma")
