#!/usr/bin/env python3
"""Build a morphological-inflection FST for Lingala verbs with PyFoma.

Input format  : feature tags first, then the lemma, every symbol quoted so it is
                atomic, e.g.  '[V]''[PRS]''r''u''n'
Output format : the same feature tags, followed by the inflected wordform, again
                one atomic symbol per character, e.g. '[V]''[PRS]''r''u''n''i'

The Lingala verb paradigm in the training data is almost perfectly regular:

    PRS  : root + i          (yamb -> yambi)
    PST  : root + aki        (yamb -> yambaki)
    FUT  : ko + root + a     (yamb -> koyamba)
    NFIN : ko + root + a     (yamb -> koyamba)

The only irregularities are two lexical NFIN forms whose prefix is "ka" instead
of "ko":  kang -> kakanga  and  kabwan -> kakabwana.  These two roots are
excluded from the general NFIN rule and given explicit paths so the transducer
stays unambiguous.
"""

from pyfoma import FST

# Every character that occurs in the lemmas / wordforms.
CHARS = list("abdefgiklmnopstuvwyz")

# C : identity over any single character (used to copy the root).
C = "(" + "|".join("'%s'" % c for c in CHARS) + ")"

# Feature tags are copied verbatim.
V = "'[V]':'[V]'"


def tag(t):
    return "'[%s]':'[%s]'" % (t, t)


# NFIN roots that take the irregular "ka" prefix.
IRREG_NFIN = {"kang": "kakanga", "kabwan": "kakabwana"}


def seq(s):
    """A concatenation of quoted atomic symbols for the string s."""
    return " ".join("'%s'" % ch for ch in s)


# Regular rules -------------------------------------------------------------
PRS = "%s %s %s* '':'i'" % (V, tag("PRS"), C)
PST = "%s %s %s* '':'a' '':'k' '':'i'" % (V, tag("PST"), C)
FUT = "%s %s '':'k' '':'o' %s* '':'a'" % (V, tag("FUT"), C)

# General NFIN, with the two irregular roots removed from its domain.
excl = "|".join("(%s)" % seq(r) for r in IRREG_NFIN)
NFIN_gen = "%s %s '':'k' '':'o' (%s* - (%s)) '':'a'" % (V, tag("NFIN"), C, excl)

# Explicit paths for the irregular NFIN forms.
nfin_exc = [
    "(%s %s %s) : (%s %s %s)" % (V, tag("NFIN"), seq(root), V, tag("NFIN"), seq(out))
    for root, out in IRREG_NFIN.items()
]

# Full transducer = union of all rules, then minimized for compactness.
regex = " | ".join([PRS, PST, FUT, NFIN_gen] + nfin_exc)

fst = FST.re(regex)
fst = fst.determinize().minimize()

print("states:", len(fst.states))

with open("test.foma", "w") as fh:
    fh.write(fst.to_fomastring())

print("wrote test.foma")
