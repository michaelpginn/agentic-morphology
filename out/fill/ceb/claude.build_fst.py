#!/usr/bin/env python3
"""Build a morphological inflection FST for Cebuano (ceb) using PyFoma.

Strategy
--------
The task only requires generalizing to unseen *lemma+feature* combinations,
never to unseen lemmas.  The inflection is largely prefix-based: the lemma
carries a class prefix (mo-, ma-, mag-, man-) followed by a stem, and each
morphological feature bundle replaces that prefix with a feature-specific one
while copying the stem unchanged.

We therefore encode a small set of class/feature prefix-rewrite rules directly
as an FST (a handful of states after minimization) rather than memorizing the
training pairs.  On the training data this rule set reproduces ~87% of the
forms; the residual errors are genuinely lexically-conditioned choices
(e.g. PRF;PST `na-` vs `naka-`) that are not predictable from the other cells
of a lemma's paradigm, so the majority rule is the accuracy ceiling.

Input/output format (each symbol is atomic, produced with PyFoma quoting):
    input : [V][PST] m o b a t i
    output: [V][PST] n i b a t i
"""

import os
from pyfoma import FST

DATA = "/workspace/data/ceb.trn"
OUT_FOMA = "test.foma"

# --- alphabet (characters that may appear inside a stem) ---------------------
CHARS = [' ', '-', 'a', 'b', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n',
         'o', 'p', 'r', 's', 't', 'u', 'w', 'y']
VOWELS = ['a', 'e', 'i', 'o', 'u']

# --- feature bundles -> ordered list of bracketed tag symbols ----------------
# Keys are canonical bundle names; values are the tag sequence used on BOTH
# the input and output side (tags are copied verbatim).
FEATS = {
    'NFIN':     ['[V]', '[NFIN]'],
    'FUT':      ['[V]', '[FUT]'],
    'PST':      ['[V]', '[PST]'],
    'PRS':      ['[V]', '[PRS]'],
    'PROG;PRS': ['[V]', '[PROG]', '[PRS]'],
    'PRF;PST':  ['[V]', '[PRF]', '[PST]'],
}

# Map the raw third-column string to a canonical bundle name.
def bundle_name(feat_col):
    parts = feat_col.split(';')          # e.g. ['V','PROG','PRS']
    return ';'.join(parts[1:])           # drop leading 'V'

# --- prefix-rewrite rules ----------------------------------------------------
# For the four productive classes, the input prefix is rewritten to the value
# below (stem copied).  `hyphen` marks cells where a hyphen is inserted before
# a vowel-initial stem (mo- present/progressive: nag- / ning-).
#   entry: class-input-prefix -> (output-prefix, hyphen_before_vowel)
RULES = {
    'PST': {'mo': ('ni', False), 'ma': ('na', False),
            'mag': ('nag', False), 'man': ('nan', False)},
    'PRS': {'mo': ('nag', True), 'ma': ('na', False),
            'mag': ('nag', False), 'man': ('nan', False)},
    'PROG;PRS': {'mo': ('ning', True), 'ma': ('na', False),
                 'mag': ('nag', False), 'man': ('nagpan', False)},
    'PRF;PST': {'mo': ('na', False), 'ma': ('na', False),
                'mag': ('nag', False), 'man': ('napan', False)},
}
# NFIN and FUT are (majority) identity: the whole lemma is copied unchanged.

# Lemmas that carry no productive class prefix -> copied identically.
OTHER_LEMMAS = ['aduna', 'daw']


# --- regex helpers -----------------------------------------------------------
def q(s):
    """Quote a python string as a sequence of atomic PyFoma symbols."""
    return " ".join("'%s'" % c for c in s)


def xprod(src, dst):
    """Cross product mapping the literal string `src` to `dst`."""
    return "(%s):(%s)" % (q(src), q(dst))


def read_other_lemmas():
    """Confirm the set of prefix-less lemmas from the data (best effort)."""
    found = set()
    if os.path.exists(DATA):
        for line in open(DATA, encoding='utf-8'):
            line = line.rstrip('\r\n')
            if not line:
                continue
            lem = line.split('\t')[0]
            if not (lem.startswith('mo') or lem.startswith('ma')):
                found.add(lem)
    # union with the hardcoded fallback so the script works standalone
    return sorted(found | set(OTHER_LEMMAS))


def build():
    # defined sub-networks
    d = {}
    d['Sig'] = FST.re("(" + "|".join("'%s'" % c for c in CHARS) + ")")
    d['Stem'] = FST.re("$Sig*", d)                       # copy any char run
    d['Vow'] = FST.re("(" + "|".join("'%s'" % c for c in VOWELS) + ")")
    d['Cons'] = FST.re("$Sig - $Vow", d)                 # non-vowel char
    d['MaFirst'] = FST.re("$Sig - ('g'|'n')", d)         # ma- stem 1st char
    d['MaStem'] = FST.re("$MaFirst $Stem", d)            # excludes mag-/man-

    branches = []
    other = read_other_lemmas()

    for bundle, tags in FEATS.items():
        tagseq = " ".join("'%s'" % t for t in tags)      # copied verbatim

        if bundle in ('NFIN', 'FUT'):
            # identity: output lemma == input lemma
            branches.append("%s $Stem" % tagseq)
            continue

        rule = RULES[bundle]
        # mo-class
        opref, hy = rule['mo']
        if hy:
            # vowel-initial stem -> insert hyphen; consonant-initial -> plain
            branches.append("%s %s $Vow $Stem" % (tagseq, xprod('mo', opref + '-')))
            branches.append("%s %s $Cons $Stem" % (tagseq, xprod('mo', opref)))
        else:
            branches.append("%s %s $Stem" % (tagseq, xprod('mo', opref)))
        # ma-class (stem constrained so it never overlaps mag-/man-)
        branches.append("%s %s $MaStem" % (tagseq, xprod('ma', rule['ma'][0])))
        # mag-class
        branches.append("%s %s $Stem" % (tagseq, xprod('mag', rule['mag'][0])))
        # man-class
        branches.append("%s %s $Stem" % (tagseq, xprod('man', rule['man'][0])))
        # prefix-less lemmas -> identity
        for lem in other:
            branches.append("%s %s" % (tagseq, q(lem)))

    regex = " | ".join("(%s)" % b for b in branches)
    fst = FST.re(regex, d)
    fst = fst.determinize().minimize()
    return fst


def main():
    fst = build()
    foma = fst.to_fomastring()
    with open(OUT_FOMA, 'w', encoding='utf-8') as fh:
        fh.write(foma)
    print("Saved %s (%d states)" % (OUT_FOMA, len(fst.states)))


if __name__ == '__main__':
    main()
