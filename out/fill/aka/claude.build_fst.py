#!/usr/bin/env python3
"""Build a morphological-inflection FST for Akan (aka) with PyFoma.

Input format (upper side): the morphological features first, each feature a
single quoted bracket-tag atomic symbol (e.g. '[V]' '[PST+IMMED]'), followed by
each character of the lemma as a quoted atomic symbol, e.g.

    '[V]''[PST+IMMED]''b''o''r''o'

Output (lower side): the same feature tags echoed, followed by each character of
the inflected wordform.

The grammar is a single productive system: every feature combination selects a
prefix (with regular nasal assimilation) and an optional -ee/-e suffix which is
appended to the (unchanged) lemma.  This reproduces every training pair exactly
and generalizes to unseen lemma+feature combinations.
"""

from pyfoma import FST

# ---------------------------------------------------------------------------
# Alphabet
# ---------------------------------------------------------------------------
ALL = [' ', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'm', 'n', 'o', 'p',
       'r', 's', 't', 'u', 'w', 'y', 'ɔ', 'ɛ']
LAB = set('bpfm')          # labials trigger the m-nasal
NONLAB = [c for c in ALL if c not in LAB]

# ---------------------------------------------------------------------------
# Per-feature-combination grammar.
#   template: literal prefix string; a trailing 'N' marks a geminate assimilating
#             nasal (nn/mm) adjacent to the stem, '1' marks a single nasal (n/m).
#   suffix:   1 if the -ee (~ -e after final e) suffix is appended, else 0.
# ---------------------------------------------------------------------------
FEAT = {
    'V;HAB;PRS': ('', 0), 'V;IMP;PRS': ('', 0), 'V;NFIN': ('', 0),
    'V;HAB;PST': ('', 1),
    'V;HAB;FUT': ('bɛ', 0), 'V;PST+IMMED': ('bɛ', 1),
    'V;PRS;LGSPEC1': ('kɔ', 0), 'V;PST;LGSPEC1': ('kɔ', 1),
    'V;HAB+PRF;PRS': ('a', 0), 'V;HAB+PRF;PST': ('nna a', 0),
    'V;HAB+PROG;PRS': ('re', 0), 'V;HAB+PROG;PST': ('nna re', 0),
    'V;PROG;PRS+IMMED': ('rebɛ', 0), 'V;PROG;PST+IMMED': ('nna rebɛ', 0),
    'V;PROG;PRS;LGSPEC1': ('rekɔ', 0), 'V;PROG;PST;LGSPEC1': ('nna rekɔ', 0),
    'V;PRF;PRS+IMMED': ('abɛ', 0), 'V;PRF;PST+IMMED': ('nna abɛ', 0),
    'V;PRF;PRS;LGSPEC1': ('akɔ', 0), 'V;PRF;PST;LGSPEC1': ('nna akɔ', 0),
    'V;NEG;PRS;LGSPEC1': ('nnkɔ', 0), 'V;NEG;PST+IMMED': ('ammbɛ', 0),
    'V;NEG;PST;LGSPEC1': ('ammkɔ', 0),
    'V;PROG;NEG;PRS+IMMED': ('remmbɛ', 0), 'V;PROG;NEG;PST+IMMED': ('nna remmbɛ', 0),
    'V;PROG;NEG;PRS;LGSPEC1': ('rennkɔ', 0), 'V;PROG;NEG;PST;LGSPEC1': ('nna rennkɔ', 0),
    'V;PRF;NEG;PRS+IMMED': ('mmbɛ', 1), 'V;PRF;NEG;PST+IMMED': ('nna mmbɛ', 1),
    'V;PRF;NEG;PRS;LGSPEC1': ('nnkɔ', 1), 'V;PRF;NEG;PST;LGSPEC1': ('nna nnkɔ', 1),
    'V;HAB+PRF;NEG;PRS': ('N', 1), 'V;HAB+PRF;NEG;PST': ('nna N', 1),
    'V;HAB+PROG;NEG;PRS': ('reN', 0), 'V;HAB+PROG;NEG;PST': ('nna reN', 0),
    'V;HAB;NEG;FUT': ('reN', 0), 'V;HAB;NEG;PRS': ('N', 0), 'V;HAB;NEG;PST': ('aN', 0),
    'V;IMP;NEG;PRS': ('mma N', 0), 'V;SBJV;NEG;PRS': ('mma N', 0),
    'V;SBJV;PRS': ('1', 0),
}

# ---------------------------------------------------------------------------
# Regex-fragment builders
# ---------------------------------------------------------------------------
def q(c):
    """Quote a single character as an atomic FST.re symbol."""
    return "'%s'" % c

def union(chars):
    """Identity-copy union over a set/list of characters."""
    return "(" + "|".join(q(c) for c in chars) + ")"

def ins(s):
    """Insert a literal string on the output (epsilon -> chars)."""
    return " ".join("'':%s" % q(c) for c in s)

ANY = union(ALL)

def stembody(Fset, Lset):
    """Copy a non-empty stem whose first char is in Fset and last char in Lset."""
    Fset = list(Fset)
    Lset = list(Lset)
    both = [c for c in Fset if c in set(Lset)]
    alts = []
    if both:                                   # length-1 stem
        alts.append(union(both))
    alts.append(union(Fset) + " " + ANY + "* " + union(Lset))   # length >= 2
    return "(" + " | ".join(alts) + ")"

def build_stem(nasal, suf):
    """Transducer fragment mapping the lemma to prefix-nasal+lemma+suffix."""
    if nasal:                                  # split on labiality of first char
        gem = 2 if nasal == 'N' else 1
        firsts = [(LAB, 'm' * gem), (NONLAB, 'n' * gem)]
    else:
        firsts = [(ALL, '')]
    branches = []
    for Fset, nas in firsts:
        pre = (ins(nas) + " ") if nas else ""
        if suf:                                # -e after final 'e', else -ee
            branches.append(pre + stembody(Fset, ['e']) + " " + ins('e'))
            branches.append(pre + stembody(Fset, [c for c in ALL if c != 'e'])
                            + " " + ins('ee'))
        else:
            branches.append(pre + stembody(Fset, ALL))
    return "(" + " | ".join(branches) + ")"

def feat_fst(ft):
    tpl, suf = FEAT[ft]
    tag = " ".join("'[%s]'" % t for t in ft.split(';'))    # echo feature tags
    nasal = None
    lit = tpl
    if tpl.endswith('N'):
        nasal, lit = 'N', tpl[:-1]
    elif tpl.endswith('1'):
        nasal, lit = '1', tpl[:-1]
    parts = [tag]
    if lit:
        parts.append(ins(lit))
    parts.append(build_stem(nasal, suf))
    return "(" + " ".join(parts) + ")"

# ---------------------------------------------------------------------------
# Assemble, minimize, save
# ---------------------------------------------------------------------------
regex = " | ".join(feat_fst(ft) for ft in FEAT)
grammar = FST.re(regex)
grammar = grammar.determinize().minimize()

with open("test.foma", "w") as fh:
    fh.write(grammar.to_fomastring())

print("states:", len(grammar.states))
