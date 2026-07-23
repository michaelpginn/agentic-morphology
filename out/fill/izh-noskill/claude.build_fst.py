#!/usr/bin/env python3
"""Build an FST for Izhorian nominal inflection from izh.trn.

Strategy: this is a paradigm-cell-filling problem (lemmas are seen in training,
but with different feature bundles).  We do the morphological generalisation in
Python by analogy, producing a complete lemma x bundle table (gold forms for
seen cells, predicted forms for unseen ones), then compile that table into a
single minimised FST.

Input side  : feature tags (as atomic bracketed symbols) followed by the lemma,
              one atomic symbol per character, e.g.  '[N]''[TRANS]''[PL]''k''u''k''k''a'
Output side : the same feature tags followed by the inflected wordform.
"""

from collections import defaultdict, Counter
from pyfoma import FST

TRAIN = '/workspace/data/izh.trn'
OUT = 'test.foma'

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load(path):
    table = defaultdict(dict)      # table[lemma][(case,num)] = form
    order = {}                     # (case,num) -> feature list order seen in data
    pos_of = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')
            if not line.strip():
                continue
            lemma, form, feats = line.split('\t')
            fl = feats.split(';')
            pos, cn = fl[0], tuple(fl[1:])
            table[lemma][cn] = form
            pos_of[lemma] = pos
            order[cn] = fl[1:]
    return table, order, pos_of


# ---------------------------------------------------------------------------
# Analogical prediction (paradigm cell filling)
# ---------------------------------------------------------------------------

def lcp(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n

def csuf(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n

SIM_POW = 2      # weight of donor-lemma similarity
REL_POW = 4      # weight of learned pivot->target reliability


def similarity(table, D, L, exclude):
    """How much does donor lemma D look like target lemma L (shared endings)?"""
    s = 0
    for C in table[L]:
        if C == exclude:
            continue
        if C in table[D]:
            s += csuf(table[D][C], table[L][C])
    return s


def apply_pair(table, P, B, L, skip):
    """Learn suffix-substitution rules for pivot cell P -> target cell B from all
    donor lemmas (except `skip`) and apply them to L's form in cell P.
    Returns a Counter of weighted candidate output strings."""
    Lform = table[L].get(P)
    out = Counter()
    if Lform is None:
        return out
    for D in table:
        if D == skip:
            continue
        DP = table[D].get(P)
        DB = table[D].get(B)
        if DP is None or DB is None:
            continue
        p = lcp(DP, DB)
        src, dst = DP[p:], DB[p:]
        if Lform.endswith(src):
            cand = Lform[:len(Lform) - len(src)] + dst
            w = (len(src) + 1) * ((1 + similarity(table, D, L, B)) ** SIM_POW)
            out[cand] += w
    return out


def build_reliability(table):
    """rel[(P,B)] = leave-one-out accuracy of predicting cell B from cell P."""
    cells = set()
    for L in table:
        cells |= set(table[L])
    rel = {}
    for P in cells:
        for B in cells:
            if P == B:
                continue
            corr = tot = 0
            for L in table:
                if P in table[L] and B in table[L]:
                    v = apply_pair(table, P, B, L, skip=L)
                    if v:
                        tot += 1
                        if v.most_common(1)[0][0] == table[L][B]:
                            corr += 1
            if tot >= 3:
                rel[(P, B)] = corr / tot
    return rel


def predict(table, rel, L, B):
    """Predict the form of lemma L in cell B by reliability-weighted analogy."""
    if B == ('NOM', 'SG'):
        return L                       # nominative singular is always the lemma
    votes = Counter()
    for P in table[L]:
        if P == B:
            continue
        r = rel.get((P, B))
        if not r or r <= 0:
            continue
        cand = apply_pair(table, P, B, L, skip=L)
        tot = sum(cand.values()) or 1
        for form, c in cand.items():
            votes[form] += (r ** REL_POW) * c / tot
    if not votes:
        return None
    return votes.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# FST compilation
# ---------------------------------------------------------------------------

def tok(ch):
    """Turn a single character into a quoted (atomic) PyFoma regex token."""
    return r"\'" if ch == "'" else "'" + ch + "'"

def feat_tok(feat):
    return "'[" + feat + "]'"


def build():
    table, order, pos_of = load(TRAIN)
    rel = build_reliability(table)

    all_cells = sorted({cn for L in table for cn in table[L]})

    pairs = []          # (input_tokens, output_tokens)
    for L in table:
        pos = pos_of[L]
        for cn in all_cells:
            feats = [pos] + list(order[cn])
            if cn in table[L]:
                form = table[L][cn]          # gold for seen cells
            else:
                form = predict(table, rel, L, cn)
                if form is None:
                    continue
            ftok = [feat_tok(f) for f in feats]
            inp = ftok + [tok(c) for c in L]
            outp = ftok + [tok(c) for c in form]
            pairs.append((inp, outp))

    # Compile into one FST: union of all input:output string pairs.
    print(f"compiling {len(pairs)} paths ...")
    regex = " | ".join(f"({' '.join(i)}):({' '.join(o)})" for i, o in pairs)
    fst = FST.re(regex)
    print("states before minimize:", len(fst.states))
    fst = fst.determinize_as_dfa().minimize()
    print("states after minimize :", len(fst.states))

    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(fst.to_fomastring())
    print(f"saved -> {OUT}")
    return fst, table, order, pos_of, rel


if __name__ == '__main__':
    build()
