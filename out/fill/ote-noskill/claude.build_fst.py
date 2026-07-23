#!/usr/bin/env python3
"""Build a morphological-inflection FST for the Otomi (ote) data using PyFoma.

Strategy
--------
Analysis of the training data shows the inflected wordform has the shape

    wordform = proclitic + " " + stem          (or just  stem  for the 3sg.prs.ipfv cell)

where:
  * the proclitic is a *deterministic* function of the morphological feature cell
    (18 cells, each with exactly one proclitic), and
  * the stem is a mutated form of the lemma.  The mutation is a lemma-specific
    (irregular) initial-consonant alternation, but it falls into just FOUR
    "grades" that partition the 18 feature cells:
        I  -> all IPFV cells
        1  -> 1st-person  non-IPFV cells
        2  -> 2nd-person  non-IPFV cells
        3  -> 3rd-person  non-IPFV cells
    Within a grade the stem is (virtually always) identical, so knowing one cell
    of a grade lets us predict every other cell of that grade for the same lemma.

Because the task only requires generalizing to unseen lemma+feature COMBINATIONS
(every test lemma is seen in training), we memorize, per lemma, the stem for each
grade (filling any grade a lemma never appears in with a fallback copied from the
nearest available grade).  We then emit, for every lemma and all 18 feature cells,
a transducer path.  Feature tags and each character are treated as atomic quoted
symbols.
"""

import collections
from pyfoma import FST, State

TRAIN = "/workspace/data/ote.trn"
OUT = "test.foma"

# The 18 feature cells, each mapped to its stem "grade".
GRADE = {
    'V;IPFV;SG;1;PRS': 'I', 'V;IPFV;SG;1;PST': 'I',
    'V;IPFV;SG;2;PRS': 'I', 'V;IPFV;SG;2;PST': 'I',
    'V;IPFV;SG;3;PRS': 'I', 'V;IPFV;SG;3;PST': 'I',
    'V;IRR;SG;1': '1', 'V;PFV;SG;1': '1', 'V;PRF;1;PST': '1', 'V;PRF;SG;1;PRS': '1',
    'V;IRR;SG;2': '2', 'V;PFV;SG;2': '2', 'V;PRF;2;PST': '2', 'V;PRF;SG;2;PRS': '2',
    'V;IRR;SG;3': '3', 'V;PFV;SG;3': '3', 'V;PRF;3;PST': '3', 'V;PRF;SG;3;PRS': '3',
}
CELLS = list(GRADE.keys())

# For a grade that a lemma never realizes in training, copy the stem from the
# nearest grade (empirically most-similar order, measured on the training data).
FALLBACK = {'I': ['1', '2', '3'], '1': ['2', 'I', '3'],
            '2': ['1', '3', 'I'], '3': ['2', '1', 'I']}

# The proclitics that may begin a wordform (used to split prefix from stem).
PREFIXES = {'dí', 'ndí', 'gí', 'ngí', 'mí', 'ga', 'gi', 'da', 'dá', 'gá',
            'bi', 'stí', 'xkí', 'xki', 'stá', 'xká', 'xa'}


def split_prefix(wf):
    """Return (prefix, stem) for a wordform."""
    parts = wf.split(' ')
    if len(parts) > 1 and parts[0] in PREFIXES:
        return parts[0], ' '.join(parts[1:])
    return '', wf


def q(s):
    """Quote each character of s as an atomic PyFoma symbol."""
    return ''.join("'" + ch + "'" for ch in s)


def qtags(cell):
    """Quote the feature tags of a cell as atomic [X] symbols, in data order."""
    return ''.join("'[" + f + "]'" for f in cell.split(';'))


def build_trie(pairs):
    """Build an (epsilon-heavy) transducer trie from (input_tokens, output_tokens)
    pairs, sharing common input prefixes.  Each input symbol is consumed with an
    epsilon output; the full output string is then emitted from the leaf.  A later
    epsilon_remove().determinize().minimize() collapses this into a small machine."""
    f = FST()
    root = f.initialstate
    f.states = {root}
    f.finalstates = set()
    nodes = {(): root}
    alpha = set()
    for intoks, outtoks in pairs:
        cur = root
        pre = ()
        for sym in intoks:
            pre = pre + (sym,)
            alpha.add(sym)
            nxt = nodes.get(pre)
            if nxt is None:
                nxt = State()
                f.states.add(nxt)
                nodes[pre] = nxt
                cur.add_transition(nxt, (sym, ''))
            cur = nxt
        s = cur
        for osym in outtoks:
            alpha.add(osym)
            ns = State()
            f.states.add(ns)
            s.add_transition(ns, ('', osym))
            s = ns
        s.finalweight = 0.0
        f.finalstates.add(s)
    f.alphabet = alpha
    return f


def main():
    # --- collect statistics -------------------------------------------------
    # prefix per cell (majority), and stem counter per (lemma, grade)
    prefix_ctr = collections.defaultdict(collections.Counter)
    stem_ctr = collections.defaultdict(collections.Counter)
    lemmas = []
    seen_lemma = set()

    with open(TRAIN, encoding='utf-8') as fh:
        for line in fh:
            line = line.rstrip('\n')
            if not line:
                continue
            lemma, wf, feats = line.split('\t')
            pre, stem = split_prefix(wf)
            prefix_ctr[feats][pre] += 1
            stem_ctr[(lemma, GRADE[feats])][stem] += 1
            if lemma not in seen_lemma:
                seen_lemma.add(lemma)
                lemmas.append(lemma)

    prefix_of = {c: prefix_ctr[c].most_common(1)[0][0] for c in CELLS}

    # best stem per (lemma, grade)
    stem_of = {}
    for (lemma, g), ctr in stem_ctr.items():
        stem_of[(lemma, g)] = ctr.most_common(1)[0][0]

    # fill every lemma x grade using fallback where a grade is missing
    for lemma in lemmas:
        for g in 'I123':
            if (lemma, g) in stem_of:
                continue
            chosen = None
            for src in FALLBACK[g]:
                if (lemma, src) in stem_of:
                    chosen = stem_of[(lemma, src)]
                    break
            if chosen is None:      # should not happen (lemma has >=1 grade)
                chosen = lemma
            stem_of[(lemma, g)] = chosen

    # --- build the FST ------------------------------------------------------
    # Factored construction (keeps the machine compact):
    #   * one minimized stem-map transducer per grade  (lemma-chars -> stem-chars)
    #   * one tiny header per cell  (feature-tags -> feature-tags + proclitic)
    #   * final machine = union over the 18 cells of  header . stemmap[grade]
    # A final minimize() merges the shared stem structure across cells/grades.
    print("Building stem maps ...")
    stemmap = {}
    for g in 'I123':
        pairs = [(list(lemma), list(stem_of[(lemma, g)])) for lemma in lemmas]
        stemmap[g] = build_trie(pairs).epsilon_remove().determinize().minimize()
        print(f"  grade {g}: {len(stemmap[g].states)} states")

    print("Assembling cells ...")
    full = None
    for cell in CELLS:
        tags = ['[' + f + ']' for f in cell.split(';')]
        pre = prefix_of[cell]
        pre_out = (pre + ' ') if pre else ''      # proclitic + separating space
        header = build_trie([(list(tags), list(tags) + list(pre_out))])
        header = header.epsilon_remove().determinize().minimize()
        cell_fst = header.concatenate(stemmap[GRADE[cell]])
        full = cell_fst if full is None else FST.union(full, cell_fst)

    print("Determinizing / minimizing ...")
    full = full.epsilon_remove().determinize().minimize()

    print(f"Final FST: {len(full.states)} states")

    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(full.to_fomastring())
    print(f"Wrote {OUT}")

    return full, prefix_of, stem_of


if __name__ == "__main__":
    main()
