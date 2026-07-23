#!/usr/bin/env python3
"""Build a morphological-inflection FST for Crimean Tatar (crh) with PyFoma.

Input  (lexical) : feature tags (each bracketed + quoted atomic symbol) then lemma chars,
                   e.g.  '[N]''[DAT]''i''k''l''i''m'
Output (surface) : the same feature tags then the inflected wordform chars,
                   e.g.  '[N]''[DAT]''i''k''l''i''m''g''e'

Approach
--------
Crimean Tatar inflection is agglutinative with front/back vowel harmony and
voicing assimilation, so most words are covered by a compact set of phonological
rules (`rules_fst.build_rules`, 66 states).  Every test lemma is guaranteed to be
seen in training (only lemma+feature *combinations* are unseen), so we learn each
lemma's harmony/voicing/linking profile (`model.Model`) and detect the lemmas
whose behaviour deviates from the plain rules (loanwords, compound possessives,
irregular voicing, ...).  Those exceptions are stored in a small lexicon, and the
rule FST is restricted to the remaining (regular) lemmas so the two never overlap.

The final transducer is the union of the restricted rule FST and the exception
lexicon, determinized and minimized -- accurate (secondary: as compact as the
exceptions allow, ~880 states).
"""

from pyfoma import FST
from model import Model, NOUN_CASES, VERB_FEATS
import rules_fst as RF

TRAIN = '/workspace/data/crh.trn'


def quote(symbols):
    """Quote each symbol as an atomic PyFoma symbol: ['[N]','i'] -> \"'[N]''i'\"."""
    return "".join("'" + s + "'" for s in symbols)


def feats_for(poses):
    feats = []
    for pos in poses:
        if pos == 'V':
            feats += ['V;' + f for f in VERB_FEATS]
        else:
            feats += [pos + ';' + c for c in NOUN_CASES]
    return feats


def build():
    rows = [l.rstrip('\n').split('\t') for l in open(TRAIN)]
    model = Model(rows)

    # Every POS each lemma is attested with (a word may be both N and ADJ).
    lemma_pos = {}
    for lem, wf, feat in rows:
        lemma_pos.setdefault(lem, set()).add(feat.split(';')[0])

    # Compact phonological rule transducer for regular lemmas.
    rules = RF.build_rules()

    def to_input(lem, feat):
        return ['[' + p + ']' for p in feat.split(';')] + list(lem)

    def to_output(wf, feat):
        return ''.join('[' + p + ']' for p in feat.split(';')) + wf

    # A lemma is an "exception" if the rule FST cannot produce the desired form
    # (from observed gold or the learned per-lemma model) for any of its combos.
    exceptions = set()
    for lem, poses in lemma_pos.items():
        for feat in feats_for(poses):
            want = to_output(model.predict(lem, feat, use_gold=True), feat)
            if want not in rules.generate(to_input(lem, feat)):
                exceptions.add(lem)
                break

    # Exception lexicon: all feature combos for exceptional lemmas.
    lex_entries = []
    exc_inputs = []
    for lem in exceptions:
        for feat in feats_for(lemma_pos[lem]):
            wf = model.predict(lem, feat, use_gold=True)
            tagsyms = ['[' + p + ']' for p in feat.split(';')]
            lex_entries.append((quote(tagsyms + list(lem)), quote(tagsyms + list(wf))))
            exc_inputs.append(quote(tagsyms + list(lem)))
    lexicon = FST.rlg({'Start': [((i, o), '#') for i, o in lex_entries]}, 'Start')
    lexicon = lexicon.determinize_as_dfa().minimize()

    # Restrict the rule FST to inputs that are NOT exceptional lemmas, so the
    # union with the lexicon stays unambiguous (single output per input).
    allsyms = set(RF.ALL)
    for feat, _ in RF.COMBOS:
        for p in feat.split(';'):
            allsyms.add('[' + p + ']')
    universe = FST.re(RF.U(sorted(allsyms)) + '*')
    exc_acc = None
    for i in exc_inputs:
        a = FST.re(i)
        exc_acc = a if exc_acc is None else exc_acc.union(a)
    exc_acc = exc_acc.determinize_as_dfa().minimize()
    not_exc = FST.re('$U - $E', {'U': universe, 'E': exc_acc}).determinize_as_dfa().minimize()
    rules_reg = FST.re('$N @ $R', {'N': not_exc, 'R': rules}).determinize_as_dfa().minimize()

    final = rules_reg.union(lexicon).determinize_as_dfa().minimize()
    return final


if __name__ == '__main__':
    fst = build()
    print("states:", len(fst.states))
    with open('test.foma', 'w') as f:
        f.write(fst.to_fomastring())
    print("wrote test.foma")
