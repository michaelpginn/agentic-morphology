#!/usr/bin/env python3
"""Build a morphological-inflection FST for Crimean Tatar (crh) with PyFoma.

Input  : feature tags (each ';'-separated feature wrapped in [ ]) followed by the
         lemma, every symbol atomic/quoted, e.g.  [N][DAT]iklim
Output : the same feature tags followed by the inflected wordform, e.g. [N][DAT]iklimge

Strategy
--------
Crimean Tatar is an agglutinative Turkic language: a wordform is the stem plus a
case/tense suffix chosen by (a) two-way front/back vowel harmony and (b) consonant
voicing assimilation.

The FST is a per-lemma paradigm LOOKUP.  Because the task guarantees test lemmas
are seen in training (only the lemma+feature *combination* is new), we pin down
each lemma's harmony/voicing class - and a couple of minor sub-classes - from the
other forms it was observed in, then emit its whole paradigm (the observed form
where known, otherwise the class-driven prediction).  Nouns and adjectives
inflect identically, so a nominal lemma gets both paradigms (a lemma seen only as
one may be queried as the other).

A compact phonological RULE transducer (`build_rules`) implementing the same
alternations for *arbitrary* stems is kept for reference / analysis, but is not
part of the saved FST: combining it with the lookup as a priority union relies on
`difference`/`compose`, whose result does not survive PyFoma's foma
serialisation cleanly.  The lookup alone already covers every seen lemma.
"""
from collections import defaultdict
from pyfoma import FST

TRAIN = 'data/crh.trn'

# ---------------------------------------------------------------------------
# Phonology
# ---------------------------------------------------------------------------
CHARS     = " -MRUZabcdefghijklmnopqrstuvyzÇâçñöüğİış"
BACK_V    = set('aıouâU')          # back vowels (incl. uppercase U)
FRONT_V   = set('eiöüİ')           # front vowels (incl. uppercase İ)
VOWELS    = BACK_V | FRONT_V
CONS      = [c for c in CHARS if c not in VOWELS]
VOICELESS = set('ptkqçşsfhÇ')      # consonants that trigger t/q/k suffixes

# case feature -> (back-voiced, back-voiceless, front-voiced, front-voiceless)
CASE_SUFFIX = {
    'ACC': ('nı',  'nı',  'ni',  'ni'),    # no voicing distinction
    'GEN': ('nıñ', 'nıñ', 'niñ', 'niñ'),
    'DAT': ('ğa',  'qa',  'ge',  'ke'),
    'LOC': ('da',  'ta',  'de',  'te'),
    'ABL': ('dan', 'tan', 'den', 'ten'),
}
# 3sg-possessive compounds ("el işi") take an -n- linker in DAT/LOC/ABL.
POSS_SUFFIX = {'DAT': ('na', 'ne'), 'LOC': ('nda', 'nde'), 'ABL': ('ndan', 'nden')}

NOUN_FEATS = ['NOM', 'ACC', 'GEN', 'DAT', 'LOC', 'ABL']
FEATS = {
    'N':   ['N;' + c for c in NOUN_FEATS],
    'ADJ': ['ADJ;' + c for c in NOUN_FEATS],
    'V':   ['V;NFIN', 'V;IMP;SG;2', 'V;SG;3;PRS', 'V;SG;3;PST'],
}
ALL_FEATS = FEATS['N'] + FEATS['ADJ'] + FEATS['V']
ALL_TAGS = sorted({'[%s]' % t for f in ALL_FEATS for t in f.split(';')})


def stem(lemma, feat):
    """Verb stem = lemma minus the -maq/-mek infinitive; else the lemma itself."""
    return lemma[:-3] if feat.startswith('V') else lemma


def rule_harmony(s):
    for c in reversed(s):
        if c in BACK_V:
            return 'B'
        if c in FRONT_V:
            return 'F'
    return 'B'


def rule_voicing(s):
    return 'voiceless' if s[-1:] in VOICELESS else 'voiced'


def predict(lemma, feat, h, v, poss=False):
    """Regular inflection given harmony class h, voicing class v, possessive flag."""
    st = stem(lemma, feat)
    case = feat.split(';')[-1]
    if feat in ('N;NOM', 'ADJ;NOM', 'V;NFIN'):
        return lemma
    if poss and case in POSS_SUFFIX and st[-1:] in VOWELS:
        bk, fr = POSS_SUFFIX[case]
        return st + (bk if h == 'B' else fr)
    if case in CASE_SUFFIX:
        bv, bl, fv, fl = CASE_SUFFIX[case]
        if h == 'B':
            return st + (bv if v == 'voiced' else bl)
        return st + (fv if v == 'voiced' else fl)
    if feat == 'V;IMP;SG;2':
        return st
    if feat == 'V;SG;3;PRS':
        if st[-1:] in VOWELS:
            return st + 'y'
        return st + ('a' if h == 'B' else 'e')
    if feat == 'V;SG;3;PST':
        if h == 'B':
            return st + 'dı'
        return st + ('ti' if v == 'voiceless' else 'di')
    return lemma


# ---------------------------------------------------------------------------
# Per-lemma paradigm lookup
# ---------------------------------------------------------------------------
def load():
    by_lemma = defaultdict(list)      # lemma -> [(feat, form)]
    pos = defaultdict(set)            # lemma -> {N, ADJ, V}
    seen = {}                         # (lemma, feat) -> form
    for line in open(TRAIN, encoding='utf-8'):
        lemma, form, feat = line.rstrip('\n').split('\t')
        by_lemma[lemma].append((feat, form))
        pos[lemma].add(feat.split(';')[0])
        seen[(lemma, feat)] = form
    return by_lemma, pos, seen


def suffix_of(lemma, feat, form):
    st = stem(lemma, feat)
    return form[len(st):] if form.startswith(st) else None


def infer_classes(lemma, forms):
    """Unanimous-vote override of harmony (any suffix) and case voicing, plus
    detection of the 3sg-possessive -n- linker class."""
    h_votes, v_votes = set(), set()
    poss = False
    for feat, form in forms:
        suf = suffix_of(lemma, feat, form)
        if not suf:
            continue
        for c in suf:                      # first vowel of the suffix fixes harmony
            if c in BACK_V:
                h_votes.add('B')
                break
            if c in FRONT_V:
                h_votes.add('F')
                break
        if feat.split(';')[-1] in ('DAT', 'LOC', 'ABL'):
            st = stem(lemma, feat)
            if suf[:1] == 'n' and st[-1:] in VOWELS:
                poss = True
            elif suf[:1] in 'dğg':
                v_votes.add('voiced')
            elif suf[:1] in 'tqk':
                v_votes.add('voiceless')
    h = next(iter(h_votes)) if len(h_votes) == 1 else None
    v = next(iter(v_votes)) if len(v_votes) == 1 else None
    return h, v, poss


def target_feats(observed_pos):
    """Feature sets to emit.  Nouns and adjectives share morphology, so a nominal
    lemma gets both paradigms (test may query either)."""
    feats = []
    if observed_pos & {'N', 'ADJ'}:
        feats += FEATS['N'] + FEATS['ADJ']
    if 'V' in observed_pos:
        feats += FEATS['V']
    return feats


def paradigm():
    """Yield (input_symbols, output_symbols) for every seen lemma x feature."""
    by_lemma, pos, seen = load()
    for lemma, forms in by_lemma.items():
        oh, ov, poss = infer_classes(lemma, forms)
        for feat in target_feats(pos[lemma]):
            st = stem(lemma, feat)
            h = oh if oh is not None else rule_harmony(st)
            v = ov if ov is not None else rule_voicing(st)
            form = seen.get((lemma, feat)) or predict(lemma, feat, h, v, poss)
            tags = ['[%s]' % t for t in feat.split(';')]
            yield tags + list(lemma), tags + list(form)


# ---------------------------------------------------------------------------
# Compact phonological rule transducer (fallback for unseen lemmas)
# ---------------------------------------------------------------------------
def _cls(symbols):
    return '(' + '|'.join("'" + c + "'" for c in symbols) + ')'


def ins(s):
    return ''.join("('':'%s')" % c for c in s)


def dele(s):
    return ''.join("('%s':'')" % c for c in s)


def build_rules():
    defined = {
        'Sig': FST.re(_cls(CHARS)),
        'BV':  FST.re(_cls(BACK_V)),
        'FV':  FST.re(_cls(FRONT_V)),
        'C':   FST.re(_cls(CONS)),
        'VL':  FST.re(_cls(VOICELESS)),
        'VD':  FST.re(_cls([c for c in CHARS if c not in VOICELESS])),
    }
    LB, LF = '($Sig* $BV $C*)', '($Sig* $FV $C*)'
    VDE, VLE = '($Sig* $VD)', '($Sig* $VL)'

    def h2(back, front):                       # harmony only (ACC/GEN)
        return '%s%s | %s%s' % (LB, ins(back), LF, ins(front))

    def hv(bv, bl, fv, fl):                    # harmony + voicing (DAT/LOC/ABL)
        return ' | '.join([
            '(%s & %s)%s' % (LB, VDE, ins(bv)), '(%s & %s)%s' % (LB, VLE, ins(bl)),
            '(%s & %s)%s' % (LF, VDE, ins(fv)), '(%s & %s)%s' % (LF, VLE, ins(fl)),
        ])

    stems = {
        'N;NOM': '$Sig*',
        'N;ACC': h2('nı', 'ni'),
        'N;GEN': h2('nıñ', 'niñ'),
        'N;DAT': hv('ğa', 'qa', 'ge', 'ke'),
        'N;LOC': hv('da', 'ta', 'de', 'te'),
        'N;ABL': hv('dan', 'tan', 'den', 'ten'),
        'V;NFIN': '$Sig*',
        'V;IMP;SG;2': '%s%s | %s%s' % (LB, dele('maq'), LF, dele('mek')),
        'V;SG;3;PRS': ' | '.join([
            '($Sig* $BV)%s%s' % (dele('maq'), ins('y')),
            '($Sig* $BV $C $C*)%s%s' % (dele('maq'), ins('a')),
            '($Sig* $FV)%s%s' % (dele('mek'), ins('y')),
            '($Sig* $FV $C $C*)%s%s' % (dele('mek'), ins('e')),
        ]),
        'V;SG;3;PST': ' | '.join([
            '%s%s%s' % (LB, dele('maq'), ins('dı')),
            '(%s & %s)%s%s' % (LF, VDE, dele('mek'), ins('di')),
            '(%s & %s)%s%s' % (LF, VLE, dele('mek'), ins('ti')),
        ]),
    }
    for c in NOUN_FEATS:
        stems['ADJ;' + c] = stems['N;' + c]

    branches = []
    for feat, st in stems.items():
        tags = ' '.join("'[%s]'" % t for t in feat.split(';'))
        branches.append('(%s (%s))' % (tags, st))
    return FST.re(' | '.join(branches), defined=defined)


# ---------------------------------------------------------------------------
# Assemble: priority-union of lookup over rules
# ---------------------------------------------------------------------------
def quote(sym):
    return "'" + sym.replace('\\', '\\\\').replace("'", "\\'") + "'"


def build_lookup(batch=200):
    branches = []
    for insyms, outsyms in paradigm():
        i = ' '.join(quote(s) for s in insyms)
        o = ' '.join(quote(s) for s in outsyms)
        branches.append('(%s):(%s)' % (i, o))
    # Compiling one giant union is O(n^2); compile in batches then fold.
    fst = None
    for k in range(0, len(branches), batch):
        part = FST.re(' | '.join(branches[k:k + batch]))
        fst = part if fst is None else fst.union(part)
    return fst.epsilon_remove().determinize().minimize()


def build():
    return build_lookup()


if __name__ == '__main__':
    fst = build()
    print('states:', len(list(fst.states)))
    with open('test.foma', 'w', encoding='utf-8') as f:
        f.write(fst.to_fomastring())
    print('wrote test.foma')
