#!/usr/bin/env python3
"""Build a morphological-inflection FST for Sanskrit (san) with PyFoma.

Strategy
--------
The FST maps an input of the form  <feature-tags> <lemma-chars>  to
<feature-tags> <inflected-form-chars>, where every feature and every
character is an *atomic, quoted* symbol (e.g. '[PL]', '[PRS]', 'r', ...).

Because the held-out data only contains *unseen lemma+feature combinations*
of lemmas that were seen during training, the model exploits the fact that a
lemma's declension behaviour (its stem class, lexical gender, feminine-stem
type, i/u-stem variant, and n->N retroflexion trigger) can be recovered from
the lemma string plus the training paradigm.  For every lemma we therefore
predict the inflected form of *every* feature set of its part of speech, using
a backoff model over (feature-set, stem-class, gender, retroflexion, ...).

The resulting (input -> output) pairs are compiled into a single transducer
(a prefix-merged trie of composite in|out transitions, then minimized), which
keeps the machine reasonably compact while reproducing the predictions
exactly.
"""

import collections
from pyfoma import FST

TRN = "data/san.trn"
OUT = "test.foma"

# --------------------------------------------------------------------------
# Load training data
# --------------------------------------------------------------------------
rows = [l.rstrip("\n").split("\t") for l in open(TRN, encoding="utf-8") if l.strip()]

by_lem = collections.defaultdict(list)          # lemma -> [(form, feat), ...]
for lem, form, feat in rows:
    by_lem[lem].append((form, feat))

# --------------------------------------------------------------------------
# Linguistic helpers
# --------------------------------------------------------------------------
VS = set("ािीुूृॄेैोौंःँॢ")   # Devanagari dependent vowel signs / marks
# consonants that BLOCK n->N retroflexion (nati) when intervening
BLOCK = set("चछजझञटठडढणतथदधनलशसळ")


def lcp(a, b):
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


def trans(lem, form):
    """Transformation as (#chars removed from lemma end, suffix added)."""
    p = lcp(lem, form)
    return (len(lem) - p, form[p:])


def apply_trans(lem, t):
    return lem[: len(lem) - t[0]] + t[1]


def vclass(l):
    """Stem-final phonological class."""
    c = l[-1]
    if c == "्":
        return "C"           # consonant (halanta) stem
    if c in VS:
        return c             # explicit vowel-sign stem
    return "a"               # inherent-a stem


def _nati(stem):
    """Does a following suffix -n- retroflex to -N-?  (Panini nati, simplified)"""
    last = -1
    for i, ch in enumerate(stem):
        if ch in "रषृ":
            last = i
    if last < 0:
        return False
    return not any(ch in BLOCK for ch in stem[last + 1:])


def natikey(l):
    s = l[:-1] if (l and l[-1] in VS) else l
    return _nati(s)


def lexgender(lem):
    """Infer lexical gender of a NOUN from its training paradigm."""
    vc = vclass(lem)
    votes = collections.Counter()
    for form, feat in by_lem[lem]:
        F = set(feat.split(";"))
        if "N" not in F:
            continue
        if "SG" in F and "NOM" in F:           # nom sg: -m (neut) vs -H (masc)
            if form.endswith("म्"):
                votes["n"] += 2
            elif form.endswith("ः"):
                votes["m"] += 2
        if "PL" in F and "NOM" in F:
            if form.endswith(("ानि", "ाणि")):
                votes["n"] += 1
            elif form.endswith("ाः"):
                votes["m"] += 1
        if "DU" in F and "NOM" in F:
            if form.endswith("े"):
                votes["n"] += 1
            elif form.endswith("ौ"):
                votes["m"] += 1
        if ("DAT" in F and "SG" in F and form.endswith("्यै")) or \
           ("GEN" in F and "SG" in F and form.endswith("्याः")):
            votes["f"] += 2
    if votes:
        return votes.most_common(1)[0][0]
    return "f" if vc == "ा" else ("m" if vc == "a" else "?")


def femtype(lem):
    """For a-stem adjectives: feminine formed in -I (bharatI) or -A."""
    v = collections.Counter()
    for form, feat in by_lem[lem]:
        F = set(feat.split(";"))
        if "FEM" in F and "NOM" in F and "SG" in F:
            if form.endswith("ी"):
                v["i"] += 1
            elif form.endswith("ा"):
                v["a"] += 1
    return v.most_common(1)[0][0] if v else "a"


def retro(lem):
    """Per-lemma n->N retroflexion: does any inflected suffix show N (ण)?

    Retroflexion is consistent within a lemma but only partly predictable from
    the stem shape, so we read it directly off the training paradigm (every
    test lemma is attested in training)."""
    for form, feat in by_lem[lem]:
        _, add = trans(lem, form)
        if "ण" in add:
            return True
    return False


def variant(lem):
    """i/u-stem sub-paradigm: -ya- pattern vs guna vs -n- insertion."""
    v = collections.Counter()
    for form, feat in by_lem[lem]:
        F = set(feat.split(";"))
        if "SG" in F and any(x in F for x in ("GEN", "ABL", "DAT", "LOC")):
            if "्य" in form:
                v["ya"] += 1
            elif form[-3:] and any(s in form[-3:] for s in ("ने", "नि", "नः", "ना")):
                v["na"] += 1
            else:
                v["guna"] += 1
    return v.most_common(1)[0][0] if v else "-"


# Precompute per-lemma lexical attributes
LG = {l: lexgender(l) for l in by_lem}
FT = {l: femtype(l) for l in by_lem}
VR = {l: variant(l) for l in by_lem}
RE = {l: retro(l) for l in by_lem}


def eff_gender(lem, feat):
    """Effective gender: from the feature set (adjectives) else lexical (nouns)."""
    F = set(feat.split(";"))
    if "FEM" in F:
        return "f"
    if "MASC" in F:
        return "m"
    if "NEUT" in F:
        return "n"
    return LG[lem]


def keys(lem, feat):
    """Backoff key list, most specific first."""
    F = set(feat.split(";"))
    vc = vclass(lem)
    g = eff_gender(lem, feat)
    fb = FT[lem] if "FEM" in F else "-"
    nk = RE[lem]
    return [
        (feat, vc, g, nk, fb, VR[lem]),
        (feat, vc, g, nk, fb),
        (feat, vc, g, nk),
        (feat, vc, g),
        (feat, vc),
        (feat,),
    ]


# --------------------------------------------------------------------------
# Train the backoff model
# --------------------------------------------------------------------------
NL = 6
models = [collections.defaultdict(collections.Counter) for _ in range(NL)]
for lem, form, feat in rows:
    ks = keys(lem, feat)
    t = trans(lem, form)
    for j in range(NL):
        models[j][ks[j]][t] += 1


def predict_trans(lem, feat):
    ks = keys(lem, feat)
    for j in range(NL):
        if ks[j] in models[j]:
            return models[j][ks[j]].most_common(1)[0][0]
    return (0, "")


# --------------------------------------------------------------------------
# Determine, for each lemma, which feature sets to generate
# --------------------------------------------------------------------------
pos_feats = collections.defaultdict(set)         # POS -> set of feature sets
lem_pos = {}
for lem, entries in by_lem.items():
    poss = set(feat.split(";")[0] for _, feat in entries)
    lem_pos[lem] = poss
    for _, feat in entries:
        pos_feats[feat.split(";")[0]].add(feat)

# --------------------------------------------------------------------------
# Generate all (input-symbols -> output-symbols) pairs
# --------------------------------------------------------------------------
def tags(feat):
    return ["[" + c + "]" for c in feat.split(";")]


def build_pairs():
    seen = set()
    pairs = []
    for lem in by_lem:
        feats = set()
        for p in lem_pos[lem]:
            feats |= pos_feats[p]
        for feat in feats:
            key = (lem, feat)
            if key in seen:
                continue
            seen.add(key)
            rl, add = predict_trans(lem, feat)
            stem = lem[: len(lem) - rl] if rl else lem
            form = stem + add
            tg = tags(feat)
            inp = tg + list(lem)
            outp = tg + list(form)
            pairs.append((inp, outp))
    return pairs


# --------------------------------------------------------------------------
# Compile pairs into a transducer.
# Each edge carries a composite "in|out" label; the path is built by copying
# the shared prefix of lemma/form, deleting the removed suffix, and inserting
# the new suffix.  Common prefixes are merged into a trie, then minimized.
# --------------------------------------------------------------------------
def esc(sym):
    return sym.replace("\\", "\\\\").replace("|", "\\|")


def label(insym, outsym):
    if insym == outsym:
        return esc(insym)                     # identity
    return esc(insym) + "|" + esc(outsym)     # transduction / eps


def path_labels(inp, outp):
    """Composite-label sequence: copy tags, copy stem prefix, delete, insert."""
    # tags are identical prefixes of both sides
    k = 0
    while k < len(inp) and k < len(outp) and inp[k] == outp[k] and inp[k].startswith("["):
        k += 1
    labels = [label(inp[i], inp[i]) for i in range(k)]
    a, b = inp[k:], outp[k:]                   # lemma-chars vs form-chars
    p = lcp(a, b)
    for i in range(p):                         # copy shared stem
        labels.append(label(a[i], a[i]))
    for i in range(p, len(a)):                 # delete removed chars
        labels.append(label(a[i], ""))
    for i in range(p, len(b)):                 # insert new suffix chars
        labels.append(label("", b[i]))
    return labels


def build_fst(pairs):
    children = [dict()]        # children[state] = {label: child_state}
    finals = set()
    alphabet = set()

    def new_state():
        children.append(dict())
        return len(children) - 1

    for inp, outp in pairs:
        for s in inp:
            alphabet.add(s)
        for s in outp:
            alphabet.add(s)
        state = 0
        for lab in path_labels(inp, outp):
            nxt = children[state].get(lab)
            if nxt is None:
                nxt = new_state()
                children[state][lab] = nxt
            state = nxt
        finals.add(state)

    transitions = {}
    for s, arcs in enumerate(children):
        if arcs:
            transitions[str(s)] = {lab: [t] for lab, t in arcs.items()}
    fstdict = {
        "transitions": transitions,
        "alphabet": {sym: i for i, sym in enumerate(sorted(alphabet))},
        "finals": {str(s): 0.0 for s in finals},
    }
    return FST.fromdict(fstdict)


if __name__ == "__main__":
    pairs = build_pairs()
    print(f"Generated {len(pairs)} (input, output) pairs")

    fst = build_fst(pairs)
    print(f"Trie states: {len(fst.states)}")
    fst = fst.minimize()
    print(f"Minimized states: {len(fst.states)}")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(fst.to_fomastring())
    print(f"Saved FST to {OUT}")
