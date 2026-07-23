#!/usr/bin/env python3
"""Build a morphological-inflection FST for the O'odham (ood) data with PyFoma.

Strategy
--------
The data is a paradigm-inflection task: every lemma appears with several
feature bundles, and the held-out items are *unseen feature combinations of
seen lemmas*.  We therefore learn, per lemma, its stem allomorphs (imperfective
sg/pl, perfective sg/pl, noun sg/pl) from whatever training rows exist for that
lemma, and realize the productive suffixes (future -ad/-d, imperative -iñ/-ñ)
with a small phonological rule set.  Missing stems are recovered cross-aspect
(the reduplication learned from one sub-paradigm transfers to another) with an
identity fallback, which empirically beats any blind reduplication rule.

For every (lemma, feature-bundle) that occurs in training we emit the gold form
verbatim; for the remaining combinations we emit the predicted form.  All
(input -> output) pairs are compiled into one transducer as an input-sharing
trie and minimized.

I/O format (as required):
    input  = feature tags (each bracketed, e.g. [V][IPFV][PL][PRS]) followed by
             each character of the lemma, every tag/character an atomic symbol.
    output = the same feature tags followed by each character of the wordform.
"""

import collections
from pyfoma import FST
from pyfoma.fst import State

DATA = "/workspace/data/ood.trn"
OUT = "test.foma"

VOW = "aeiou"
BREVE = "̆"   # combining breve  -> reduced final vowel (ĭ, ă)
RING = "̥"    # combining ring below -> consonant diacritic (d̥, s̥)

# All feature bundles seen per part of speech (drives generation of unseen combos).
VERB_FEATS = [
    "V;IPFV;SG;PRS", "V;IPFV;SG;FUT", "V;IMP;SG;PRS",
    "V;IPFV;PL;PRS", "V;IPFV;PL;FUT", "V;IMP;PL;PRS",
    "V;PRF;SG;PRS", "V;PRF;PL;PRS",
]
NOUN_FEATS = ["N;SG", "N;PL"]


# --------------------------------------------------------------------------- #
# Morphophonology of the productive suffixes                                   #
# --------------------------------------------------------------------------- #
def defective_vowel(s):
    """A stem-final reduced vowel (Vĭ / Vă) surfaces as a full vowel before a
    suffix, so drop the combining breve."""
    return s[:-1] if s.endswith(BREVE) else s


def vowel_final(s):
    return bool(s) and (s[-1] in VOW or s[-1] == ":")


def add_fut(s):
    s = defective_vowel(s)
    return s + "d" if vowel_final(s) else s + "ad"


def add_imp(s):
    s = defective_vowel(s)
    if s.endswith("s" + RING):          # s̥ -> s before the imperative suffix
        return s[:-2] + "siñ"
    return s + "ñ" if vowel_final(s) else s + "iñ"


def _tail_change(ref, target):
    k = 0
    while k < len(ref) and k < len(target) and ref[k] == target[k]:
        k += 1
    return ref[k:], target[k:]


def recover(surf, ref, kind):
    """Recover a (plural) stem from a suffixed form ``surf`` using the singular
    stem ``ref`` (= the lemma) as a paradigm template: plural and singular share
    the stem-final morphophonology, so the change ref->add_X(ref) inverts the
    same suffix on the plural form."""
    target = add_fut(ref) if kind == "fut" else add_imp(ref)
    ref_tail, tgt_tail = _tail_change(ref, target)
    if tgt_tail and surf.endswith(tgt_tail):
        return surf[: len(surf) - len(tgt_tail)] + ref_tail
    if kind == "fut":
        if surf.endswith("ad"):
            return surf[:-2]
        if surf.endswith("d"):
            return surf[:-1]
    else:
        if surf.endswith("iñ"):
            return surf[:-2]
        if surf.endswith("ñ"):
            return surf[:-1]
    return surf


# --------------------------------------------------------------------------- #
# Per-lemma stem inventory and cross-aspect reduplication transfer            #
# --------------------------------------------------------------------------- #
def learn_transform(sg, pl):
    """A singular->plural relation learned from a known pair.  We only transfer
    the reliable cases: identity (no reduplication) and pure prefixation."""
    if sg == pl:
        return ("id", None)
    if pl.endswith(sg):
        return ("prefix", pl[: len(pl) - len(sg)])
    return ("other", None)


def apply_transform(t, base):
    if base is None or t is None:
        return None
    kind, data = t
    if kind == "id":
        return base
    if kind == "prefix":
        return data + base
    return None


def invert_transform(t, pl):
    if t is None:
        return None
    kind, data = t
    if kind == "id":
        return pl
    if kind == "prefix" and pl.startswith(data):
        return pl[len(data):]
    return None


def build_stems(lem, d):
    isV = any(k.startswith("V") for k in d)
    isN = any(k.startswith("N") for k in d)
    S = {"lem": lem, "vsg": lem if isV else None, "nsg": lem if isN else None}

    vpl = None
    if "V;IPFV;PL;PRS" in d:
        vpl = d["V;IPFV;PL;PRS"]
    elif "V;IPFV;PL;FUT" in d:
        vpl = recover(d["V;IPFV;PL;FUT"], lem, "fut")
    elif "V;IMP;PL;PRS" in d:
        vpl = recover(d["V;IMP;PL;PRS"], lem, "imp")
    S["vpl"] = vpl
    S["psg"] = d.get("V;PRF;SG;PRS")
    S["ppl"] = d.get("V;PRF;PL;PRS")
    S["npl"] = d.get("N;PL")

    # Learn the reduplication transform from any available SG/PL pair,
    # preferring a clean prefix-type relation.
    cands = []
    for a, b in (("vsg", "vpl"), ("psg", "ppl"), ("nsg", "npl")):
        if S.get(a) and S.get(b):
            cands.append((S[a], S[b]))
    tsel = None
    for sg, pl in cands:
        tt = learn_transform(sg, pl)
        if tt[0] == "prefix":
            tsel = tt
            break
    if tsel is None:
        for sg, pl in cands:
            tt = learn_transform(sg, pl)
            if tt[0] == "id":
                tsel = tt
                break
    if tsel is None and cands:
        tsel = learn_transform(*cands[0])
    S["t"] = tsel
    return S


def get_vpl(S):
    if S["vpl"] is not None:
        return S["vpl"]
    base = S["vsg"] or S["lem"]
    r = apply_transform(S["t"], base)
    return r if r else base


def get_ppl(S):
    if S["ppl"] is not None:
        return S["ppl"]
    if S["psg"] is not None:
        r = apply_transform(S["t"], S["psg"])
        return r if r else S["psg"]
    return get_vpl(S)


def get_psg(S):
    if S["psg"] is not None:
        return S["psg"]
    if S["ppl"] is not None:
        r = invert_transform(S["t"], S["ppl"])
        if r is not None:
            return r
    return S["vsg"] or S["lem"]


def get_npl(S):
    base = S["nsg"] or S["lem"]
    if S["npl"] is not None:
        return S["npl"]
    if S["t"] is not None and S["t"][0] != "id":
        r = apply_transform(S["t"], base)
        if r:
            return r
    return base


def predict(feat, lem, d):
    """Predict the wordform for feature bundle ``feat`` of ``lem`` given that
    lemma's training rows ``d`` (feat -> form)."""
    if feat in d:                       # memorize anything actually attested
        return d[feat]
    S = build_stems(lem, d)
    vsg = S["vsg"] or lem
    return {
        "V;IPFV;SG;PRS": vsg,
        "V;IPFV;SG;FUT": add_fut(vsg),
        "V;IMP;SG;PRS": add_imp(vsg),
        "V;IPFV;PL;PRS": get_vpl(S),
        "V;IPFV;PL;FUT": add_fut(get_vpl(S)),
        "V;IMP;PL;PRS": add_imp(get_vpl(S)),
        "V;PRF;SG;PRS": get_psg(S),
        "V;PRF;PL;PRS": get_ppl(S),
        "N;SG": S["nsg"] or lem,
        "N;PL": get_npl(S),
    }.get(feat, lem)


# --------------------------------------------------------------------------- #
# FST construction                                                             #
# --------------------------------------------------------------------------- #
def feat_tags(feat):
    return ["[" + t + "]" for t in feat.split(";")]


def build_fst(entries):
    """entries: iterable of (input_symbols, output_symbols).

    Compact transducer:
      * input side: a prefix trie over feature-tag identity transitions
        (tag:tag) followed by lemma-character read transitions (char:epsilon),
        so shared feature/lemma prefixes reuse states;
      * output side: each leaf feeds an interned *suffix* trie of the wordform
        emitted as (epsilon:char) transitions, so shared wordform suffixes
        (including the single common final state) reuse states.
    """
    fst = FST()
    root = fst.initialstate
    alphabet = set()

    # --- output suffix trie (built from the end; keyed by the remaining tail) ---
    final = State()
    fst.states.add(final)
    fst.finalstates.add(final)
    final.finalweight = 0.0
    suffix_state = {(): final}

    def out_chain(chars):
        """Return the state from which emitting ``chars`` (a tuple) leads to
        final, creating interned suffix states as needed."""
        key = tuple(chars)
        st = suffix_state.get(key)
        if st is not None:
            return st
        nxt = out_chain(key[1:])
        st = State()
        fst.states.add(st)
        st.add_transition(nxt, ("", key[0]), 0.0)
        alphabet.add(key[0])
        suffix_state[key] = st
        return st

    # --- input prefix trie ---
    children = collections.defaultdict(dict)

    def read_step(cur, lbl):
        nxt = children[id(cur)].get(lbl)
        if nxt is None:
            nxt = State()
            fst.states.add(nxt)
            children[id(cur)][lbl] = nxt
            cur.add_transition(nxt, lbl, 0.0)
        return nxt

    for in_syms, out_syms in entries:
        cur = root
        for sym in in_syms["tags"]:
            alphabet.add(sym)
            cur = read_step(cur, (sym, sym))
        for ch in in_syms["chars"]:
            alphabet.add(ch)
            cur = read_step(cur, (ch, ""))
        # link the input-trie leaf into the shared output suffix trie via an
        # epsilon:epsilon transition, so identical wordforms/suffixes are shared.
        start = out_chain(out_syms)
        if ("", "") not in children[id(cur)]:
            cur.add_transition(start, ("", ""), 0.0)
            children[id(cur)][("", "")] = start

    fst.alphabet = alphabet
    return fst


def main():
    rows = [l.rstrip("\n").split("\t") for l in open(DATA, encoding="utf-8") if l.strip()]
    by = collections.defaultdict(dict)
    for lem, form, feat in rows:
        by[lem][feat] = form

    entries = []
    seen_inputs = set()
    for lem, d in by.items():
        feats = []
        if any(k.startswith("V") for k in d):
            feats += VERB_FEATS
        if any(k.startswith("N") for k in d):
            feats += NOUN_FEATS
        for feat in feats:
            key = (feat, lem)
            if key in seen_inputs:
                continue
            seen_inputs.add(key)
            out = predict(feat, lem, d)
            tags = feat_tags(feat)
            # tags are emitted by the identity tag-transitions; the output tail
            # carries only the wordform characters.
            entries.append((
                {"tags": tags, "chars": list(lem)},
                list(out),
            ))

    fst = build_fst(entries)
    print(f"Built trie: {len(fst.states)} states, {len(entries)} entries")

    # Minimize for compactness (secondary goal); keep the trie if it changes behavior.
    compact = fst
    try:
        m = fst.copy_filtered() if hasattr(fst, "copy_filtered") else fst
        m = fst
        m.epsilon_remove()
        m.determinize()
        m.minimize()
        compact = m
        print(f"After minimize: {len(compact.states)} states")
    except Exception as e:
        print("Minimize skipped:", e)
        compact = fst

    # Sanity check on training data (both variants) before committing.
    ok = 0
    tot = 0
    for lem, d in by.items():
        for feat, gold in d.items():
            tot += 1
            inp = "".join(feat_tags(feat)) + lem
            outs = list(compact.generate(inp))
            exp = "".join(feat_tags(feat)) + gold
            if exp in outs:
                ok += 1
    print(f"Training reproduction (minimized): {ok}/{tot}")
    if ok < tot:
        # fall back to the trie if minimization broke anything
        ok2 = 0
        for lem, d in by.items():
            for feat, gold in d.items():
                inp = "".join(feat_tags(feat)) + lem
                if ("".join(feat_tags(feat)) + gold) in list(fst.generate(inp)):
                    ok2 += 1
        print(f"Training reproduction (trie): {ok2}/{tot}")
        if ok2 >= ok:
            compact = fst

    foma = compact.to_fomastring()
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(foma)
    print(f"Wrote {OUT}: {len(compact.states)} states, {len(foma)} bytes")


if __name__ == "__main__":
    main()
