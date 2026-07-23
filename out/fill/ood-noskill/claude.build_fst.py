#!/usr/bin/env python3
"""Build a morphological-inflection FST for the O'odham (ood) data with PyFoma.

Strategy
--------
Every lemma in the dev/test sets is also present in the training data (only the
lemma+feature *combinations* are unseen).  We therefore learn, per lemma, a
small analogical paradigm: a singular stem and a plural stem are recovered from
whatever training forms exist for that lemma, and the eight verbal / two nominal
cells are produced by attaching regular suffixes (and, where no same-number form
exists, by (de)reduplication).  For cells that *are* attested in training we
simply reuse the gold form.

The resulting (input -> output) pairs are compiled as quoted, atomic-symbol
transducer paths, unioned and minimized, and written out as a foma string.
Input/output are formatted feature-tags-first, e.g.
    '[V]''[IPFV]''[SG]''[PRS]''r''u''n'  ->  '[V]''[IPFV]''[SG]''[PRS]''r''a''n'
"""

import collections
import re
from pyfoma import FST

TRAIN = "/workspace/data/ood.trn"
OUT_FOMA = "test.foma"

# ----------------------------------------------------------------------------
# Phonology helpers: split a string into "units" (a base char plus any trailing
# combining marks / length / breve so multi-codepoint graphemes stay together).
# ----------------------------------------------------------------------------
VOWL = set("aeiou")
# whispered/reduced vowels are written base-vowel + COMBINING BREVE (U+0306);
# they surface as the plain vowel before a suffix.
BREVE = {"̆": "", "ĭ": "i", "ă": "a", "ŏ": "o", "ŭ": "u", "ĕ": "e"}


def units(s):
    out = []
    for ch in s:
        if out and (0x300 <= ord(ch) <= 0x36F or ch in ":̥̆"):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def isvow(u):
    return u[0].lower() in VOWL


def short(u):
    return u.replace(":", "")


def debreve(u):
    return "".join(BREVE.get(c, c) for c in u)


def denorm(stem):
    """A reduced (breve) vowel at the end surfaces as a plain vowel before a
    suffix, e.g. s-mohogĭ + iñ -> s-mohogiñ."""
    u = units(stem)
    if u:
        u[-1] = debreve(u[-1])
    return "".join(u)


def parse(base):
    u = units(base)
    i = 0
    while i < len(u) and not isvow(u[i]):
        i += 1
    if i >= len(u):
        return None
    return u[:i], u[i], u[i + 1:]


def strip_pref(b):
    m = re.match(r"^([sS]-)", b)
    return (m.group(1), b[len(m.group(1)):]) if m else ("", b)


# ----------------------------------------------------------------------------
# (De)reduplication used only when no same-number form is available.
# ----------------------------------------------------------------------------
def reduplicate(base):
    pre, base = strip_pref(base)
    p = parse(base)
    if not p:
        return pre + base
    c1, v1, rest = p
    long = ":" in v1
    sv = short(v1)
    open_syl = len(rest) >= 2 and (not isvow(rest[0])) and isvow(rest[1])
    if long:
        if len(rest) == 1:
            body = "".join(c1) + v1 + "".join(c1) + "".join(rest)
        else:
            body = "".join(c1) + sv + "".join(c1) + "".join(rest)
        return pre + body
    if open_syl:
        return pre + "".join(c1) + sv + "".join(c1) + "".join(rest)
    return pre + "".join(c1) + sv + "".join(c1) + sv + "".join(rest)


def dereduplicate(pl):
    pre, base = strip_pref(pl)
    u = units(base)
    p = parse(base)
    if not p:
        return pre + base
    c1, v1, rest = p
    n = len(c1)
    if len(u) >= 2 * n + 2 and u[:n] == u[n + 1:2 * n + 1] and short(u[n]) == short(u[2 * n + 1]):
        return pre + "".join(u[n + 1:])
    if len(u) >= 2 * n + 1 and u[:n] == u[n + 1:2 * n + 1]:
        return pre + "".join(u[:n + 1] + u[2 * n + 1:])
    return pre + base


# ----------------------------------------------------------------------------
# Suffixation.
# ----------------------------------------------------------------------------
def add_imp(stem):
    stem = denorm(stem)
    u = units(stem)
    last = u[-1]
    if last == "s̥":                       # s̥ -> s before the -iñ imperative
        return "".join(u[:-1]) + "siñ"
    return stem + "ñ" if isvow(last) else stem + "iñ"


def fut(stem):
    stem = denorm(stem)
    u = units(stem)
    return stem + "d" if isvow(u[-1]) else stem + "ad"


def perf(stem):
    u = units(stem)
    if not u:
        return stem
    if isvow(u[-1]):
        return stem
    if u[-1] == "d" and len(u) >= 2 and u[-2] == "a":   # ...ad / ...mad keeps
        return stem
    return "".join(u[:-1])                                # else drop final C


def strip_ad(x):
    return x[:-2] if x.endswith("ad") else x


def strip_imp(x):
    if x.endswith("iñ"):
        return x[:-2]
    if x.endswith("ñ"):
        return x[:-1]
    return x


# ----------------------------------------------------------------------------
# Stem recovery + cell prediction.
# ----------------------------------------------------------------------------
def get_sg_only(v):
    if "V;IPFV;SG;PRS" in v:
        return v["V;IPFV;SG;PRS"]
    if "N;SG" in v:
        return v["N;SG"]
    if "V;IPFV;SG;FUT" in v:
        return strip_ad(v["V;IPFV;SG;FUT"])
    if "V;IMP;SG;PRS" in v:
        return strip_imp(v["V;IMP;SG;PRS"])
    return None


def rawpl(v):
    if "V;IPFV;PL;PRS" in v:
        return v["V;IPFV;PL;PRS"]
    if "N;PL" in v:
        return v["N;PL"]
    if "V;IPFV;PL;FUT" in v:
        return strip_ad(v["V;IPFV;PL;FUT"])
    if "V;IMP;PL;PRS" in v:
        return strip_imp(v["V;IMP;PL;PRS"])
    return None


def sgstem(v):
    s = get_sg_only(v)
    if s is not None:
        return s
    pl = rawpl(v)                    # only PL attested -> strip reduplication
    return dereduplicate(pl) if pl is not None else None


def plstem(v):
    pl = rawpl(v)
    if pl is not None:
        return pl
    return get_sg_only(v)            # only SG attested -> identity is safest


def predict(feat, v):
    if feat in ("N;SG", "V;IPFV;SG;PRS"):
        return sgstem(v)
    if feat in ("N;PL", "V;IPFV;PL;PRS"):
        return plstem(v)
    if feat == "V;IPFV;SG;FUT":
        s = sgstem(v); return fut(s) if s else None
    if feat == "V;IPFV;PL;FUT":
        s = plstem(v); return fut(s) if s else None
    if feat == "V;IMP;SG;PRS":
        s = sgstem(v); return add_imp(s) if s else None
    if feat == "V;IMP;PL;PRS":
        s = plstem(v); return add_imp(s) if s else None
    if feat == "V;PRF;SG;PRS":
        s = sgstem(v); return perf(s) if s else None
    if feat == "V;PRF;PL;PRS":
        s = plstem(v); return perf(s) if s else None
    return None


# ----------------------------------------------------------------------------
# FST construction.
# ----------------------------------------------------------------------------
N_COMBOS = ["N;SG", "N;PL"]
V_COMBOS = ["V;IPFV;SG;PRS", "V;IPFV;PL;PRS", "V;IPFV;SG;FUT", "V;IPFV;PL;FUT",
            "V;IMP;SG;PRS", "V;IMP;PL;PRS", "V;PRF;SG;PRS", "V;PRF;PL;PRS"]


def esc(sym):
    """Escape a symbol for use inside a PyFoma single-quoted atom."""
    return sym.replace("\\", "\\\\").replace("'", "\\'")


def atoms_feat(feat):
    # each ';'-separated feature becomes one bracketed atomic symbol
    return ["'[%s]'" % esc(t) for t in feat.split(";")]


def atoms_word(w):
    # each character (codepoint) becomes one atomic symbol
    return ["'%s'" % esc(c) for c in w]


ALL_COMBOS = N_COMBOS + V_COMBOS
# stem-transform class per feature (PL stems fall back to identity in the
# unseen-lemma regime, so PL and SG share the same transform).
TRANSFORM = {
    "N;SG": "id", "N;PL": "id",
    "V;IPFV;SG;PRS": "id", "V;IPFV;PL;PRS": "id",
    "V;IPFV;SG;FUT": "fut", "V;IPFV;PL;FUT": "fut",
    "V;IMP;SG;PRS": "imp", "V;IMP;PL;PRS": "imp",
    "V;PRF;SG;PRS": "prf", "V;PRF;PL;PRS": "prf",
}

# codepoint classes for the general (rule-based) transducer.  The combining
# breve (whispered vowel) is grouped with the vowels for suffix purposes.
VOWEL_CP = set("aeiouAO:")          # ':' (length) behaves as a vowel
BREVE_CP = "̆"                       # U+0306 combining breve


def clean(fst):
    return fst.epsilon_remove().determinize().minimize()


def build_general(chars, seen_lemmas):
    """A rule-based transducer that inflects any lemma over the alphabet, built
    only from concatenation / intersection / difference / union so that it
    serializes losslessly (PyFoma's compose does not round-trip at scale).
    Its input is restricted to lemmas NOT in the training lexicon, so it is
    disjoint from the memorized table and the union stays functional."""
    vowels = sorted(c for c in chars if c in VOWEL_CP or c == BREVE_CP)
    cons = sorted(c for c in chars if c not in VOWEL_CP and c != BREVE_CP)

    def acc(members):                      # identity acceptor over a char set
        return "(" + "|".join("'%s'" % esc(c) for c in members) + ")"

    charstar = FST.re(acc(chars) + "*")
    # acceptor of every seen lemma (each as a char sequence)
    seen_re = " | ".join("(" + " ".join("'%s'" % esc(c) for c in lem) + ")"
                         for lem in seen_lemmas)
    seen = FST.re(seen_re)
    notseen = clean(charstar.difference(seen))          # unseen lemmas only

    end_vow = FST.re(acc(chars) + "* " + acc(vowels))   # ...vowel/breve final
    end_cons = FST.re(acc(chars) + "* " + acc(cons))    # ...consonant final
    ns_vow = clean(notseen.intersection(end_vow))
    ns_cons = clean(notseen.intersection(end_cons))

    ins = lambda s: FST.re(" ".join("'':'%s'" % esc(c) for c in s))

    def stem(kind):
        if kind == "id" or kind == "prf":   # copy the lemma unchanged
            return notseen
        if kind == "fut":                   # +d after vowel, +ad after consonant
            return ns_vow.concatenate(ins("d")).union(ns_cons.concatenate(ins("ad")))
        if kind == "imp":                   # +ñ after vowel, +iñ after consonant
            return ns_vow.concatenate(ins("ñ")).union(ns_cons.concatenate(ins("iñ")))
        raise ValueError(kind)

    stem_cache = {}
    result = None
    for feat in ALL_COMBOS:
        fp = FST.re(" ".join("'[%s]':'[%s]'" % (esc(t), esc(t))
                             for t in feat.split(";")))
        kind = TRANSFORM[feat]
        if kind not in stem_cache:
            stem_cache[kind] = stem(kind)
        branch = fp.concatenate(stem_cache[kind])
        result = branch if result is None else result.union(branch)
    return clean(result)


def build():
    lem = collections.defaultdict(dict)
    pos = collections.defaultdict(set)
    with open(TRAIN, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            lemma, wf, feat = line.split("\t")
            lem[lemma][feat] = wf
            pos[lemma].add(feat.split(";")[0])

    chars = set()
    for lemma, forms in lem.items():
        chars.update(lemma)
        for w in forms.values():
            chars.update(w)

    paths = []
    for lemma, forms in lem.items():
        combos = []
        if "N" in pos[lemma]:
            combos += N_COMBOS
        if "V" in pos[lemma]:
            combos += V_COMBOS
        for feat in combos:
            out = forms[feat] if feat in forms else predict(feat, forms)
            if out is None:
                continue
            fa = atoms_feat(feat)
            src = " ".join(fa + atoms_word(lemma))
            dst = " ".join(fa + atoms_word(out))
            paths.append("(%s):(%s)" % (src, dst))

    # --- memorized per-lemma paradigm (M) ---------------------------------
    # A single huge alternation is pathologically slow to parse, so compile in
    # modest batches, then union + minimize.
    BATCH = 150
    fsts = [FST.re(" | ".join(paths[i:i + BATCH]))
            for i in range(0, len(paths), BATCH)]
    M = fsts[0]
    for g in fsts[1:]:
        M = M.union(g)
    M = M.determinize().minimize()

    # --- general rule-based fallback (G) for lemmas absent from training -----
    # G's input is restricted to unseen lemmas, so M and G are input-disjoint
    # and their union is functional.
    G = build_general(chars, sorted(lem.keys()))
    fst = clean(M.union(G))

    print("paths: %d   M-states: %d   G-states: %d   final-states: %d"
          % (len(paths), len(M.states), len(G.states), len(fst.states)))
    return fst


def main():
    fst = build()
    foma_str = fst.to_fomastring()
    with open(OUT_FOMA, "w", encoding="utf-8") as fh:
        fh.write(foma_str)
    print("wrote %s" % OUT_FOMA)


if __name__ == "__main__":
    main()
