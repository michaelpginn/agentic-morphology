#!/usr/bin/env python3
"""Build an FST for Ingrian (izh) morphological inflection with PyFoma.

Strategy
--------
The dev/test sets only require generalizing to *unseen (lemma, feature)
combinations* of lemmas that were already seen in training.  So for every
lemma we learn its paradigm stems (weak/strong singular and plural oblique
stems, plus a per-lemma vowel-lengthening flag) from whatever slots are
present in the training data, and generate the missing slots with a small set
of productive case/number endings.  Seen slots are kept verbatim (gold).

The whole 22-cell case/number grid is generated for every lemma and compiled
into a single transducer that maps

    [POS][CASE][NUM] + lemma-chars   ->   [POS][CASE][NUM] + wordform-chars

with each feature tag and each character as an atomic (quoted) symbol.
"""

import collections
from pyfoma import FST

TRAIN = "/workspace/data/izh.trn"
OUT = "/workspace/test.foma"

# --------------------------------------------------------------------------
# 1. Read training data
# --------------------------------------------------------------------------
rows = [l.split("\t") for l in open(TRAIN, encoding="utf-8").read().splitlines() if l.strip()]
para = collections.defaultdict(dict)   # lemma -> {(case,num): surface}
pos_of = {}                            # lemma -> POS tag (N / ADJ)
cases_nums = set()                     # all (case,num) pairs seen
for lem, surf, feat in rows:
    pos, case, num = feat.split(";")
    para[lem][(case, num)] = surf
    pos_of[lem] = pos
    cases_nums.add((case, num))

# --------------------------------------------------------------------------
# 2. Morphological generator
# --------------------------------------------------------------------------
V = "aeiouäöüy"

def is_long(s):        # ends in a doubled vowel (e.g. "kalaa")
    return len(s) >= 2 and s[-1] in V and s[-2] == s[-1]

def is_diph_i(s):      # ends in a diphthong closing in -i (e.g. "karhuloi")
    return s.endswith("i") and len(s) >= 2 and s[-2] in V and s[-2] != "i"

def lengthen(s):       # lengthen the final short vowel (skip long vowels / -i diphthongs)
    if not s or s[-1] not in V:
        return s
    if is_long(s) or is_diph_i(s):
        return s
    return s + s[-1]

def delen(s):          # undo a final vowel lengthening
    return s[:-1] if is_long(s) else s

def collapse_i(s):     # collapse a lengthened plural -ii back to -i
    return s[:-1] if s.endswith("ii") else s

def harmony(s):        # vowel harmony: front (ä) vs back (a)
    return "ä" if any(c in "äöüy" for c in s) else "a"

def strp(v, suf):
    if v is None:
        return None
    return v[:-len(suf)] if (suf == "" or v.endswith(suf)) else None

def vote(d, cands, post=lambda x: x):
    """Extract a stem by majority vote over several 'strip this ending' donors."""
    c = collections.Counter()
    for k, suf in cands:
        r = strp(d.get(k), suf)
        if r is not None:
            c[post(r)] += 1
    return c.most_common(1)[0][0] if c else None

def weak_sg(d):
    return vote(d, [(("GEN", "SG"), "n"), (("IN+ABL", "SG"), "st"),
                    (("TRANS", "SG"), "ks"), (("AT+ABL", "SG"), "lt"),
                    (("AT+ALL", "SG"), "lle")])

def weak_pl(d):
    return vote(d, [(("IN+ABL", "PL"), "st"), (("TRANS", "PL"), "ks"),
                    (("AT+ABL", "PL"), "lt"), (("AT+ALL", "PL"), "lle")])

def strong_pl(d):
    v = vote(d, [(("GEN", "PL"), "n"), (("ESS", "PL"), "n")], collapse_i)
    return v if v else vote(d, [(("IN+ALL", "PL"), "he")])

def illative_stem(v):  # recover the strong stem from an illative-sg surface
    if v.endswith("sse"):
        return None
    if len(v) >= 3 and v[-2] == "h" and v[-1] in V and v[-3] in V:  # -hV
        return v[:-2]
    if is_long(v):
        return v[:-1]
    return None

def part_stem(v):      # recover the strong stem from a partitive-sg surface
    if v.endswith("t") or v.endswith("ta") or v.endswith("tä"):
        return None
    if v and v[-1] in V:
        return v[:-1]
    return None

def strong_sg(d, lem):
    """Strong (often geminated) singular stem, learned from strong-grade slots."""
    base = lem if lem[-1] in V else (weak_sg(d) or lem)
    cands = [
        delen(strp(d.get(("ESS", "SG")), "n") or "") if d.get(("ESS", "SG")) else None,
        illative_stem(d["IN+ALL", "SG"]) if ("IN+ALL", "SG") in d else None,
        part_stem(d["PRT", "SG"]) if ("PRT", "SG") in d else None,
    ]
    for cand in cands:
        if cand and len(cand) >= len(base):
            return cand
    return base

def sg_lengthens(d, ws):
    """Does this lemma lengthen the stem vowel before -z / -l (inessive/adessive)?"""
    for key, suf in [(("AT+ESS", "SG"), "l"), (("IN+ESS", "SG"), "z")]:
        v = d.get(key)
        if v is None or ws is None:
            continue
        if v == ws + suf:
            return False
        if v == lengthen(ws) + suf:
            return True
    return True

def generate(lem, case, num, d):
    ws = weak_sg(d)
    ss = strong_sg(d, lem)
    wp = weak_pl(d)
    sp = strong_pl(d) or wp
    h = harmony(lem)
    L = sg_lengthens(d, ws)
    lw = (lambda s: lengthen(s)) if L else (lambda s: s)
    if num == "SG":
        if case == "NOM":
            return lem
        if case == "ESS":
            return lengthen(ss) + "n"
        if case == "PRT":
            return (ss + "t" + h) if is_long(ss) else (ss + h)
        if case == "IN+ALL":
            return ss + "h" + ss[-1] if is_long(ss) else lengthen(ss)
        if ws is None:
            return None
        if case == "GEN":    return ws + "n"
        if case == "TRANS":  return ws + "ks"
        if case == "IN+ESS": return lw(ws) + "z"
        if case == "IN+ABL": return ws + "st"
        if case == "AT+ESS": return lw(ws) + "l"
        if case == "AT+ALL": return ws + "lle"
        if case == "AT+ABL": return ws + "lt"
    else:
        if case == "NOM":
            return (ws or lem) + "t"
        if case in ("GEN", "ESS"):
            return lengthen(sp) + "n" if sp else None
        if wp is None and sp is None:
            return None
        wp = wp or sp
        if case == "IN+ALL":
            return sp + "he" if is_diph_i(sp) else lengthen(sp)
        if case == "IN+ESS": return lengthen(wp) + "z"
        if case == "TRANS":  return wp + "ks"
        if case == "IN+ABL": return wp + "st"
        if case == "AT+ESS": return lengthen(wp) + "l"
        if case == "AT+ALL": return wp + "lle"
        if case == "AT+ABL": return wp + "lt"
        if case == "PRT":
            b = sp
            if is_diph_i(b):
                return b + "t" + h if len(b) <= 3 else b[:-1] + "j" + h
            return b + h
    return None

# --------------------------------------------------------------------------
# 3. Build the (input, output) pair table over the full grid
# --------------------------------------------------------------------------
def predict(lem, case, num):
    d = para[lem]
    if (case, num) in d:          # seen slot -> keep gold
        return d[(case, num)]
    surf = generate(lem, case, num, d)
    return surf if surf else lem  # never emit nothing

pairs = []
for lem in sorted(para):
    pos = pos_of[lem]
    for (case, num) in sorted(cases_nums):
        surf = predict(lem, case, num)
        tags = ["[%s]" % pos, "[%s]" % case, "[%s]" % num]
        inp = tags + list(lem)
        out = tags + list(surf)
        pairs.append((inp, out))

# --------------------------------------------------------------------------
# 4. Compile into a single minimal FST
# --------------------------------------------------------------------------
def q(sym):
    """Quote a symbol so PyFoma treats it as a single atomic symbol."""
    return "'" + sym.replace("\\", "\\\\").replace("'", "\\'") + "'"

def seq(symbols):
    return " ".join(q(s) for s in symbols)

regex = " | ".join("(%s):(%s)" % (seq(i), seq(o)) for i, o in pairs)
fst = FST.re(regex)
fst = fst.minimize()

foma = fst.to_fomastring()
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(foma)

print("pairs: %d" % len(pairs))
print("states: %d" % len(fst.states))
print("saved -> %s" % OUT)
