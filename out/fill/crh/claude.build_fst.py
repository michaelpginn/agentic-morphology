#!/usr/bin/env python3
"""Build a Crimean Tatar (crh) morphological inflection FST with PyFoma.

Input  side:  the morphological features (as [TAG] symbols) followed by the
              lemma characters, every symbol quoted so it is atomic, e.g.
              '[N]''[DAT]''i''k''l''i''m'
Output side:  the same feature tags followed by the inflected wordform
              characters, again each quoted, e.g.
              '[N]''[DAT]''i''k''l''i''m''g''e'

Strategy
--------
Crimean Tatar case/tense inflection is almost fully productive Turkic
morphology (vowel harmony + consonant voicing assimilation).  Because the task
only requires generalisation to unseen *lemma+feature combinations* (never to
completely unseen lemmas), we:

  1. Learn, per lemma, a small set of harmony/voicing parameters from every
     training row of that lemma (low vowel a/e, high vowel i/i, the dental
     d/t and the velar g/q/g/k of its suffixes).  These parameters transfer
     across the case slots the lemma was never seen in.
  2. For a (lemma, feature) pair that WAS observed in training we simply reuse
     the gold wordform (handles the handful of truly suppletive stems).
  3. For every other slot we synthesise the wordform with productive rules
     driven by the learned parameters.

We then enumerate every (lemma, feature-combo) allowed by the lemma's part of
speech, build the transducer with a right-linear grammar, and let foma
determinise + minimise it so the saved machine is state-minimal for exactly
this mapping.
"""

from collections import defaultdict, Counter
from pyfoma import FST

TRAIN = "/workspace/data/crh.trn"
OUT = "test.foma"

# ---------------------------------------------------------------------------
# Phonology
# ---------------------------------------------------------------------------
BACK = set("aıou")          # a ı o u
FRONT = set("eiöüâ")  # e i ö ü â  (â patterns as front here)
VOWELS = BACK | FRONT
VOICELESS = set("pçtkqf sşh".replace(" ", ""))  # p ç t k q f s ş h


def last_vowel(s):
    for ch in reversed(s):
        if ch.lower() in VOWELS:
            return ch.lower()
    return None


def phon_back(s):
    lv = last_vowel(s)
    return lv in BACK if lv is not None else True


def phon_voiceless(s):
    for ch in reversed(s):
        if ch == " ":
            continue
        return ch.lower() in VOICELESS
    return False


# Feature-combos each part of speech may take (as seen in the data).
POS_COMBOS = {
    "N":   ["N;NOM", "N;ACC", "N;GEN", "N;DAT", "N;LOC", "N;ABL"],
    "ADJ": ["ADJ;NOM", "ADJ;ACC", "ADJ;GEN", "ADJ;DAT", "ADJ;LOC", "ADJ;ABL"],
    "V":   ["V;NFIN", "V;SG;3;PRS", "V;SG;3;PST", "V;IMP;SG;2"],
}


# ---------------------------------------------------------------------------
# Productive generation from learned per-lemma parameters
# ---------------------------------------------------------------------------
def generate(lemma, feat, P):
    pos = feat.split(";")[0]
    case = feat.split(";")[-1]
    lo, hi = P["lowV"], P["highV"]
    if pos in ("N", "ADJ"):
        if case == "NOM":
            return lemma
        if case == "ACC":
            return lemma + "n" + hi
        if case == "GEN":
            return lemma + "n" + hi + "ñ"          # ñ
        if P["buffer"]:
            # Possessive (izafet) stems take a pronominal -n- buffer and drop
            # the dative velar:  el işi -> el işine, don mayı -> don mayında.
            if case == "DAT":
                return lemma + "n" + lo
            if case == "LOC":
                return lemma + "n" + P["dent"] + lo
            if case == "ABL":
                return lemma + "n" + P["dent"] + lo + "n"
        else:
            if case == "DAT":
                return lemma + P["velar"] + lo
            if case == "LOC":
                return lemma + P["dent"] + lo
            if case == "ABL":
                return lemma + P["dent"] + lo + "n"
    if pos == "V":
        if lemma.endswith("maq"):
            st, back = lemma[:-3], True
        elif lemma.endswith("mek"):
            st, back = lemma[:-3], False
        else:
            st, back = lemma, (lo == "a")
        st_voiceless = bool(st) and st[-1].lower() in VOICELESS
        if feat == "V;NFIN":
            return lemma
        if feat == "V;IMP;SG;2":
            return st
        if feat == "V;SG;3;PST":
            # Back stems keep -dı even after voiceless; only front voiceless -> -ti.
            d = "t" if (st_voiceless and not back) else "d"
            return st + d + ("ı" if back else "i")
        if feat == "V;SG;3;PRS":
            if st and st[-1].lower() in VOWELS:
                return st + "y"
            return st + ("a" if back else "e")
    return lemma


def learn_params(rows, lemma):
    """Infer per-lemma suffix parameters from that lemma's training rows."""
    lowV, highV, dent, velar = Counter(), Counter(), Counter(), Counter()
    buf = Counter()
    for _, wf, feat in rows:
        pos, case = feat.split(";")[0], feat.split(";")[-1]
        if pos not in ("N", "ADJ") or not wf.startswith(lemma):
            continue
        suf = wf[len(lemma):]
        if not suf:
            continue
        sv = last_vowel(suf)
        buffered = case in ("DAT", "LOC", "ABL") and suf[0] == "n"
        if case in ("DAT", "LOC", "ABL"):
            buf[buffered] += 1
        if case in ("DAT", "LOC", "ABL") and sv in ("a", "e"):
            lowV[sv] += 1
        if case in ("ACC", "GEN") and sv in ("ı", "i"):
            highV[sv] += 1
        if buffered:
            if len(suf) > 1 and suf[1] in "td":    # dental in -nda-/-nde-
                dent[suf[1]] += 1
        else:
            if case in ("LOC", "ABL") and suf[0] in "td":
                dent[suf[0]] += 1
            if case == "DAT" and suf[0] in "qkğg":     # q k ğ g
                velar[suf[0]] += 1

    pback, pvl = phon_back(lemma), phon_voiceless(lemma)
    if lowV:
        back = lowV.most_common(1)[0][0] == "a"
    elif highV:
        back = highV.most_common(1)[0][0] == "ı"
    else:
        back = pback
    lo = lowV.most_common(1)[0][0] if lowV else ("a" if back else "e")
    hi = highV.most_common(1)[0][0] if highV else ("ı" if back else "i")
    dt = dent.most_common(1)[0][0] if dent else ("t" if pvl else "d")
    if velar:
        ve = velar.most_common(1)[0][0]
    else:
        b2 = lo == "a"
        ve = ("q" if b2 else "k") if pvl else ("ğ" if b2 else "g")
    buffer = buf.most_common(1)[0][0] if buf else False
    return {"lowV": lo, "highV": hi, "dent": dt, "velar": ve, "buffer": buffer}


# ---------------------------------------------------------------------------
# Symbol encoding
# ---------------------------------------------------------------------------
def feat_tags(feat):
    return ["[" + f + "]" for f in feat.split(";")]


def input_symbols(feat, lemma):
    return feat_tags(feat) + list(lemma)


def output_symbols(feat, wordform):
    return feat_tags(feat) + list(wordform)


def quote(syms):
    """Join symbols as quoted atomic tokens, e.g. ['[N]','a'] -> "'[N]''a'"."""
    return "".join("'" + s.replace("'", r"\'") + "'" for s in syms)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def main():
    rows = [l.rstrip("\n").split("\t") for l in open(TRAIN, encoding="utf-8") if l.strip()]

    by_lemma = defaultdict(list)
    for r in rows:
        by_lemma[r[0]].append(r)

    grammar_rules = []
    alphabet = set()
    seen_inputs = set()

    for lemma, lrows in by_lemma.items():
        params = learn_params(lrows, lemma)
        observed = {feat: wf for _, wf, feat in lrows}
        pos_set = {feat.split(";")[0] for _, _, feat in lrows}
        combos = []
        for pos in pos_set:
            combos.extend(POS_COMBOS.get(pos, []))

        for feat in combos:
            wf = observed.get(feat)
            if wf is None:
                wf = generate(lemma, feat, params)
            ins = input_symbols(feat, lemma)
            outs = output_symbols(feat, wf)
            key = tuple(ins)
            if key in seen_inputs:      # guard against accidental duplicates
                continue
            seen_inputs.add(key)
            alphabet |= set(ins) | set(outs)
            grammar_rules.append(((quote(ins), quote(outs)), "#"))

    print(f"lemmas={len(by_lemma)}  pairs={len(grammar_rules)}  symbols={len(alphabet)}")

    grammar = {"Start": grammar_rules}
    fst = FST.rlg(grammar, "Start", multichar_symbols=alphabet)
    print(f"raw states={len(fst.states)}")
    fst = fst.determinize_as_dfa().minimize()
    print(f"minimized states={len(fst.states)}")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(fst.to_fomastring("crh"))
    print(f"wrote {OUT}")

    # ---- quick self-check against the training data ----
    ok = 0
    for lemma, wf, feat in rows:
        inp = "".join(input_symbols(feat, lemma))
        exp = "".join(output_symbols(feat, wf))
        if exp in set(fst.generate(inp)):
            ok += 1
    print(f"train reproduction: {ok}/{len(rows)} = {ok/len(rows):.4f}")


if __name__ == "__main__":
    main()
