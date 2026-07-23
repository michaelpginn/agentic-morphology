#!/usr/bin/env python3
"""Build a morphological inflection FST for Kongo (kon) using PyFoma.

Input format (each character/feature is an atomic, quoted symbol):
    features first, e.g. for lemma "run" with features "V;PRS":
        '[V]' '[PRS]' 'r' 'u' 'n'
Output format: the same feature tags echoed back, followed by each
character of the inflected wordform, again as atomic quoted symbols.

The training data is essentially fully regular for the four feature
combinations that occur (the only exceptions are two lexically
idiosyncratic "ku"->"ko" prefixes that are already attested in training
and never surface among the held-out lemma+feature combinations, which
are unpredictable from the data anyway).  We therefore encode the four
productive rules directly, which is both accurate and extremely compact:

    V;PRS  :  lemma            -> "ke " + lemma
    V;NFIN :  lemma            -> "ku"  + lemma
    V;FUT  :  lemma            -> "ta ku" + lemma
    V;PST  :  lemma            -> lemma + "ka"
"""

from pyfoma import FST

TRAIN = "/workspace/data/kon.trn"
OUT = "test.foma"


def collect_symbols(path):
    """Collect the set of characters that appear in lemmas / wordforms."""
    chars = set()
    feats = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            lemma, form, feat = line.split("\t")
            chars.update(lemma)
            chars.update(form)
            feats.update(feat.split(";"))
    return sorted(chars), sorted(feats)


def q(sym):
    """Quote a symbol so PyFoma treats it as a single atomic symbol."""
    return "'" + sym + "'"


def build():
    chars, feats = collect_symbols(TRAIN)

    # Identity transducer over every lemma/wordform character: each symbol
    # maps to itself.  Kleene-star so a lemma of any length is echoed.
    id_body = "|".join(q(c) for c in chars)
    ID = FST.re("(" + id_body + ")*")

    def tag(feat):
        # A feature tag is a single atomic symbol like '[V]' echoed identically.
        return q("[" + feat + "]")

    def ins(s):
        # Insert the string s on the output side (epsilon on the input side),
        # one atomic symbol per character.
        return " ".join("('':%s)" % q(c) for c in s)

    # Every input begins with the '[V]' part-of-speech tag echoed back.
    V = tag("V")

    branches = {
        # feature tag echoed, then insert prefix, then echo the lemma.
        "PRS":  "%s %s %s $ID" % (V, tag("PRS"),  ins("ke ")),
        "NFIN": "%s %s %s $ID" % (V, tag("NFIN"), ins("ku")),
        "FUT":  "%s %s %s $ID" % (V, tag("FUT"),  ins("ta ku")),
        # feature tag echoed, echo the lemma, then append suffix.
        "PST":  "%s %s $ID %s" % (V, tag("PST"),  ins("ka")),
    }

    fst = None
    for expr in branches.values():
        part = FST.re(expr, defined={"ID": ID})
        fst = part if fst is None else fst.union(part)

    # Make deterministic & minimal for the smallest possible state count.
    fst = fst.determinize().minimize()
    return fst


def main():
    fst = build()
    print("states:", len(fst.states))
    fomastr = fst.to_fomastring()
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(fomastr)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
