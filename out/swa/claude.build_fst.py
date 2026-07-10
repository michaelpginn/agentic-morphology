#!/usr/bin/env python3
"""Build a morphological inflection FST for Swahili verbs using PyFoma.

The training data is a three-column format: lemma, inflected form, and a
semicolon-separated list of morphological features.

Analysis of the training data shows that every inflected form is exactly a
feature-determined *prefix* concatenated with the (unchanged) lemma stem:

    inflected_form == prefix(features) + lemma

and the mapping from a full feature combination to its prefix is completely
deterministic (49 feature combinations, each with a single prefix).

The FST therefore:
  * reads the feature tags (as atomic, quoted [FEAT] symbols) and copies them
    unchanged to the output,
  * inserts the feature-determined prefix characters on the output tape, and
  * copies each lemma character through unchanged.

Input  format:  [F1][F2]...[Fn] c1 c2 ... cm   (features then lemma chars)
Output format:  [F1][F2]...[Fn] p1 ... pk c1 ... cm  (features then inflected chars)

Every symbol (each feature tag and each character) is made atomic via PyFoma's
single-quote quoting.

We build the FST as:

    ( union over feature-combos of ( tag-identity . prefix-insertion ) ) . COPY

where COPY copies an arbitrary lemma through unchanged.  Sharing the single
COPY tail and running minimize() keeps the machine compact.
"""

from pyfoma import FST

TRAIN = "/workspace/data/swa.trn"
OUT = "test.foma"


def q(sym: str) -> str:
    """Quote a symbol so PyFoma treats it as a single atomic symbol."""
    return "'" + sym + "'"


def main():
    combo2prefix = {}   # ordered feature tuple -> prefix string
    alphabet = set()

    with open(TRAIN, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            lemma, form, feats = line.split("\t")
            assert form.endswith(lemma), (lemma, form)
            prefix = form[: len(form) - len(lemma)]
            feat_tuple = tuple(feats.split(";"))
            if feat_tuple in combo2prefix:
                assert combo2prefix[feat_tuple] == prefix, (
                    "ambiguous prefix for", feat_tuple)
            else:
                combo2prefix[feat_tuple] = prefix
            alphabet.update(lemma)
            alphabet.update(form)

    # COPY: copy an arbitrary sequence of lemma characters through unchanged.
    copy = "(" + "|".join(q(c) for c in sorted(alphabet)) + ")*"

    # One branch per feature combination: identity over the (bracketed) tags
    # followed by insertion of the prefix characters on the output tape.
    branches = []
    for feat_tuple, prefix in sorted(combo2prefix.items()):
        tags = " ".join(q("[" + f + "]") for f in feat_tuple)
        ins = " ".join("('':" + q(c) + ")" for c in prefix)
        branches.append("(" + tags + (" " + ins if ins else "") + ")")

    regex = "(" + " | ".join(branches) + ") " + copy

    fst = FST.re(regex)
    fst = fst.minimize()

    print("feature combinations:", len(combo2prefix))
    print("alphabet size:", len(alphabet))
    print("states after minimize:", len(fst.states))

    fomastring = fst.to_fomastring()
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(fomastring)
    print("saved to", OUT)

    return fst, combo2prefix


if __name__ == "__main__":
    main()
