#!/usr/bin/env python3
"""Build a morphological-inflection FST for Zarma (dje) with PyFoma.

Training data (data/dje.trn) is a three-column TSV: lemma, inflected form,
and a semicolon-delimited feature list.  Inspecting the data shows the
inflection is fully regular and depends only on the tense feature:

    V;PRS  ->  form == lemma            (no change)
    V;FUT  ->  form == "na" + lemma     (prefix "na")
    V;PST  ->  form == "ga" + lemma     (prefix "ga")

The FST reads a quoted, feature-first input such as '[V]''[FUT]''k''a''a''h',
echoes the feature tags, and reproduces the characters of the inflected form.
Every character and feature is written as a PyFoma-quoted atomic symbol.
"""

from pyfoma import FST

# --- Alphabet ---------------------------------------------------------------
# All characters that occur in the lemmas / word forms of the training data.
# Test lemmas are drawn from the same set of lemmas (only the feature
# combinations are unseen), so this alphabet is complete.
CHARS = ["'", "-", "A", "a", "b", "c", "d", "e", "f", "g", "h", "i",
         "k", "m", "n", "o", "r", "s", "t", "u", "w", "y"]


def quote(sym: str) -> str:
    """Render a symbol as a PyFoma-quoted atomic symbol.

    The single-quote character must itself be escaped inside the quotes."""
    if sym == "'":
        return r"'\''"
    return "'%s'" % sym


# Identity copy over any single character symbol, e.g. ('k'|'a'|...).
CHAR_UNION = "(" + "|".join(quote(c) for c in CHARS) + ")"

# Each branch echoes the feature tags, then (for FUT/PST) inserts the prefix
# via empty-string-to-symbol mappings, then copies the word characters.
#   [V][PRS] : identity
#   [V][FUT] : insert 'n' 'a'
#   [V][PST] : insert 'g' 'a'
REGEX = (
    "'[V]'"
    "("
    "'[PRS]'"
    "|'[FUT]'('':'n')('':'a')"
    "|'[PST]'('':'g')('':'a')"
    ")"
    "{C}*".format(C=CHAR_UNION)
)


def build() -> FST:
    fst = FST.re(REGEX)
    fst.minimize()
    return fst


def main() -> None:
    fst = build()

    # Sanity-check against the full training set.
    total = correct = 0
    for line in open("data/dje.trn"):
        line = line.rstrip("\n")
        if not line:
            continue
        lemma, form, feats = line.split("\t")
        tag = feats.split(";")[1]
        inp = "[V][%s]%s" % (tag, lemma)
        expected = "[V][%s]%s" % (tag, form)
        outputs = list(fst.generate(inp))
        total += 1
        correct += outputs == [expected]
    print("training accuracy: %d/%d (%d states)"
          % (correct, total, len(fst.states)))

    # Save as a foma string.
    fomastring = fst.to_fomastring()
    with open("test.foma", "w") as fh:
        fh.write(fomastring)
    print("wrote test.foma")


if __name__ == "__main__":
    main()
