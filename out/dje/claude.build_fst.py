#!/usr/bin/env python3
"""Build an FST for Zarma (dje) verb morphological inflection using PyFoma.

Training data (/workspace/data/dje.trn) is three columns: lemma, inflected
wordform, and a semicolon-delimited feature list. Inspecting the data shows a
fully regular pattern for the three tenses present:

    V;PRS  ->  wordform == lemma            (identity)
    V;PST  ->  wordform == "ga" + lemma      (prefix "ga")
    V;FUT  ->  wordform == "na" + lemma      (prefix "na")

Inputs are formatted with the morphological features first, each feature wrapped
in square brackets (e.g. [V][PRS]), followed by the lemma characters. PyFoma's
quoting (single quotes) is used so every feature tag and every character is an
atomic symbol. The output repeats the feature tags, then emits each character of
the inflected wordform.

The FST reads the [V] tag (identity), then the tense tag -- inserting the "ga"
or "na" prefix right after the tags for PST/FUT -- and finally copies the rest
of the input (the lemma) through unchanged. The trailing `.*` copies any symbol
identically, so unseen characters in the held-out data still pass through.
"""

from pyfoma import FST

# '[V]'                    : copy the POS tag through unchanged.
# '[PRS]'                  : present tense, no prefix.
# '[PST]' '':'g' '':'a'    : past tense, insert the "ga" prefix.
# '[FUT]' '':'n' '':'a'    : future tense, insert the "na" prefix.
# .*                       : identity-copy every remaining symbol (the lemma).
REGEX = "'[V]' ('[PRS]' | '[PST]' '':'g' '':'a' | '[FUT]' '':'n' '':'a') .*"

fst = FST.re(REGEX)

fomastring = fst.to_fomastring()
with open("test.foma", "w") as fh:
    fh.write(fomastring)

print("Wrote test.foma")
