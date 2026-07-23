#!/usr/bin/env python3
"""Build an FST for Zarma (dje) verb inflection.

Training data analysis (data/dje.trn) shows a fully regular, three-way system:
    V;PRS  -> identity            (wordform == lemma)
    V;FUT  -> prefix "na"         (wordform == "na" + lemma)
    V;PST  -> prefix "ga"         (wordform == "ga" + lemma)

Input side (features first, each symbol atomic/quoted):
    '[V]''[PRS]''r''u''n'
Output side (features echoed, then the inflected wordform char-by-char):
    '[V]''[PRS]''r''u''n'          (PRS)
    '[V]''[FUT]''n''a''r''u''n'    (FUT)
    '[V]''[PST]''g''a''r''u''n'    (PST)

The FST echoes the two feature tags, inserts the tense prefix on the output
side for FUT/PST, then copies the lemma characters unchanged. Because the
inflection is a single productive rule per tense, this compiles to a very small
transducer.
"""

from pyfoma import FST

# --- Character inventory (all chars seen across lemmas in training) -----------
# Apostrophe is written \' and hyphen is quoted so it is not read as an operator.
CHARS = ["A", "a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "m",
         "n", "o", "r", "s", "t", "u", "w", "y", "-", "'"]


def _q(c):
    """Quote a single character as an atomic symbol for FST.re."""
    if c == "'":
        return r"\'"          # literal apostrophe symbol
    return "'" + c + "'"


char_union = " | ".join(_q(c) for c in CHARS)

defined = {"Char": FST.re(char_union)}

# Each branch: echo [V] and the tense tag, insert the prefix on the output side
# (if any), then copy the lemma characters ($Char*, an identity transducer).
grammar_re = (
    "  '[V]' '[PRS]' $Char* "
    "| '[V]' '[FUT]' '':'n' '':'a' $Char* "
    "| '[V]' '[PST]' '':'g' '':'a' $Char* "
)

grammar = FST.re(grammar_re, defined)

# Optimize for the most compact state count.
grammar = grammar.epsilon_remove().determinize().minimize()

print("States:", len(grammar.states))

# --- Save --------------------------------------------------------------------
fomastring = grammar.to_fomastring()
with open("test.foma", "w") as f:
    f.write(fomastring)
print("Wrote test.foma")
