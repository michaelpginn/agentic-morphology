#!/usr/bin/env python3
"""Build a morphological generation FST for Kongo (kon) verbs using PyFoma.

Input format:  feature tags first (each bracketed and quoted as an atomic
symbol) followed by the quoted lemma characters, e.g. '[V]''[PRS]''r''u''n'.
Output format: the same feature tags repeated, followed by the quoted
characters of the inflected wordform.

The training data (/workspace/data/kon.trn) shows a fully productive paradigm:

    V;PRS   ->  "ke " + lemma
    V;NFIN  ->  "ku"  + lemma        (irregular "ko" for two stems)
    V;FUT   ->  "ta ku" + lemma      (irregular "ta ko" for those two stems)
    V;PST   ->  lemma + "ka"

Only two lemmas ("sosila", "tutana") take the harmonized prefix vowel "ko"
instead of "ku"; these are handled as explicit exceptions via a targeted
rewrite of the prefix vowel before the exact stem. Everything else is one
broad productive rule per feature.
"""

from pyfoma import FST

# Alphabet observed in the training lemmas / wordforms (single-char symbols,
# including the space that appears in some multi-word forms and lemmas).
CHARS = [' ', 'a', 'b', 'd', 'e', 'f', 'g', 'i', 'k', 'l', 'm',
         'n', 'o', 'p', 's', 't', 'u', 'v', 'w', 'z']

# Stems that irregularly take the "ko" prefix vowel instead of "ku".
KO_STEMS = ['sosila', 'tutana']


def q(sym):
    """Quote a symbol so PyFoma treats it as one atomic symbol."""
    return "'" + sym + "'"


def insert(text):
    """Regex that inserts `text` on the output side (epsilon on input)."""
    return " ".join("'':" + q(c) for c in text)


# Identity copy of a non-empty lemma: each alphabet symbol maps to itself.
ALPHA = "(" + "|".join(q(c) for c in CHARS) + ")"
COPY = ALPHA + "+"

# Productive generator: emit "[V]", the inflection tag, then the affixed form.
GENERATOR = (
    "'[V]' ("
    f"  '[PRS]'  {insert('ke ')} {COPY}"
    f"| '[PST]'  {COPY} {insert('ka')}"
    f"| '[NFIN]' {insert('ku')} {COPY}"
    f"| '[FUT]'  {insert('ta ku')} {COPY}"
    ")"
)

fsts = {'generator': FST.re(GENERATOR)}

# Exception rules: after a prefix "k", harmonize "u" -> "o" for the two stems
# that idiosyncratically use the "ko" prefix (kususila -> kososila, etc.).
rules = ['$generator']
for i, stem in enumerate(KO_STEMS):
    name = f'ko_{i}'
    stem_re = " ".join(q(c) for c in stem)
    fsts[name] = FST.re(f"$^rewrite('u':'o' / 'k' _ {stem_re})")
    rules.append(f'${name}')

grammar = FST.re(" @ ".join(rules), fsts)

# Determinize + minimize for a compact automaton (secondary objective).
grammar = grammar.minimize()

fomastring = grammar.to_fomastring()
with open('test.foma', 'w') as fh:
    fh.write(fomastring)

print(f"Saved FST to test.foma ({len(grammar.states)} states).")
