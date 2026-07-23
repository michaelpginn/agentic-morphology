#!/usr/bin/env python3
"""Build an FST for Chichewa (nya) verb inflection using PyFoma.

The training data (data/nya.trn) is a three-column format:
    lemma <TAB> inflected-form <TAB> semicolon-separated-features

Analysis of the data shows the inflection is fully regular: the inflected
form is always  PREFIX + lemma, where PREFIX is determined solely by the
feature bundle (subject person/number + tense, or the infinitive marker).
The lemma stem itself is never altered.

The FST maps a *quoted* symbol sequence of the form

    '[V]''[SG]''[2]''[PRS]''t''a''y''a'

(feature tags first, then the lemma characters, every tag/char an atomic
symbol) to the same feature tags followed by the characters of the
inflected wordform:

    '[V]''[SG]''[2]''[PRS]''m''u''m''a''t''a''y''a'

We build the transducer directly with the PyFoma State/Transition API so we
have exact control over every symbol (including the apostrophe that occurs
in stems such as ng'amba), then determinize + minimize for compactness.
"""

import os
from pyfoma import FST, State

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nya.trn")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.foma")


def load_mapping(path):
    """Return (feats_tuple -> prefix) mapping and the stem alphabet."""
    prefixes = {}
    stem_alphabet = set()
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            lemma, form, feats = line.split("\t")
            assert form.endswith(lemma), (lemma, form)
            prefix = form[: len(form) - len(lemma)]
            key = tuple(feats.split(";"))
            if key in prefixes:
                assert prefixes[key] == prefix, (key, prefixes[key], prefix)
            else:
                prefixes[key] = prefix
            stem_alphabet.update(lemma)
    return prefixes, stem_alphabet


def build():
    prefixes, stem_alphabet = load_mapping(DATA)

    f = FST()
    root = f.initialstate
    f.states = {root}
    f.finalstates = set()

    # Shared stem-copy states:
    #   pre  -- reached right after the prefix is emitted; not final (a stem
    #           must contain at least one character)
    #   stem -- final; loops on every stem character (identity)
    pre = State()
    stem = State()
    f.states.update({pre, stem})
    stem.finalweight = 0.0
    f.finalstates.add(stem)

    for ch in sorted(stem_alphabet):
        # first stem character: pre -> stem
        pre.add_transition(stem, (ch,), 0.0)
        # subsequent stem characters: stem -> stem
        stem.add_transition(stem, (ch,), 0.0)

    alphabet = set(stem_alphabet)

    # For every feature bundle, lay down a path that copies the feature tags
    # through, then inserts the prefix (epsilon:char), then joins the shared
    # stem-copy machine.
    for feats, prefix in prefixes.items():
        cur = root
        # copy feature tags through as identity: e.g. [V], [SG], [2], [PRS]
        for feat in feats:
            sym = "[" + feat + "]"
            alphabet.add(sym)
            nxt = State()
            f.states.add(nxt)
            cur.add_transition(nxt, (sym,), 0.0)
            cur = nxt
        # insert the prefix characters (input epsilon -> output char)
        for ch in prefix:
            alphabet.add(ch)
            nxt = State()
            f.states.add(nxt)
            cur.add_transition(nxt, ("", ch), 0.0)
            cur = nxt
        # link the end of the prefix into the shared stem-copy machine
        cur.add_transition(pre, ("",), 0.0)  # epsilon jump

    f.alphabet = alphabet

    # Clean up and compact: remove epsilons, determinize, minimize.
    f = f.epsilon_remove()
    f = f.determinize()
    f = f.minimize()
    return f


def main():
    f = build()
    print("states:", len(f.states))
    foma = f.to_fomastring()
    with open(OUT, "w") as fh:
        fh.write(foma)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
