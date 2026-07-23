#!/usr/bin/env python3
"""Build an FST for Nyanja (Chichewa) verb inflection.

The training data is fully regular: every inflected wordform equals a
feature-determined prefix concatenated with the lemma stem, i.e.

    wordform = prefix(features) + lemma

The FST therefore:
  1. reads the feature tags (e.g. [V][SG][2][PRS]) and echoes them back,
  2. inserts the prefix characters that those features select, then
  3. copies the lemma characters through unchanged.

Input  symbols:  [V][SG][2][PRS] t a y a
Output symbols:  [V][SG][2][PRS] m u m a t a y a
"""

from pyfoma import FST

TRAIN = "/workspace/data/nya.trn"
OUT = "test.foma"


def esc(ch):
    """Return a regex fragment that denotes the literal character `ch`."""
    if ch == "'":
        return r"\'"          # apostrophe is the quote char -> escape it
    if ch == "\\":
        return r"\\"
    return ch


def tag(feat):
    """Quote a feature as an atomic bracketed multichar symbol, e.g. '[SG]'."""
    return "'[" + feat + "]'"


def main():
    combos = {}          # feature-string -> prefix string
    letters = set()      # every character that can appear in a stem
    with open(TRAIN) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            lemma, form, feats = line.split("\t")
            assert form.endswith(lemma), (lemma, form)
            prefix = form[: len(form) - len(lemma)]
            if feats in combos:
                assert combos[feats] == prefix, (feats, prefix, combos[feats])
            else:
                combos[feats] = prefix
            letters.update(lemma)

    # Identity copy over any stem: (a|b|...|z|')*
    stem = "(" + "|".join(esc(c) for c in sorted(letters)) + ")*"

    # One union branch per feature combination.
    branches = []
    for feats, prefix in sorted(combos.items()):
        tags = " ".join(tag(t) for t in feats.split(";"))
        insert = " ".join("'':" + esc(c) for c in prefix)  # epsilon -> prefix chars
        branches.append("(" + tags + " " + insert + " " + stem + ")")

    regex = " | ".join(branches)
    grammar = FST.re(regex)

    # Compact the machine (determinize + minimize) before saving.
    for method in ("determinized", "minimized"):
        fn = getattr(grammar, method, None)
        if callable(fn):
            grammar = fn()
    print("states:", len(grammar.states))

    with open(OUT, "w") as fh:
        fh.write(grammar.to_fomastring())
    print("wrote", OUT, "| feature combos:", len(combos))


if __name__ == "__main__":
    main()
