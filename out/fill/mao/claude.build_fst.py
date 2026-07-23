#!/usr/bin/env python3
"""Build an FST for Māori active/passive verb inflection (PyFoma).

Data model (from /workspace/data/mao.trn):
  * V;ACT  -> the inflected form is always identical to the lemma (identity).
  * V;PASS -> the lemma takes a lexically-determined passive suffix
              (-a, -tia, -hia, -ria, ... ), occasionally with a stem vowel
              change.  This is not phonologically predictable, so we memorise
              the observed passive form per lemma.

I/O contract:
  input  = feature tags (bracketed, e.g. [V][PASS]) followed by the lemma chars
  output = the same feature tags followed by the inflected wordform chars
Every character and every feature tag is an atomic (quoted) symbol.

Generalisation target: unseen lemma+feature combinations (the lemma itself was
seen in training, but not with this feature).
  * ACT is handled by a single general identity path -> always correct.
  * For a lemma whose PASS form was never seen we fall back to a
    phonologically-conditioned default suffix (a-final -> -tia, else -> -a).

The two PASS paths (memorised vs. default) are given DISJOINT input domains so
the resulting transducer stays functional (one output per input).
"""

from pyfoma import FST


def q(sym: str) -> str:
    """Quote a symbol so PyFoma treats it as a single atomic symbol."""
    return "'" + sym + "'"


def seq(symbols) -> str:
    """A space-separated, individually-quoted symbol sequence for FST.re."""
    return " ".join(q(s) for s in symbols)


# ---------------------------------------------------------------------------
# 1. Read the training data.
# ---------------------------------------------------------------------------
DATA = "/workspace/data/mao.trn"
rows = []
chars = set()
with open(DATA) as fh:
    for line in fh:
        line = line.rstrip("\n")
        if not line:
            continue
        lemma, form, feat = line.split("\t")
        feats = feat.split(";")          # e.g. "V;PASS" -> ["V", "PASS"]
        rows.append((lemma, form, feats))
        chars |= set(lemma) | set(form)

# Include the full Māori orthographic inventory, not just the characters seen in
# training.  Otherwise a held-out lemma containing an unseen character (e.g. "ē")
# cannot even be tokenised, and the identity ACT path returns nothing.
chars |= set("aeiou") | set("āēīōū") | set("hkmngprtw")

# tag symbols like [V], [ACT], [PASS]
def tagsyms(feats):
    return ["[" + f + "]" for f in feats]

# ---------------------------------------------------------------------------
# 2. Alphabet: an identity FST over every orthographic character.
# ---------------------------------------------------------------------------
fsts = {}
fsts["Allid"] = FST.re("|".join(q(c) for c in sorted(chars)))   # copy any 1 char
fsts["AllStar"] = FST.re("$Allid*", fsts)                        # copy any string

# ---------------------------------------------------------------------------
# 3. ACT branch: [V][ACT] + identity copy of the lemma.
# ---------------------------------------------------------------------------
# Works for every lemma (seen or not) because ACT is always the identity.
act_branches = []
for feats in {tuple(f) for (_, _, f) in rows if f[-1] == "ACT"}:
    act_branches.append("(" + seq(tagsyms(list(feats))) + " $AllStar)")
fsts["ACT"] = FST.re(" | ".join(act_branches), fsts)

# ---------------------------------------------------------------------------
# 4. PASS memorised branch: exact input:output pairs for every seen passive.
# ---------------------------------------------------------------------------
pass_pairs = []          # (lemma, form, feats)
pass_lemmas = set()      # lemmas whose passive we have observed
for lemma, form, feats in rows:
    if feats[-1] == "PASS":
        pass_pairs.append((lemma, form, feats))
        pass_lemmas.add(lemma)

known_regex = []
for lemma, form, feats in pass_pairs:
    tags = tagsyms(feats)
    inp = seq(tags + list(lemma))
    out = seq(tags + list(form))
    known_regex.append("((" + inp + "):(" + out + "))")
fsts["PASSknown"] = FST.re(" | ".join(known_regex), fsts)

# ---------------------------------------------------------------------------
# 5. PASS default branch: for lemmas whose passive was NOT seen.
# ---------------------------------------------------------------------------
# Identity over char strings that are NOT a memorised passive lemma.
if pass_lemmas:
    fsts["KnownLemmas"] = FST.re(
        " | ".join("(" + seq(list(l)) + ")" for l in sorted(pass_lemmas)), fsts
    )
    fsts["UnknownStem"] = FST.re("$AllStar - $KnownLemmas", fsts)
else:
    fsts["UnknownStem"] = FST.re("$AllStar", fsts)

# Append a default passive suffix conditioned on the lemma's final vowel:
#   a / ā  -> -tia      (bare -a is essentially never used after /a/)
#   else   -> -a        (most frequent elsewhere)
a_final = "('a'|'ā')"
other_final = "|".join(q(c) for c in sorted(chars) if c not in ("a", "ā"))
fsts["Append"] = FST.re(
    "($Allid* {a} ('':('t' 'i' 'a'))) | ($Allid* ({o}) ('':('a')))".format(
        a=a_final, o=other_final
    ),
    fsts,
)

# Restrict the append map to unknown stems, then add the tag prefix.
fsts["DefaultStem"] = FST.re("$UnknownStem @ $Append", fsts)
fsts["PASSdefault"] = FST.re("('[V]' '[PASS]') $DefaultStem", fsts)

# ---------------------------------------------------------------------------
# 6. Union everything, make it functional and compact.
# ---------------------------------------------------------------------------
grammar = FST.re("$ACT | $PASSknown | $PASSdefault", fsts)
grammar = grammar.determinize().minimize()

# ---------------------------------------------------------------------------
# 7. Quick self-test on the training data.
# ---------------------------------------------------------------------------
def encode_input(lemma, feats):
    return "".join(tagsyms(feats)) + lemma

def encode_output(form, feats):
    return "".join(tagsyms(feats)) + form

if __name__ == "__main__":
    correct = 0
    for lemma, form, feats in rows:
        got = list(grammar.generate(encode_input(lemma, feats)))
        exp = encode_output(form, feats)
        if got == [exp]:
            correct += 1
        else:
            print("MISS", feats, lemma, "->", got, "expected", exp)
    print(f"train exact-match: {correct}/{len(rows)}")
    print("states:", len(grammar.states))

    with open("test.foma", "w") as fh:
        fh.write(grammar.to_fomastring())
    print("saved test.foma")
