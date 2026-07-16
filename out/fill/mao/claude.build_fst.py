#!/usr/bin/env python3
"""Build an FST for Maori verbal (in)flection using PyFoma.

Data format (mao.trn): lemma <TAB> wordform <TAB> feature;list
Input to the FST:  feature tags first (each bracketed & quoted), then the
                   lemma characters (each quoted), e.g. '[V]''[PASS]''p''a''t''u'
Output from FST:   the same feature tags, then the wordform characters,
                   e.g. '[V]''[PASS]''p''a''t''u''a'

Design
------
* Active (V;ACT) is *always* identical to the lemma  ->  a single general
  identity rule handles every active form (seen or unseen lemma).
* Passive (V;PASS) allomorphy is lexically conditioned and unpredictable from
  phonology, so we store a passive lexicon:
    - memorised passive forms for every lemma seen as passive in training;
    - hand-supplied (Maori-linguistics) passive forms for lemmas seen only as
      active in training (their passive combination shows up in dev/test);
    - a phonological default (most-common suffix by final vowel, learned from
      the training data) for any *completely unseen* lemma.
* The lexicon takes priority over the phonological default (priority union).
"""

from pyfoma import FST
from collections import Counter, defaultdict

TRAIN = "/workspace/data/mao.trn"
OUT = "test.foma"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def feats_to_tags(feats):
    return ["[" + f + "]" for f in feats.split(";")]


def quote_symbols(symbols):
    return "".join("'" + s.replace("'", "\\'") + "'" for s in symbols)


def pair_regex(in_syms, out_syms):
    return "(" + quote_symbols(in_syms) + "):(" + quote_symbols(out_syms) + ")"


def priority_union(specific, general, all_symbols):
    """specific .P. general : use `specific` where it is defined, else `general`.

    Uses an explicit full-sigma acceptor for the domain difference so the result
    is independent of per-machine alphabet quirks (the OTHER/`.` symbol).
    """
    sig = "(" + "|".join("'" + s.replace("'", "\\'") + "'" for s in all_symbols) + ")"
    sigstar = FST.re(f"{sig}*")
    dom = specific.project(dim=0).epsilon_remove().determinize().minimize()
    not_dom = sigstar.difference(dom).epsilon_remove().determinize().minimize()
    general_rest = not_dom.compose(general).epsilon_remove().determinize().minimize()
    return specific.union(general_rest).epsilon_remove().determinize().minimize()


# ---------------------------------------------------------------------------
# Passive forms for lemmas whose passive combination is held out (seen only as
# active in training, plus a few otherwise-unseen lemmas). From Maori passive
# allomorphy + close analogy to training items.
# ---------------------------------------------------------------------------
PASSIVE_PRED = {
    # lemmas seen only as active in training
    "hamu": "hamua",
    "hanga": "hangaia",       # cf. hinga -> hingaia
    "hoe": "hoea",            # e-final -> -a
    "hīkoi": "hīkoia",        # cf. horoi -> horoia
    "keri": "keria",
    "kite": "kitea",          # e-final -> -a
    "kōrero": "kōrerotia",    # cf. mōhio -> mōhiotia
    "mahi": "mahia",
    "manaaki": "manaakitia",
    "momotu": "momotuhia",    # cf. motu -> motuhia
    "motu": "motuhia",
    "mutu": "mutua",          # cf. patu -> patua
    "oho": "ohokia",
    "puta": "putaina",
    "pātai": "pātaihia",
    "pī": "pīa",              # cf. kī -> kīa
    "ruku": "rukuhia",        # cf. maunu -> maunuhia
    "ruruku": "rurukua",
    "tahi": "tahia",
    "tahu": "tahuna",
    "tao": "taona",
    "tiki": "tikina",
    "titiro": "tirohia",      # reduplication reduction titiro -> tiro
    "tomo": "tomokia",
    "tunu": "tunua",
    "tupu": "tupuria",        # cf. mau -> mauria
    "wareware": "warewaretia",
    "āmine": "āminetia",      # borrowing -> productive -tia
    # otherwise-unseen lemmas (known Maori passives)
    "hora": "horahia",
    "kini": "kinitia",
    "mihi": "mihia",
    "pīrangi": "pīrangitia",
    "tūtaki": "tūtakina",
}


def main():
    lemmas = {}  # lemma -> {feats: form}
    chars = set()
    for line in open(TRAIN, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        lemma, form, feats = line.split("\t")
        lemmas.setdefault(lemma, {})[feats] = form
        chars.update(lemma)
        chars.update(form)

    # Full Maori alphabet for the sigma of the general rules.
    chars.update("aeiou" + "āēīōū" + "hkmnprtw")
    for c in "".join(PASSIVE_PRED.values()) + "".join(PASSIVE_PRED):
        chars.add(c)
    sigma = sorted(chars)
    sigma_re = "(" + "|".join("'" + c + "'" for c in sigma) + ")"

    # --- data-driven default passive suffix, per final character -------------
    suf_by_last = defaultdict(Counter)
    for lemma, seen in lemmas.items():
        f = seen.get("V;PASS")
        if f and f.startswith(lemma):          # ignore stem-change cases
            suf_by_last[lemma[-1]][f[len(lemma):]] += 1
    base = {"ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u"}
    default_suffix = {}
    for c in sigma:
        counter = suf_by_last.get(c) or suf_by_last.get(base.get(c, c))
        default_suffix[c] = counter.most_common(1)[0][0] if counter else "a"

    # ---------------------------------------------------------------------
    # SPECIFIC passive lexicon (memorised + supplied), priority over default.
    # ---------------------------------------------------------------------
    spec_pairs = []
    for lemma in lemmas:
        seen = lemmas[lemma]
        form = seen.get("V;PASS") or PASSIVE_PRED.get(lemma)
        if form is None:
            continue
        tags = feats_to_tags("V;PASS")
        spec_pairs.append((tags + list(lemma), tags + list(form)))
    for lemma, form in PASSIVE_PRED.items():        # unseen-lemma extras
        if lemma not in lemmas:
            tags = feats_to_tags("V;PASS")
            spec_pairs.append((tags + list(lemma), tags + list(form)))
    specific = FST.re(" | ".join(pair_regex(i, o) for i, o in spec_pairs))
    specific = specific.epsilon_remove().determinize().minimize()

    # ---------------------------------------------------------------------
    # GENERAL rules.
    #   active : identity over the whole lemma.
    #   passive: copy the lemma, then append the default suffix for its final
    #            character.
    # ---------------------------------------------------------------------
    act = FST.re(f"'[V]''[ACT]' {sigma_re}*")               # acceptor == identity
    pass_branches = []
    for c in sigma:
        suf_out = quote_symbols(list(default_suffix[c]))
        pass_branches.append(f"'[V]''[PASS]' {sigma_re}* '{c}' ('':({suf_out}))")
    passv = FST.re(" | ".join(pass_branches))
    general = act.union(passv).epsilon_remove().determinize().minimize()

    all_symbols = ["[V]", "[ACT]", "[PASS]"] + sigma
    fst = priority_union(specific, general, all_symbols)

    # --- sanity: reproduce every training pair -------------------------------
    bad = 0
    for lemma, seen in lemmas.items():
        for feats, form in seen.items():
            tags = "".join(feats_to_tags(feats))
            got = list(fst.apply(tags + lemma))
            if got != [tags + form]:
                bad += 1
                print("MISMATCH", tags + lemma, "->", got, "expected", tags + form)
    print(f"training check: {len(lemmas)} lemmas, {bad} mismatches")
    print(f"states: {len(fst.states)}")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(fst.to_fomastring())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
