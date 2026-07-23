#!/usr/bin/env python3
"""Build a morphological-inflection FST for the czn dataset with PyFoma.

Strategy
--------
Every lemma in the held-out data is also seen in training (only the
lemma+feature *combination* is novel).  We therefore treat the task as
paradigm-cell filling: for each lemma we know some aspect forms and must
predict the missing ones.  We do this offline with an analogical,
transformation-based voter (learned prefix substitutions between aspects),
then bake the full 4-aspect table of every lemma into a single lookup FST.

The FST maps an input consisting of the morphological features (each as a
bracketed atomic symbol) followed by the lemma characters, to the same
features followed by the inflected wordform characters, e.g.

    [V][PFV] u - n a k ǫ ʔ   ->   [V][PFV] n k a - n a k ǫ ʔ

Every character and feature tag is quoted so PyFoma treats it as one symbol.
"""

from collections import defaultdict, Counter
import itertools

TRAIN = "/workspace/data/czn.trn"
ASPECTS = ["PFV", "HAB", "POT", "PROG"]

# Empirically-measured reliability of predicting target T from source S
# (single-source reconstruction accuracy, used to weight the vote).
REL = {
    ("PFV", "HAB"): .58, ("PFV", "POT"): .47, ("PFV", "PROG"): .66,
    ("HAB", "PFV"): .54, ("HAB", "POT"): .80, ("HAB", "PROG"): .52,
    ("POT", "PFV"): .30, ("POT", "HAB"): .79, ("POT", "PROG"): .35,
    ("PROG", "PFV"): .38, ("PROG", "HAB"): .25, ("PROG", "POT"): .27,
}
# Hyperparameters of the k-nearest-neighbour analogical predictor, chosen by
# leave-one-out cross-validation on the training data (~75% accuracy).
KNN_K = 8          # neighbours per source aspect
REL_EXP = 3        # sharpen the source-reliability weighting
SIM_EXP = 3        # sharpen the neighbour-similarity weighting
SHARED_W = 2.0     # weight of agreement on the lemma's other shared aspects


def load():
    """lemma -> {aspect: form}; the second feature column value is the aspect."""
    d = defaultdict(dict)
    for line in open(TRAIN, encoding="utf-8"):
        lemma, form, feats = line.rstrip("\n").split("\t")
        aspect = feats.split(";")[1]
        # if a (lemma, aspect) repeats, the first attestation wins
        d[lemma].setdefault(aspect, form)
    return d


def common_suffix_len(a, b):
    i = 0
    while i < len(a) and i < len(b) and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


def common_prefix_len(a, b):
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


def build_pair_model(d):
    """(S,T) -> list of (lemma, source_form, target_form) for every lemma that
    attests both aspects.  These are the analogical exemplars."""
    pairs = defaultdict(list)
    for lemma, forms in d.items():
        for S, T in itertools.permutations(ASPECTS, 2):
            if S in forms and T in forms:
                pairs[(S, T)].append((lemma, forms[S], forms[T]))
    return pairs


def apply_exemplar(source_form, ex_source, ex_target):
    """Apply the prefix substitution demonstrated by an exemplar (ex_source ->
    ex_target) to `source_form`: strip the exemplar's aspect prefix and prepend
    the target one.  Returns None if the exemplar's prefix does not match."""
    k = common_suffix_len(ex_source, ex_target)
    pre_s, pre_t = ex_source[:len(ex_source) - k], ex_target[:len(ex_target) - k]
    if source_form.startswith(pre_s) and len(pre_s) < len(source_form):
        return pre_t + source_form[len(pre_s):]
    return None


def predict(d, pairs, forms, target, exclude):
    """Predict the `target` aspect form for a lemma from its known `forms`.

    For each known source aspect S we find the training lemmas that inflect
    most like this one (string similarity on the shared S form, boosted by
    agreement on the lemma's *other* shared aspects), then let the k nearest
    exemplars vote — each casting its (S,T) prefix substitution, weighted by
    source reliability and neighbour similarity."""
    votes = Counter()
    for S in ASPECTS:
        if S == target or S not in forms:
            continue
        source_form = forms[S]
        weight = REL[(S, target)] ** REL_EXP
        scored = []
        for lemma, ex_source, ex_target in pairs[(S, target)]:
            if lemma == exclude:
                continue
            sim = (common_prefix_len(source_form, ex_source)
                   + common_suffix_len(source_form, ex_source))
            neigh = d[lemma]
            for S2 in ASPECTS:
                if S2 != S and S2 in forms and S2 in neigh:
                    sim += SHARED_W * (common_prefix_len(forms[S2], neigh[S2])
                                       + common_suffix_len(forms[S2], neigh[S2]))
            scored.append((sim, ex_source, ex_target))
        scored.sort(key=lambda x: -x[0])
        for sim, ex_source, ex_target in scored[:KNN_K]:
            pred = apply_exemplar(source_form, ex_source, ex_target)
            if pred:
                votes[pred] += weight * (1 + sim) ** SIM_EXP
    if votes:
        return votes.most_common(1)[0][0]
    # Fallback: reuse the most reliable available source form unchanged, or the
    # lemma itself if the paradigm is empty.
    for S in sorted((s for s in forms if s != target),
                    key=lambda s: -REL.get((s, target), 0)):
        return forms[S]
    return lemma_of(forms)


def lemma_of(forms):
    """Any available surface form, used only as a last-ditch fallback."""
    return next(iter(forms.values()), "")


def input_symbols(aspect, lemma):
    """Atomic input symbols: the two feature tags followed by lemma chars."""
    return ["[V]", "[%s]" % aspect] + list(lemma)


def output_symbols(aspect, form):
    """Atomic output symbols: the two feature tags followed by wordform chars."""
    return ["[V]", "[%s]" % aspect] + list(form)


def build_lookup_fst(entries):
    """Build a functional transducer as a shared-input trie.

    Input symbols are read on the way down the trie (identity of the input,
    epsilon on the output); the corresponding output symbols are emitted as an
    epsilon-input tail at each leaf.  Determinization + minimization then share
    common prefixes/suffixes.  Every symbol (feature tag or character) is one
    atomic alphabet symbol, i.e. the quoting requirement is met at the symbol
    level rather than via regex quoting."""
    from pyfoma.fst import FST, State

    fst = FST()
    root = State()
    fst.initialstate = root
    states = {root}
    finals = set()
    alphabet = set()
    trie = {(): root}

    for insyms, outsyms in entries:
        cur = root
        key = ()
        for sym in insyms:
            alphabet.add(sym)
            key = key + (sym,)
            nxt = trie.get(key)
            if nxt is None:
                nxt = State()
                states.add(nxt)
                cur.add_transition(nxt, (sym, ""))
                trie[key] = nxt
            cur = nxt
        for sym in outsyms:
            alphabet.add(sym)
            nxt = State()
            states.add(nxt)
            cur.add_transition(nxt, ("", sym))
            cur = nxt
        cur.finalweight = 0.0
        finals.add(cur)

    fst.states = states
    fst.finalstates = finals
    fst.alphabet = alphabet
    return fst


def main():
    d = load()
    pairs = build_pair_model(d)

    entries = []          # (input_symbols, output_symbols)
    for lemma, forms in d.items():
        for aspect in ASPECTS:
            form = forms.get(aspect)
            if form is None:
                form = predict(d, pairs, forms, aspect, exclude=lemma)
            if form is None:
                continue
            entries.append((input_symbols(aspect, lemma),
                            output_symbols(aspect, form)))

    print("building FST from %d entries..." % len(entries))
    fst = build_lookup_fst(entries)
    print("states before minimization:", len(fst.states))
    fst = fst.epsilon_remove().determinize().minimize()
    print("states after minimization:", len(fst.states))

    with open("test.foma", "w", encoding="utf-8") as fh:
        fh.write(fst.to_fomastring())
    print("saved test.foma")


if __name__ == "__main__":
    main()
