import argparse
import json
import logging
import pathlib
import re

import pyfoma

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Path to .foma file (FOMA format)")
parser.add_argument("--lang", help="Isocode for language")
args = parser.parse_args()
if not re.match(r"^[a-z]{3}$", args.lang):
    raise ValueError("Invalid language code!")

MAX_ITS = 10
metrics_path = pathlib.Path(__file__).parent / "metrics.json"
if metrics_path.exists():
    with open(metrics_path, "r") as p:
        metrics = json.load(p)
    it = max([s["it"] for s in metrics]) + 1
else:
    it = 0
    metrics = []
if it >= MAX_ITS:
    print("Max tests reached! Can no longer run additional tests.")
    raise ValueError()

with open(args.path, "r") as f:
    fomastring = f.read()
fst = pyfoma.FST.from_fomastring(fomastring)

it_metrics = {"it": it, "num_states": len(fst.states)}
for split in ["dev", "test"]:
    ext = "tst" if split == "test" else "dev"
    # in, out, feats
    data: list[tuple[str, str, str]] = []
    with open(pathlib.Path(__file__).parent / f"data/{args.lang}.{ext}") as f:
        for line in f:
            data.append(tuple(line.strip().split("\t")))  # type:ignore

    predictions = []
    for lemma, wordform, feats in data:
        input_string = [f"[{f.strip()}]" for f in feats.split(";")] + list(lemma)
        logger.debug(f"Composing input string: {''.join(input_string)}")
        input_fsa = pyfoma.FST.re("".join(f"'{c}'" for c in input_string))
        logger.debug("Composing input @ fst")
        output_fst = input_fsa @ fst
        logger.debug("Minimizing")
        output_fst = output_fst.minimize()
        if len(output_fst.finalstates) == 0:
            logger.debug(
                f"FST has no accepting states for input {''.join(input_string)}"
            )
            predicted = ""
        else:
            output_fst = output_fst.project(-1)
            predicted = "".join(c[0] for c in next(output_fst.words())[1])
        predictions.append(predicted)

    score = 0
    if split == "dev":
        # Dev is semi blind, test is full blind
        print("===============")
        print("Dev results by example:")
        print("lemma\tfeats\tprediction\tcorrect?")
    for (lemma, gold, feats), pred in zip(data, predictions):
        correct_output = [f"[{f.strip()}]" for f in feats.split(";")] + list(gold)
        correct_output = "".join(correct_output)
        if correct_output == pred:
            score += 1
            if split == "dev":
                print(f"{lemma}\t{feats}\t{pred}\tY")
        else:
            if split == "dev":
                print(f"{lemma}\t{feats}\t{pred}\tN")

    print("===============")
    print(f"Final score ({split}): {score}/{len(data)}")
    it_metrics[split] = score / len(data)
metrics.append(it_metrics)

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"Wrote to {metrics_path}")
