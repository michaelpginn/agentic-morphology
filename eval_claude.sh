#!/usr/bin/env bash
set -euo pipefail

export DATASET="$1"
export LANG="$2"
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY in your environment}"

prompt="Write a script called build_fst.py using the PyFoma library, which is already installed in the system python, that creates and saves an FST for the provided data. The primary goal is to produce an FST that is accurate, with a secondary goal of making the FST as compact as possible (minimal number of states).

Training data is morphological inflection data /workspace/data/$LANG.trn, consisting of a three column format where the first column is the input string (a lemma), the second column is the output string (the inflected wordform), and the third column is a semicolon-deliminated list of morphological features. Your FST should take inputs formatted with the features first, where each feature adds square brackets, such as [PL][PRS]. Furthermore, you should use PyFoma's quoting feature to ensure each character or feature is an atomic symbol. A full input might look like: '[PL]''[PRS]''r''u''n'. The expected output repeats back the feature tags in the same way, followed by each character of the inflected wordform, again quoted. Your FST only needs to generalize to unseen lemma+feature combinations, not completely unseen lemmas.

Your script may use any strategy to build the FST. There are many possible ways, such as using the ^rewrite function to implement rewrite rules, using the lexd compiler, or simply coding FST regexes directly. At the end of the script, you should save the FST by first calling to_fomastring() and then saving the fomastring to a file, which should be named test.foma.

You can test your FST against the training data as much as you'd like. To test against the held-out dev and test sets, run the following:
    > sudo /opt/grader/grade test.foma --lang $LANG
This will first print out the dev set results, which will include an overall exact-match accuracy score and a full list of the inputs, predictions, and whether each prediction was correct. Then, the grade script will print only the accuracy score for the test set. You are not allowed to look at either full file. Furthermore, you are limited to 10 total runs of the grading script, after which it will raise an error. Thus, you should use this sparingly when the FST appears to be correct.
"

# Name that identifies which agent produced these artifacts. Future agents get
# their own eval_<agent>.sh with a different value here (e.g. "codex", "gemini").
AGENT="claude"

# Pin the exact model ID for reproducibility (aliases like "opus"/"sonnet" drift
# to newer models over time). Options: claude-opus-4-8, claude-sonnet-4-6,
# claude-haiku-4-5-20251001, claude-fable-5.
MODEL="claude-opus-4-8"

OUT="$(pwd)/out/${DATASET}/${LANG}"
mkdir -p "$OUT"
RUN_NAME="fst_run_${AGENT}_${DATASET}_${LANG}"

# Remove the named run container + proxy + networks when we finish (or error).
cleanup() {
  docker rm -f "$RUN_NAME" >/dev/null 2>&1 || true
  docker compose down --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT
docker rm -f "$RUN_NAME" >/dev/null 2>&1 || true  # clear any leftover from a prior run

# Render Claude Code's stream-json events as readable lines so you can watch the
# agent think, call tools, and run the grader live. Falls back to raw JSON if
# jq isn't installed on the host.
render() {
  if command -v jq >/dev/null 2>&1; then
    jq -rj '
      if .type=="assistant" then
        (.message.content[]? |
          if .type=="text" then .text + "\n"
          elif .type=="tool_use" then "\n[36m🔧 " + .name + "[0m " + (.input | tostring) + "\n"
          else empty end)
      elif .type=="user" then
        (.message.content[]? |
          if .type=="tool_result" then
            "[90m↳ " + ((.content // "") | if type=="array" then (map(.text? // "") | join("")) else tostring end) + "[0m\n"
          else empty end)
      elif .type=="result" then "\n[32m✅ " + (.result // "") + "[0m\n"
      else empty end'
  else
    cat
  fi
}

# Produce a clean, plain-text transcript (no colors) from the raw stream-json for
# a .log file. Assistant text is kept verbatim; each tool call is condensed to a
# single "→ Tool: <key arg>" line; tool results are indented beneath their call.
format_log() {
  jq -r '
    def arg:
      (.input // {}) as $i
      | ($i.command // $i.file_path // $i.path // $i.pattern // $i.query // $i.description // ($i | tostring))
      | tostring | gsub("\n"; " ")
      | if length > 140 then .[0:139] + "…" else . end;
    if .type=="assistant" then
      ( .message.content[]?
        | if .type=="text" then (.text | rtrimstr("\n"))
          elif .type=="tool_use" then "→ " + .name + ": " + arg
          else empty end )
    elif .type=="user" then
      ( .message.content[]?
        | if .type=="tool_result" then
            ( (.content // "") | if type=="array" then (map(.text? // "") | join("")) else tostring end )
            | rtrimstr("\n")
            | if . == "" then empty else ( split("\n") | map("    " + .) | join("\n") ) end
          else empty end )
    elif .type=="result" then "\n=== FINAL RESULT ===\n" + ((.result // "") | rtrimstr("\n"))
    else empty end
  '
}

# `run` starts the proxy dependency (waiting for its healthcheck) first, then
# runs the agent on the internal-only network. The command here overrides the
# service default, so the prompt lives in this script. -T disables the pseudo-TTY
# so the stream can be piped. No --rm + a fixed name so we can copy artifacts out
# before cleanup. `|| true` keeps us going to the export step even if the agent
# errors. `tee` saves the complete, lossless stream-json log (one event per line)
# while render pretty-prints it live to the terminal.
docker compose run --name "$RUN_NAME" -T agent \
  claude -p "$prompt" --dangerously-skip-permissions --model "$MODEL" \
    --verbose --output-format stream-json \
  | tee "$OUT/${AGENT}.log.jsonl" | render || true

# Export artifacts to ./out/$DATASET/$LANG/, prefixed by agent name. docker cp reads via
# the daemon (root), so it pulls metrics.json out of the 700 root-only /opt/grader
# without the agent ever being able to read or reset it at runtime.
echo
echo "Exporting artifacts to $OUT ..."
echo "  ${AGENT}.log.jsonl"
# Human-readable transcript derived from the raw log (needs jq).
if command -v jq >/dev/null 2>&1 && [ -s "$OUT/${AGENT}.log.jsonl" ]; then
  format_log < "$OUT/${AGENT}.log.jsonl" > "$OUT/${AGENT}.log" && echo "  ${AGENT}.log"
fi
docker cp "$RUN_NAME:/workspace/build_fst.py" "$OUT/${AGENT}.build_fst.py" 2>/dev/null \
  && echo "  ${AGENT}.build_fst.py"  || echo "  (no ${AGENT}.build_fst.py — agent may not have saved one)"
docker cp "$RUN_NAME:/workspace/test.foma" "$OUT/${AGENT}.foma" 2>/dev/null \
  && echo "  ${AGENT}.foma"  || echo "  (no ${AGENT}.foma — agent may not have saved one)"
docker cp "$RUN_NAME:/opt/grader/metrics.json" "$OUT/${AGENT}.metrics.json" 2>/dev/null \
  && echo "  ${AGENT}.metrics.json" || echo "  (no ${AGENT}.metrics.json — agent may not have run the grader)"
