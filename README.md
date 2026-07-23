# Agentic Induction of Morphological Transducers

## Getting started

Build the docker image and make the eval script executable (only once):

```bash
docker build .
DATASET=x LANG=x ANTHROPIC_API_KEY=x docker compose build agent
chmod +x eval_claude.sh
```

Run an agentic eval:

```bash
./eval_claude <lang> # Use any ISO-code from data/
```
