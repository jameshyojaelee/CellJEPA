# Scripts

This directory holds runnable entry points (train/eval/report). Keep scripts thin:
- argument parsing + config loading
- calling library code in `src/celljepa/`

Avoid implementing core logic directly in scripts.

