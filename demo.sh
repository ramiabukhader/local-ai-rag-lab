#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"
VENV_DIR="${VENV_DIR:-.venv-demo}"
PYTHON="${PYTHON:-python3}"

run() {
  printf '\n$'
  printf ' %s' "$@"
  printf '\n'
  "$@"
}

run "$PYTHON" -m venv "$VENV_DIR"
run "$VENV_DIR/bin/python" -m pip install -e .
run "$VENV_DIR/bin/rag-ingest"
run "$VENV_DIR/bin/rag-evaluate"
