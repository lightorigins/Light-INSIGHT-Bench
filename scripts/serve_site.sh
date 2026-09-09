#!/usr/bin/env bash
# Preview docs/ at the same URL shape GitHub Pages serves it from.
#
# A project page lives under /<repo>/, not at /. Serving docs/ directly at the
# root hides every absolute-path bug, so this mounts it under the repository
# name and prints the URL to open.
set -euo pipefail

REPO_NAME="${REPO_NAME:-light-insight-bench}"
PORT="${PORT:-8000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOUNT="$(mktemp -d)"

ln -sfn "$ROOT/docs" "$MOUNT/$REPO_NAME"
trap 'rm -rf "$MOUNT"' EXIT

echo "serving $ROOT/docs at http://127.0.0.1:$PORT/$REPO_NAME/"
cd "$MOUNT"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
