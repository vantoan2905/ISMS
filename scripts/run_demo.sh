#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --group pose --group training python -m src.isms.main "$@"
