#!/usr/bin/env bash
# Usage: ./full_pipeline_default.sh [-s|--start N] [-e|--end N]
#   Runs pipeline stages N..N (inclusive). Defaults: start 0, end = last stage. Range checks happen in src/pipeline.py.

usage() { echo "Usage: $0 [-s|--start N] [-e|--end N]"; }

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--start|-e|--end)
      if [[ ! "$2" =~ ^[0-9]+$ ]]; then echo "[ERROR] $1 needs a stage number, got '$2'"; usage; exit 1; fi
      ARGS+=("$1" "$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown option '$1'"; usage; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "$(pwd -P)" != "$SCRIPT_DIR" ]]; then
  printf '\033[0;31m'
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo ""
  echo "[ERROR] Not running: please run this script from $SCRIPT_DIR."
  echo ""
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  printf '\033[0m'

  exit 0
fi
printf '\033[0;32m'
echo "#################################################"
echo ""
echo "[SUCCESS] Running from the correct directory: $SCRIPT_DIR."
echo ""
echo "#################################################"
printf '\033[0m'

# install all required packages
cd ..
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run the pipeline over the chosen stage range
.venv/bin/python ./src/pipeline.py "${ARGS[@]}"
