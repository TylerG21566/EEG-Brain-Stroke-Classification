
#!/usr/bin/env bash

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

# run full defualt pipeline
.venv/bin/python ./src/pipeline.py
