# ABOMINATION

## Setup

```bash
cd ABOMINATION
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
scripts/fix_libomp.sh  # macOS: torch + xgboost share Homebrew's libomp (else they crash together)
```

## Adding packages

```bash
cd ABOMINATION
source .venv/bin/activate
pip install <package>
pip freeze > requirements.txt
```
