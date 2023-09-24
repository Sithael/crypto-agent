ENV_DIR=$(realpath $(dirname "${BASH_SOURCE[0]}"))
ENV_ROOT=$SCRAPER_DIR
export ENV_ROOT

poetry lock
poetry install
source $(poetry env info --path)/bin/activate
pre-commit install
echo "Env Activated"
