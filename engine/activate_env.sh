poetry lock
poetry install
source $(poetry env info --path)/bin/activate
pre-commit install
echo "Engine Activated"
