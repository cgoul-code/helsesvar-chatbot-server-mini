if [ ! -d "antenv" ]; then
    python3 -m venv antenv
    source antenv/bin/activate
    pip install --no-cache-dir -r requirements.txt
else
    source antenv/bin/activate
fi

python -m hypercorn app:app --bind 0.0.0.0:${PORT:-8000}