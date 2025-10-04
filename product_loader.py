### Min Li - AI Agent

import json
from pathlib import Path

### path to products.json
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "products.json"

def load_products():
    ### load product info from JSON
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        products = json.load(f)
    return products


if __name__ == "__main__":
    items = load_products()
    print(f"Loaded {len(items)} products")
    print(items[0])
### #%#
