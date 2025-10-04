### Min Li - AI Agent
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from engine.product_loader import load_products
import os
import re


class ProductVectorStore:
    def __init__(self, text_model="sentence-transformers/all-MiniLM-L6-v2"):
        ### text-only embeddings, no image model here
        self.text_model = SentenceTransformer(text_model)

        self.products = load_products()
        self.text_index = None
        self.id_map = {}

        ### quick synonym map for common queries (manual, not fancy)
        self.synonym_map = {
            "sneakers": ["shoes", "running shoes", "trainers"],
            "tee": ["t-shirt", "shirt"],
            "cap": ["hat", "baseball cap"],
            "backpack": ["bag", "rucksack", "knapsack"]
        }

    def build_index(self):
        ### build embeddings from product info + image filename
        texts = []
        for p in self.products:
            img_name = os.path.basename(p["image_path"]).split(".")[0]
            texts.append(p["name"] + " " + p["description"] + " " + img_name)

        emb = self.text_model.encode(texts, convert_to_numpy=True)
        self.text_index = faiss.IndexFlatL2(emb.shape[1])
        self.text_index.add(emb)

        self.id_map = {i: p for i, p in enumerate(self.products)}
        print(f"index ready: {len(self.products)} items (dim={emb.shape[1]})")

    def expand_query_with_synonyms(self, query: str):
        ### expand user query with our simple synonym table
        words = query.lower().split()
        expanded = set(words)
        for w in words:
            if w in self.synonym_map:
                expanded.update(self.synonym_map[w])
        return " ".join(expanded)

    def extract_keywords(self, text: str):
        ### basic regex tokenizer, lowercase all words
        return re.findall(r"\w+", text.lower())

    def search(self, query_text=None, query_image=None, top_k=5):
        if not query_text and not query_image:
            raise ValueError("Need at least text or image query")

        ### treat image name as extra text keyword #%#
        if query_image:
            img_name = os.path.basename(query_image).split(".")[0]
            query_text = f"{query_text} {img_name}" if query_text else img_name

        expanded = self.expand_query_with_synonyms(query_text)
        q_emb = self.text_model.encode([expanded], convert_to_numpy=True)

        ### fetch more candidates, then filter down
        D, I = self.text_index.search(q_emb, top_k * 3)

        raw = [self.id_map[idx] for idx in I[0] if idx in self.id_map]

        ### keyword check
        kws = self.extract_keywords(query_text)
        filtered = [r for r in raw if any(kw in (r["name"] + " " + r["description"]).lower() for kw in kws)]
        final = filtered[:top_k] if filtered else raw[:top_k]

        print(f"[DEBUG] query='{query_text}', expanded='{expanded}', keywords={kws}, kept={len(final)}/{len(raw)}")
        return final


if __name__ == "__main__":
    store = ProductVectorStore()
    store.build_index()

    print("\nsearch by text:")
    res = store.search(query_text="cheap sneakers", top_k=3)
    for r in res:
        print(r["id"], r["name"], "-", r["description"], f"${r['price']} ⭐{r['rating']}")

    print("\nsearch by image:")
    res = store.search(query_image="data/images/backpack.jpg", top_k=3)
    for r in res:
        print(r["id"], r["name"], "-", r["description"], f"${r['price']} ⭐{r['rating']}")
### #%#