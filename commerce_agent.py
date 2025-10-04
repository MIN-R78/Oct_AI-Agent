### Min Li - AI Agent
import os
import re
from engine.vector_store import ProductVectorStore

### we use SentenceTransformer for small talk matching #%#
from sentence_transformers import SentenceTransformer, util

### project base + images folder #%#
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")


class CommerceAgent:
    def __init__(self):
        ### init product index (FAISS + embeddings) #%#
        self.vector_store = ProductVectorStore()
        self.vector_store.build_index()

        ### light model for chat-like answers #%#
        self.chat_model = SentenceTransformer("all-MiniLM-L6-v2")

        ### some canned replies for casual chit-chat #%#
        self.small_talk = {
            "what's your favorite color": "I like blue, it feels calm and smart.",
            "did you eat": "Haha, I don’t eat, but I’m always ready to help you shop.",
            "do you like sports": "Yes, I like running—fast like a search engine",
            "are you human": "Nope, I’m an AI assistant, but I try my best to sound friendly!",
            "tell me a joke": "Why don’t programmers like nature? Too many bugs.",
            "how's your day": "I’m doing great, thanks for asking! How’s your day going?",
            "i'm good": "Glad to hear that! 😊",
            "i'm sad": "I’m sorry to hear that. Sometimes shopping for something nice can help cheer you up.",
        }

        ### pre-calc embeddings for above small talk keys (faster lookup) #%#
        self.st_embeddings = self.chat_model.encode(
            list(self.small_talk.keys()), convert_to_tensor=True
        )

    def resolve_image_path(self, image_name: str):
        ### sanity check: does this image file exist in /data/images ? #%#
        candidate = os.path.join(IMAGES_DIR, image_name)
        if os.path.exists(candidate):
            return candidate
        return None

    def extract_keywords(self, text: str):
        ### quick+dirty keyword extractor (regex tokenize, remove stopwords)
        tokens = re.findall(r"\w+", text.lower())
        stopwords = {"the", "is", "a", "an", "and", "or", "in", "on", "for", "with", "to", "of", "at"}
        return [t for t in tokens if t not in stopwords]

    def keyword_filter(self, query: str, results: list, top_k: int = 3, strict=True):
        ### strict keyword filter (force all keywords appear in product text) #%#
        if not results:
            return []

        if not strict:
            return results[:top_k]

        keywords = self.extract_keywords(query)
        if not keywords:
            return results[:top_k]

        filtered = [
            r for r in results
            if all(kw in (r.get("name", "") + " " + r.get("description", "")).lower() for kw in keywords)
        ]

        print(f"[DEBUG] Keywords: {keywords}, Filtered: {len(filtered)} / {len(results)}")

        if filtered:
            return filtered[:top_k]
        return []

    def safe_search(self, **kwargs):
        ### run vector_store.search but catch crashes (FAISS can throw errors)
        try:
            results = self.vector_store.search(**kwargs) or []
            return results
        except Exception as e:
            print(f"[DEBUG] Search error: {e}")
            return []

    def handle_query(self, query: str):
        ### text query only (with strict filter) #%#
        results = self.safe_search(query_text=query, top_k=15)
        results = self.keyword_filter(query, results, top_k=3, strict=True)
        return self.format_results(results)

    def handle_image_query(self, image_name: str):
        ### image search (skip strict filter, filenames like .jpg are useless)
        path = self.resolve_image_path(image_name)
        if not path:
            available = os.listdir(IMAGES_DIR)
            return f"Image not found: {image_name}\nAvailable images: {available}"

        results = self.safe_search(query_image=path, top_k=15)
        results = self.keyword_filter(image_name, results, top_k=3, strict=False)
        return self.format_results(results)

    def handle_mixed_query(self, text: str, image_name: str):
        ### hybrid search: combine text + image, filter only on text part #%#
        path = self.resolve_image_path(image_name)
        if not path:
            available = os.listdir(IMAGES_DIR)
            return f"Image not found: {image_name}\nAvailable images: {available}"

        results = self.safe_search(query_text=text, query_image=path, top_k=15)
        results = self.keyword_filter(text, results, top_k=3, strict=True)
        return self.format_results(results)

    def handle_general_conversation(self, query: str):
        ### rule-based responses first (fast path) #%#
        q = query.lower()

        if any(greet in q for greet in ["hello", "hi", "hey"]):
            return "Hello! How can I help you today?"

        elif "your name" in q or "who are you" in q:
            return "I’m Palona AI Agent, here to help you with products."

        elif "what can you do" in q:
            return "I can chat and also help you search products. Try asking about shoes, jackets, or images."

        elif "weather" in q:
            return "I can’t fetch real-time weather, but let’s just say it’s a nice day."

        elif "time" in q:
            from datetime import datetime
            return f"It’s {datetime.now().strftime('%H:%M')}."

        elif "date" in q or "today" in q or "what day is it" in q:
            from datetime import datetime
            return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."

        ### no rule matched → semantic similarity on small talk #%#
        query_emb = self.chat_model.encode(q, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_emb, self.st_embeddings)[0]
        best_match_id = int(cos_scores.argmax())
        best_score = float(cos_scores[best_match_id])

        if best_score > 0.6:  # threshold tweakable
            return list(self.small_talk.values())[best_match_id]

        ### fallback: just try product search #%
        results = self.safe_search(query_text=query, top_k=15)
        results = self.keyword_filter(query, results, top_k=3, strict=True)
        if results:
            return self.format_results(results)

        return "No products found. Please try with another keyword."

    def format_results(self, results):
        ### format into chat-friendly lines (name, desc, price, rating) #%#
        if not results:
            return "No products found."
        output = "\nHere are some products I found for you:\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r.get('name','')} - {r.get('description','')} (${r.get('price','?')}) ⭐{r.get('rating','?')}\n"
        return output


if __name__ == "__main__":
    agent = CommerceAgent()
    print("Hello! I am Palona AI Agent.")
    print("You can:")
    print(" - Chat with me (e.g., 'What's your name?')")
    print(" - Ask casual questions (e.g., 'Do you like sports?')")
    print(" - Search by text (e.g., 'cheap sneakers')")
    print(" - Search by image (just type the image file name, e.g., 'backpack.jpg')")
    print(" - Mixed search: text | image_file_name")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter query: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if "|" in user_input:
            parts = [p.strip() for p in user_input.split("|")]
            if len(parts) == 2:
                text_query, image_name = parts
                print(agent.handle_mixed_query(text_query, image_name))
            else:
                print("Invalid format. Use: text | image_file_name")

        elif user_input.endswith(".jpg") or user_input.endswith(".png"):
            print(agent.handle_image_query(user_input))

        else:
            print(agent.handle_general_conversation(user_input))
### #%#