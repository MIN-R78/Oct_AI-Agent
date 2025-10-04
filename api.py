### MIn Li - AI Agent
from fastapi import FastAPI, Query
from pydantic import BaseModel
from agent.commerce_agent import CommerceAgent

### create FastAPI app
app = FastAPI(title="Palona AI Agent API", version="1.0")

### init the backend agent once
agent = CommerceAgent()

### request schema for mixed queries
class MixedQuery(BaseModel):
    text: str
    image: str

@app.get("/")
def root():
    return {"message": "Palona AI Agent API is running."}

### endpoint for general chat (small talk or fallback search)
@app.get("/chat")
def chat(query: str = Query(..., description="User input for small talk or general query")):
    response = agent.handle_general_conversation(query)
    return {"query": query, "response": response}

### endpoint for text-only product search
@app.get("/search/text")
def search_text(query: str = Query(..., description="Search by text, e.g. 'running shoes'")):
    response = agent.handle_query(query)
    return {"query": query, "results": response}

### endpoint for image-only product search
@app.get("/search/image")
def search_image(image: str = Query(..., description="Filename of the image inside /data/images")):
    response = agent.handle_image_query(image)
    return {"image": image, "results": response}

### endpoint for mixed (text + image) search
@app.post("/search/mixed")
def search_mixed(payload: MixedQuery):
    response = agent.handle_mixed_query(payload.text, payload.image)
    return {"query": payload.text, "image": payload.image, "results": response}
