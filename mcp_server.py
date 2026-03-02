import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

# [1] Get absolute path
ROOT = os.path.dirname(os.path.abspath(__file__))
def get_path(name):
    return os.path.join(ROOT, name)

# [2] Server setup
mcp = FastMCP("AgeWell-Connect")

# [3] Load core assets
load_dotenv(get_path(".env"))
MODEL = SentenceTransformer('BAAI/bge-m3')
INDEX = faiss.read_index(get_path("health_index.faiss"))

# Fixed: changed back to doc_map.pkl
with open(get_path("doc_map.pkl"), "rb") as f:
    DOC_MAP = pickle.load(f)

# [4] Expert tool with 5-step logic
@mcp.tool()
def query_longevity_expert(query: str) -> str:
    """Search clinical evidence and return a structured report."""
    
    # Vector Search
    emb = MODEL.encode([query], normalize_embeddings=True)
    _, ids = INDEX.search(np.array(emb).astype('float32'), 3)
    evidence = "\n\n".join([DOC_MAP[i] for i in ids[0]])
    
    # Logic Wrapper
    prompt_guide = (
        "TASK: Create a 500-word clinical report using the evidence below.\n"
        "FORMAT: 1. OVERVIEW, 2. MECHANISM, 3. TRIAD PROTOCOL (SHIFT/FIX/ENHANCE), "
        "4. 90-DAY GOALS, 5. FOOTNOTE.\n\n"
        f"EVIDENCE CONTENT:\n{evidence}"
    )
    
    return prompt_guide

if __name__ == "__main__":
    mcp.run(transport='stdio')