"""RAG (Retrieval Augmented Generation) for brain science knowledge.

Uses LanceDB (embedded vector DB) + FastEmbed (ONNX-based, no PyTorch).
Knowledge base covers: neuroanatomy, MRI modalities, brain atlases, preprocessing.
"""

import os
from pathlib import Path
from typing import Any

from pybnt.core.logconf import logger

DEFAULT_STORE_DIR = os.path.expanduser("~/.pybnt/rag_store")
KNOWLEDGE_DIR = Path(__file__).parent.parent / "data" / "knowledge"


class BrainKnowledgeStore:
    """Brain science knowledge base using LanceDB + FastEmbed.

    Stores brain science documents as vector embeddings in an embedded
    LanceDB database. Supports semantic search over neuroanatomy,
    MRI modalities, brain atlases, and preprocessing topics.

    Attributes:
        store_dir: Path to the LanceDB database directory.
    """

    def __init__(self, store_dir: str | None = None):
        self.store_dir = store_dir or DEFAULT_STORE_DIR
        self._table = None
        self._embedder = None

    @property
    def _is_initialized(self) -> bool:
        return self._table is not None

    def _ensure_deps(self):
        """Lazy import dependencies."""
        try:
            import lancedb  # noqa: F401
        except ImportError:
            raise ImportError(
                "LanceDB not installed. Run: pip install lancedb"
            )
        try:
            from fastembed import TextEmbedding  # noqa: F401
        except ImportError:
            raise ImportError(
                "FastEmbed not installed. Run: pip install fastembed"
            )

    def _get_embedder(self):
        if self._embedder is None:
            self._ensure_deps()
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return self._embedder

    def _ensure_db(self):
        """Initialize or open LanceDB database and table."""
        if self._table is not None:
            return self._table
        import lancedb
        os.makedirs(self.store_dir, exist_ok=True)
        db = lancedb.connect(self.store_dir)
        table_name = "brain_knowledge"

        if table_name in db.list_tables().tables:
            self._table = db.open_table(table_name)
        else:
            self._create_knowledge_base(db, table_name)
        return self._table

    def _create_knowledge_base(self, db, table_name: str):
        """Create table from knowledge documents with embeddings.

        Loads markdown documents from KNOWLEDGE_DIR, generates embeddings
        using FastEmbed, and creates a LanceDB table with the vectors.
        Falls back to built-in defaults if no documents are found.
        """
        documents = self._load_documents()
        if not documents:
            logger.warning(
                "No knowledge documents found. Using built-in defaults."
            )
            documents = _DEFAULT_KNOWLEDGE

        embedder = self._get_embedder()
        texts = [d["text"] for d in documents]
        embeddings = list(embedder.embed(texts))

        data = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{i}"
            text = doc["text"]
            category = doc.get("category", "general")
            vector = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            data.append({
                "id": doc_id,
                "text": text,
                "category": category,
                "vector": vector,
            })

        self._table = db.create_table(table_name, data)
        logger.info(
            "Created knowledge base with %d documents", len(documents)
        )

    def _load_documents(self) -> list[dict]:
        """Load plain-text documents from the knowledge markdown directory.

        Each .md file is treated as a single document. The file stem
        becomes the category label (e.g. ``neuroanatomy.md`` →
        ``"neuroanatomy"``).
        """
        documents = []
        if not KNOWLEDGE_DIR.exists():
            return documents
        for f in sorted(KNOWLEDGE_DIR.glob("*.md")):
            text = f.read_text(encoding="utf-8").strip()
            if text:
                documents.append({
                    "text": text,
                    "category": f.stem,
                })
        return documents

    def query(self, query_text: str, k: int = 5) -> list[dict[str, Any]]:
        """Query the knowledge base for relevant documents.

        Args:
            query_text: Natural language query string.
            k: Number of top results to return (default: 5).

        Returns:
            List of dicts with ``"text"``, ``"category"``, and ``"score"``
            keys, ordered by descending relevance.
        """
        self._ensure_db()
        embedder = self._get_embedder()
        query_embedding = list(embedder.embed([query_text]))[0]

        results = self._table.search(query_embedding).limit(k).to_list()
        return [
            {
                "text": r["text"],
                "category": r.get("category", "general"),
                "score": float(r.get("_distance", 0)),
            }
            for r in results
        ]

    def reindex(self) -> int:
        """Drop and rebuild the knowledge base from source documents.

        Returns:
            Number of documents loaded.
        """
        self._table = None
        import lancedb
        db = lancedb.connect(self.store_dir)
        table_name = "brain_knowledge"
        if table_name in db.list_tables().tables:
            db.drop_table(table_name)
        self._ensure_db()
        return len(self._load_documents())


# ---------------------------------------------------------------------------
# Built-in default knowledge base (used when no .md files are present)
# ---------------------------------------------------------------------------
_DEFAULT_KNOWLEDGE = [
    {
        "text": (
            "fMRI (functional MRI) measures brain activity by detecting "
            "changes in blood flow. It relies on the BOLD (Blood Oxygen "
            "Level Dependent) effect, where active brain regions consume "
            "more oxygen, changing the magnetic properties of blood."
        ),
        "category": "imaging",
    },
    {
        "text": (
            "The Default Mode Network (DMN) is a network of brain regions "
            "that are active when the brain is at rest. It includes the "
            "medial prefrontal cortex, posterior cingulate cortex, and "
            "angular gyrus. DMN activity decreases during task-focused "
            "activities."
        ),
        "category": "functional",
    },
    {
        "text": (
            "Brain atlases parcellate the brain into regions. AAL "
            "(Anatomical Automatic Labeling) has 90 regions. "
            "Desikan-Killiany has 68 cortical regions. Power264 defines "
            "264 functional regions. Brodmann areas map 52 "
            "cytoarchitectonic regions."
        ),
        "category": "atlases",
    },
    {
        "text": (
            "T1-weighted MRI provides high contrast between gray matter "
            "and white matter, making it ideal for anatomical imaging and "
            "segmentation. T2-weighted MRI is sensitive to water content, "
            "useful for detecting edema, inflammation, and lesions."
        ),
        "category": "imaging",
    },
    {
        "text": (
            "DTI (Diffusion Tensor Imaging) measures water diffusion "
            "along axons to map white matter tracts. Key metrics include "
            "FA (Fractional Anisotropy), MD (Mean Diffusivity), and "
            "tractography-based connectivity."
        ),
        "category": "imaging",
    },
    {
        "text": (
            "BrainVISA is a free, open-source neuroimaging software "
            "platform. It provides tools for cortical surface extraction, "
            "sulcal identification, and morphometric analysis."
        ),
        "category": "tools",
    },
]


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
def build_knowledge_base(store_dir: str | None = None) -> int:
    """Build or re-index the knowledge base from source documents.

    Args:
        store_dir: Optional custom database directory.

    Returns:
        Number of documents indexed.
    """
    store = BrainKnowledgeStore(store_dir)
    return store.reindex()


def query_knowledge(
    query: str,
    k: int = 5,
    store_dir: str | None = None,
) -> list[dict]:
    """Query the brain science knowledge base.

    Args:
        query: Natural language query string.
        k: Number of results to return (default: 5).
        store_dir: Optional custom database directory.

    Returns:
        List of matching documents with text, category, and score.
    """
    store = BrainKnowledgeStore(store_dir)
    return store.query(query, k)


# ---------------------------------------------------------------------------
# CLI support
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--index":
        count = build_knowledge_base()
        print(f"Indexed {count} knowledge documents.")
    elif len(sys.argv) > 2 and sys.argv[1] == "--query":
        results = query_knowledge(sys.argv[2])
        for r in results:
            print(
                f"[{r['category']}] ({r['score']:.3f}) "
                f"{r['text'][:120]}..."
            )
    else:
        print("Usage: python -m pybnt.ai.rag --index")
        print(
            "       python -m pybnt.ai.rag --query 'default mode network'"
        )
