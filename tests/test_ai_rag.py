"""Tests for pybnt.ai.rag — LanceDB + FastEmbed knowledge store."""

from unittest.mock import MagicMock, patch

import pytest

from pybnt.ai.rag import (
    BrainKnowledgeStore,
    _DEFAULT_KNOWLEDGE,
    build_knowledge_base,
    query_knowledge,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _mock_embed(texts):
    """Return dummy 384-dim float vectors."""
    return [[float(i + j) for j in range(384)] for i in range(len(texts))]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestKnowledgeStoreInit:
    """Initialization tests that do not touch disk or network."""

    def test_knowledge_store_init(self):
        store = BrainKnowledgeStore(store_dir="/tmp/test_rag")
        assert store.store_dir == "/tmp/test_rag"
        assert store._table is None
        assert store._embedder is None
        assert store._is_initialized is False

    def test_default_store_dir(self):
        store = BrainKnowledgeStore()
        assert store.store_dir.endswith(".pybnt/rag_store")

    def test_ensure_deps_raises_on_lancedb(self):
        store = BrainKnowledgeStore()
        with patch.dict("sys.modules", {"lancedb": None}):
            with pytest.raises(ImportError, match="LanceDB not installed"):
                store._ensure_deps()

    def test_ensure_deps_raises_on_fastembed(self):
        store = BrainKnowledgeStore()
        with patch.dict("sys.modules", {"lancedb": MagicMock()}):
            with patch("pybnt.ai.rag.TextEmbedding", create=True) as mock_te:
                mock_te.side_effect = ImportError("no fastembed")
                with patch.dict("sys.modules", {"fastembed": None}):
                    with pytest.raises(
                        ImportError, match="FastEmbed not installed"
                    ):
                        store._ensure_deps()


class TestQuery:
    """Query tests that mock external dependencies."""

    def test_query_returns_results(self):
        store = BrainKnowledgeStore(store_dir="/tmp/test_rag_q")
        store._table = MagicMock()
        store._table.search.return_value.limit.return_value.to_list.return_value = [
            {"text": "T1 MRI contrast", "category": "imaging", "_distance": 0.12},
            {"text": "BOLD fMRI effect", "category": "imaging", "_distance": 0.18},
        ]
        store._embedder = MagicMock()
        store._embedder.embed.return_value = [[0.0] * 384]

        results = store.query("What is T1 MRI?")
        assert len(results) == 2
        assert results[0]["text"] == "T1 MRI contrast"
        assert results[0]["category"] == "imaging"
        assert results[0]["score"] == 0.12

    def test_empty_store_uses_defaults(self):
        """When _load_documents returns [], _DEFAULT_KNOWLEDGE is used."""
        with patch.object(
            BrainKnowledgeStore, "_load_documents", return_value=[]
        ):
            store = BrainKnowledgeStore(store_dir="/tmp/test_rag_def")
            store._embedder = MagicMock()
            store._embedder.embed = MagicMock(
                side_effect=lambda texts: _mock_embed(texts)
            )
            fake_db = MagicMock()
            fake_table = MagicMock()
            fake_db.create_table.return_value = fake_table

            store._create_knowledge_base(fake_db, "brain_knowledge")

            fake_db.create_table.assert_called_once()
            data_arg = fake_db.create_table.call_args[0][1]
            assert len(data_arg) == len(_DEFAULT_KNOWLEDGE)
            assert store._table is fake_table


class TestReindex:
    def test_reindex_clears_and_rebuilds(self):
        store = BrainKnowledgeStore(store_dir="/tmp/test_rag_reidx")
        with patch.object(
            store, "_load_documents", return_value=[]
        ) as mock_load:
            store._table = MagicMock()
            count = store.reindex()
            assert count == 0
            mock_load.assert_called()

    def test_build_knowledge_base_convenience(self):
        with patch(
            "pybnt.ai.rag.BrainKnowledgeStore.reindex", return_value=4
        ) as mock_reindex:
            count = build_knowledge_base("/fake/store")
            assert count == 4
            mock_reindex.assert_called_once()


class TestDefaults:
    def test_builtin_default_knowledge_has_topics(self):
        categories = {d["category"] for d in _DEFAULT_KNOWLEDGE}
        assert "imaging" in categories
        assert "functional" in categories
        assert "atlases" in categories

    def test_builtin_default_text_is_non_empty(self):
        for doc in _DEFAULT_KNOWLEDGE:
            assert len(doc["text"]) > 50
            assert doc["category"]


class TestQueryKnowledgeConvenience:
    def test_query_knowledge_convenience(self):
        with patch.object(
            BrainKnowledgeStore,
            "query",
            return_value=[
                {"text": "result", "category": "imaging", "score": 0.5}
            ],
        ) as mock_q:
            results = query_knowledge("T1")
            assert len(results) == 1
            mock_q.assert_called_once_with("T1", 5)
