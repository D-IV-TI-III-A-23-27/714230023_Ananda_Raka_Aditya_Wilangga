"""
Qdrant Vector Store Operations — Dense-only dengan DistilBERT.

Memakai SentenceTransformer `distilbert-base-nli-stsb-mean-tokens` untuk
menghasilkan vektor **dense** (768-d), menyimpan di Qdrant sebagai
*named vector* ("dense"), lalu melakukan **dense search** dengan cosine similarity.

Setiap chunk di-embed dengan **konteks** (judul + section header) diprepend ke
isi, agar sub-chunk yang panjang tidak kehilangan konteks section.
"""

import os
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load env variables (for Qdrant Cloud)
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

from core.chunker import Chunk
from core.retry import retry_call

logger = logging.getLogger(__name__)

# === Configuration ===
COLLECTION_NAME = "scientific_articles"
EMBEDDING_MODEL = "distilbert-base-nli-stsb-mean-tokens"  # 512 context, 768-d, English
DENSE_DIM = 768
MAX_LENGTH = 512  # DistilBERT context window

# Named vectors in Qdrant
DENSE_VECTOR = "dense"

# Qdrant Database Settings (Local or Cloud)
QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_db")
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")


def _to_list(vec) -> list:
    """Konversi vektor (numpy / list) ke list float."""
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


class VectorStore:
    """
    Mengelola embedding (dense-only DistilBERT) dan penyimpanan di Qdrant.

    Usage:
        store = VectorStore()
        store.ensure_collection()
        store.add_chunks(chunks)
    """

    def __init__(
        self,
        qdrant_path: str = QDRANT_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        collection_name: str = COLLECTION_NAME,
    ):
        logger.info("Initializing VectorStore...")
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.hybrid = False  # DistilBERT is dense-only
        self.backend = "SentenceTransformer"

        self._load_model(embedding_model)

        # Connect to Cloud if URL is provided, otherwise use Local DB
        if QDRANT_URL and QDRANT_API_KEY:
            # Guard: kesalahan umum adalah menukar QDRANT_URL <-> QDRANT_API_KEY.
            # Validasi sebelum konek, dan JANGAN cetak nilai URL/key ke log
            # (cegah kebocoran key jika tertukar).
            if not QDRANT_URL.lower().startswith(("http://", "https://")):
                raise ValueError(
                    "QDRANT_URL tidak valid: harus diawali 'http://' atau 'https://' "
                    "(mis. https://xxx.qdrant.io). Kemungkinan tertukar dengan "
                    "QDRANT_API_KEY - periksa kembali Colab Secrets / environment."
                )
            if QDRANT_API_KEY.lower().startswith(("http://", "https://")):
                raise ValueError(
                    "QDRANT_API_KEY tampak berisi URL - kemungkinan tertukar dengan "
                    "QDRANT_URL. Periksa kembali Colab Secrets / environment."
                )
            host = urlparse(QDRANT_URL).hostname or "?"
            logger.info(f"Connecting to Qdrant Cloud at {host}")  # host saja, bukan key
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            logger.info(f"Connecting to Qdrant Local DB at {qdrant_path}")
            self.client = QdrantClient(path=qdrant_path)

        logger.info(
            f"VectorStore ready. Backend={self.backend}, "
            f"dim={self.vector_size}"
        )

    # ------------------------------------------------------------------ #
    #  Model loading
    # ------------------------------------------------------------------ #
    def _load_model(self, embedding_model: str):
        """Load DistilBERT model via SentenceTransformer (dense-only)."""
        orig_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        orig_trans_offline = os.environ.get("TRANSFORMERS_OFFLINE")

        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                embedding_model, trust_remote_code=True, local_files_only=True
            )
            logger.info("Loaded model from local cache (Offline Mode).")
        except Exception as e:
            logger.debug(f"Offline load attempt failed: {e}")
            if orig_hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = orig_hf_offline
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)
            if orig_trans_offline is not None:
                os.environ["TRANSFORMERS_OFFLINE"] = orig_trans_offline
            else:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)

            logger.info("Model not found in local cache. Connecting to Hugging Face...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                embedding_model, trust_remote_code=True, local_files_only=False
            )
        finally:
            if orig_hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = orig_hf_offline
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)
            if orig_trans_offline is not None:
                os.environ["TRANSFORMERS_OFFLINE"] = orig_trans_offline
            else:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)

        self.backend = "SentenceTransformer"
        self.vector_size = self.model.get_embedding_dimension()

        # Set max_seq_length untuk DistilBERT (512 tokens)
        try:
            current = getattr(self.model, "max_seq_length", None)
            logger.info(f"SentenceTransformer max_seq_length: {current}")
            if current is None or current < MAX_LENGTH:
                self.model.max_seq_length = MAX_LENGTH
                logger.info(f"Set max_seq_length to {MAX_LENGTH} for DistilBERT.")
        except Exception as e:
            logger.debug(f"Tidak bisa inspeksi/atur max_seq_length: {e}")

    # ------------------------------------------------------------------ #
    #  Embedding helpers
    # ------------------------------------------------------------------ #
    def _doc_text(self, chunk: Chunk) -> str:
        """Prepend konteks (judul + section header) ke isi chunk sebelum embedding."""
        ctx = []
        title = (chunk.title or "").strip()
        section = (chunk.section_header or "").strip()
        if title:
            ctx.append(title)
        if section and section.lower() != "unknown" and section != title:
            ctx.append(section)
        head = " | ".join(ctx)
        return f"{head}\n\n{chunk.content}" if head else chunk.content

    def _embed_documents(self, texts: list[str]):
        """Return (dense_list, None) untuk daftar teks dokumen. Dense-only."""
        dense = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=len(texts) > 64
        )
        return dense, None

    def _embed_query(self, query: str):
        """Return (dense_vec, None) untuk satu query. Dense-only."""
        return _to_list(self.model.encode(query, normalize_embeddings=True)), None

    # ------------------------------------------------------------------ #
    #  Collection management
    # ------------------------------------------------------------------ #
    def ensure_collection(self):
        """Create collection (named dense vector) jika belum ada."""
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR: VectorParams(
                        size=self.vector_size, distance=Distance.COSINE
                    )
                },
            )
            logger.info(f"Created collection (dense-only): {self.collection_name}")
        else:
            logger.info(f"Collection already exists: {self.collection_name}")

        # Ensure payload index for article_id (required for count/scroll filters)
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="article_id",
                field_schema="keyword",
            )
            logger.info("Ensured payload index on 'article_id'.")
        except Exception as e:
            logger.debug(f"Payload index creation note (might already exist): {e}")

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """Embed (dense) & simpan chunks ke Qdrant. Return jumlah chunk tersimpan."""
        if not chunks:
            logger.warning("No chunks to store")
            return 0

        logger.info(f"Embedding {len(chunks)} chunks...")

        all_points = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [self._doc_text(c) for c in batch]  # konteks diprepend
            dense, _ = self._embed_documents(texts)

            for j, chunk in enumerate(batch):
                vector = {DENSE_VECTOR: _to_list(dense[j])}

                all_points.append(
                    PointStruct(
                        id=chunk.chunk_id,
                        vector=vector,
                        payload={
                            "content": chunk.content,  # isi asli tanpa prefix konteks
                            **chunk.to_metadata_dict(),
                        },
                    )
                )

        retry_call(
            lambda: self.client.upsert(
                collection_name=self.collection_name,
                points=all_points,
            ),
            label="Qdrant upsert",
        )

        logger.info(f"Stored {len(all_points)} chunks in Qdrant")
        return len(all_points)

    def delete_article(self, article_id: str) -> bool:
        """Hapus semua chunk milik satu artikel."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="article_id", match=MatchValue(value=article_id))]
            ),
        )
        logger.info(f"Deleted all chunks for article {article_id}")
        return True

    def article_exists(self, article_id: str) -> dict | bool:
        """Cek apakah artikel sudah ter-ingest penuh; bersihkan jika parsial.
        Return dict metadata bila ada & lengkap, selain itu False."""
        result = retry_call(
            lambda: self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="article_id", match=MatchValue(value=article_id))]
                ),
                exact=True,
            ),
            label="Qdrant count",
        )

        count = result.count
        if count == 0:
            return False

        points, _ = retry_call(
            lambda: self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="article_id", match=MatchValue(value=article_id))]
                ),
                limit=1,
                with_payload=["total_chunks", "title", "doi"],
                with_vectors=False,
            ),
            label="Qdrant scroll",
        )

        if points:
            expected_chunks = points[0].payload.get("total_chunks", 0)
            if expected_chunks > 0 and count >= expected_chunks:
                return {
                    "title": points[0].payload.get("title", "Unknown"),
                    "doi": points[0].payload.get("doi", "Unknown"),
                }

            logger.warning(
                f"Detected partial insertion for article {article_id} "
                f"({count}/{expected_chunks} chunks). Cleaning up for re-processing..."
            )
            self.delete_article(article_id)

        return False

    def get_collection_info(self) -> dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": getattr(info, "vectors_count", info.points_count),
            "points_count": info.points_count,
            "status": info.status.value,
        }

    def list_articles(self) -> list[dict]:
        """List semua artikel unik di collection."""
        articles = {}
        offset = None

        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=["article_id", "title", "authors", "doi", "total_chunks"],
                with_vectors=False,
            )

            for point in results:
                aid = point.payload.get("article_id")
                if aid and aid not in articles:
                    articles[aid] = {
                        "article_id": aid,
                        "title": point.payload.get("title", "Untitled"),
                        "authors": point.payload.get("authors", ""),
                        "doi": point.payload.get("doi", ""),
                        "total_chunks": point.payload.get("total_chunks", 0),
                    }

            if offset is None:
                break

        return list(articles.values())

    # ------------------------------------------------------------------ #
    #  Search (dense-only with cosine similarity)
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        n_results: int = 10,
        article_ids: list[str] | None = None,
        section_filter: str | None = None,
        doi_filter: str | None = None,
    ) -> list[dict]:
        """
        Cari chunk relevan menggunakan dense cosine similarity.
        Return list dict {content, score, metadata}.
        """
        must_conditions = []

        if article_ids:
            if len(article_ids) == 1:
                must_conditions.append(
                    FieldCondition(key="article_id", match=MatchValue(value=article_ids[0]))
                )
            else:
                must_conditions.append(
                    FieldCondition(key="article_id", match=MatchAny(any=article_ids))
                )

        if section_filter:
            must_conditions.append(
                FieldCondition(key="section_header", match=MatchValue(value=section_filter))
            )

        if doi_filter:
            must_conditions.append(
                FieldCondition(key="doi", match=MatchValue(value=doi_filter))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        dense_vec, _ = self._embed_query(query)

        results = retry_call(
            lambda: self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vec,
                using=DENSE_VECTOR,
                query_filter=query_filter,
                limit=n_results,
                with_payload=True,
            ),
            label="Qdrant search",
        )

        return [
            {
                "content": point.payload.get("content", ""),
                "score": point.score,
                "metadata": {k: v for k, v in point.payload.items() if k != "content"},
            }
            for point in results.points
        ]
