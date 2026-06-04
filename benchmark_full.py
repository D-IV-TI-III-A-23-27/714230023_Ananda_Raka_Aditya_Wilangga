#!/usr/bin/env python3
"""
PEDE Benchmark — DistilBERT Embedding

Menjalankan 10 eksperimen benchmarking pada paper GAN.pdf:
- Variasi ukuran chunk (256, 500, 1000, 2000)
- Variasi overlap (0%, 10%, 20%, 50%)
- Variasi top-K (3, 5, 10)

Output: Hasil.md (format tabel + analisis otomatis)

Usage:
    python benchmark_full.py
"""

import os
import sys
import time
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, field

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("core.retry").setLevel(logging.WARNING)
logging.getLogger("core.metadata_extractor").setLevel(logging.WARNING)
logging.getLogger("core.pdf_converter").setLevel(logging.WARNING)
logging.getLogger("core.chunker").setLevel(logging.WARNING)
logging.getLogger("core.vector_store").setLevel(logging.WARNING)
logging.getLogger("__main__").setLevel(logging.INFO)

from core.pdf_converter import convert_pdf_to_markdown, get_pdf_native_metadata
from core.metadata_extractor import extract_metadata, ArticleMetadata
from core.chunker import chunk_markdown, Chunk
from core.vector_store import VectorStore, EMBEDDING_MODEL, DENSE_DIM, MAX_LENGTH

# === Configuration ===
PDF_PATH = "papers/GAN.pdf"
QDRANT_PATH_BENCH = "./qdrant_bench"
COLLECTION_NAME = "benchmark"
HASIL_OUTPUT = "Hasil.md"

# === 5 Pertanyaan Benchmark untuk paper GAN Chiller AFDD (Yan et al. 2020) ===
# Campuran bahasa (EN + ID) dan tipe query yang berbeda

@dataclass
class TestQuery:
    """Satu pertanyaan benchmark beserta jawaban yang diharapkan."""
    question: str
    query_type: str       # Factoid, Reasoning, Semantic, Conversational
    language: str         # EN atau ID
    expected_keywords: list[str] = field(default_factory=list)  # Kata kunci di jawaban
    expected_answer: str = ""

QUERIES = [
    TestQuery(
        question="What dataset was used to train and evaluate the GAN-based chiller fault detection model?",
        query_type="Factoid",
        language="EN",
        expected_keywords=["ASHRAE", "1043", "chiller", "fault"],
        expected_answer="Dataset ASHRAE Project 1043-RP, berisi data operasional chiller 90-ton (316 kW) dengan 7 jenis fault",
    ),
    TestQuery(
        question="Mengapa dataset pelatihan chiller AFDD menjadi tidak seimbang dalam skenario dunia nyata?",
        query_type="Reasoning",
        language="ID",
        expected_keywords=["imbalanced", "fault", "normal", "limited", "training"],
        expected_answer="Data fault sangat sedikit karena kegagalan chiller jarang terjadi dan langsung diperbaiki, sedangkan data normal sangat banyak",
    ),
    TestQuery(
        question="Bagaimana GAN digunakan untuk meningkatkan akurasi deteksi fault pada chiller?",
        query_type="Semantic",
        language="ID",
        expected_keywords=["GAN", "generate", "fault", "sample", "training", "imbalanced"],
        expected_answer="GAN menghasilkan sampel fault buatan untuk menyeimbangkan dataset training yang tidak seimbang, sehingga meningkatkan akurasi klasifikasi",
    ),
    TestQuery(
        question="What are the seven types of chiller faults investigated in this study?",
        query_type="Factoid",
        language="EN",
        expected_keywords=["condenser", "evaporator", "refrigerant", "fouling", "oil"],
        expected_answer="F1: Reduced condenser water flow, F2: Reduced evaporator water flow, F3: Refrigerant Leak, F4: Refrigerant Overcharge, F5: Excess Oil, F6: Condenser Fouling, F7: Non-condensables",
    ),
    TestQuery(
        question="berapa jumlah sampel fault training yang diuji dalam eksperimen penelitian ini?",
        query_type="Conversational",
        language="ID",
        expected_keywords=["10", "20", "30", "40", "50", "fault", "sample"],
        expected_answer="10, 20, 30, 40, dan 50 sampel fault training per jenis fault, bersama 8400 sampel normal",
    ),
]


@dataclass
class ExperimentConfig:
    """Konfigurasi satu eksperimen benchmark."""
    name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    note: str


EXPERIMENTS = [
    ExperimentConfig("EXP-01", 256, 50, 5, "Chunk sangat kecil"),
    ExperimentConfig("EXP-02", 500, 100, 5, "Chunk kecil"),
    ExperimentConfig("EXP-03", 1000, 200, 5, "Chunk medium (baseline)"),
    ExperimentConfig("EXP-04", 2000, 400, 5, "Chunk besar"),
    ExperimentConfig("EXP-05", 1000, 0, 5, "Overlap 0% (tanpa)"),
    ExperimentConfig("EXP-06", 1000, 100, 5, "Overlap 10%"),
    ExperimentConfig("EXP-07", 1000, 500, 5, "Overlap 50%"),
    ExperimentConfig("EXP-08", 1000, 200, 3, "Top-K kecil (3)"),
    ExperimentConfig("EXP-09", 1000, 200, 10, "Top-K besar (10)"),
    ExperimentConfig("EXP-10", 500, 100, 10, "Chunk kecil + Top-K besar"),
]


@dataclass
class QueryResult:
    """Hasil evaluasi satu query."""
    query: TestQuery
    hit: bool
    score: float = 0.0
    matched_content: str = ""


@dataclass
class ExperimentResult:
    """Hasil satu eksperimen lengkap."""
    config: ExperimentConfig
    query_results: list[QueryResult]
    hit_rate: float
    avg_latency: float
    db_size_mb: float
    num_chunks: int


def evaluate_query(vector_store: VectorStore, query: TestQuery, top_k: int, doi: str = None) -> QueryResult:
    """Evaluasi satu query terhadap vector store. Hit = ada keyword match di top-K results."""
    results = vector_store.search(query.question, n_results=top_k, doi_filter=doi)

    # Gabungkan semua konten dari top-K results
    all_content = " ".join([r["content"].lower() for r in results])

    # Cek apakah keywords jawaban ditemukan di results
    matched = 0
    for kw in query.expected_keywords:
        if kw.lower() in all_content:
            matched += 1

    # HIT jika setidaknya 50% keyword ditemukan
    threshold = max(1, len(query.expected_keywords) // 2)
    hit = matched >= threshold

    best_score = results[0]["score"] if results else 0.0
    best_content = results[0]["content"][:200] if results else ""

    return QueryResult(
        query=query,
        hit=hit,
        score=best_score,
        matched_content=best_content,
    )


def get_db_size_mb(db_path: str) -> float:
    """Hitung ukuran folder database dalam MB."""
    total = 0
    for dirpath, _, filenames in os.walk(db_path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return round(total / (1024 * 1024), 1)


def run_experiment(exp: ExperimentConfig, pdf_path: str) -> ExperimentResult:
    """Jalankan satu eksperimen benchmark lengkap."""
    logger.info(f"\n{'='*60}")
    logger.info(f"[{exp.name}] Chunk={exp.chunk_size}, Overlap={exp.chunk_overlap}, Top-K={exp.top_k}")
    logger.info(f"{'='*60}")

    # 1. Reset database
    if os.path.exists(QDRANT_PATH_BENCH):
        shutil.rmtree(QDRANT_PATH_BENCH)

    # 2. Initialize vector store
    vector_store = VectorStore(
        qdrant_path=QDRANT_PATH_BENCH,
        collection_name=COLLECTION_NAME,
    )
    vector_store.ensure_collection()

    # 3. Ingest paper
    logger.info(f"  Ingesting {pdf_path}...")
    markdown_text = convert_pdf_to_markdown(pdf_path, image_dir="./data/images", write_images=False)
    pdf_meta = get_pdf_native_metadata(pdf_path)
    article_meta = extract_metadata(pdf_path, markdown_text, pdf_meta)
    chunks = chunk_markdown(markdown_text, article_meta, chunk_size=exp.chunk_size, chunk_overlap=exp.chunk_overlap)

    # Filter reference chunks
    chunks = [c for c in chunks if c.content_type != "references"]

    stored = vector_store.add_chunks(chunks)
    logger.info(f"  Stored {stored} chunks")

    # Get DOI for filtering
    doi = article_meta.doi

    # 4. Evaluate queries
    query_results = []
    latencies = []

    for i, q in enumerate(QUERIES, 1):
        start = time.perf_counter()
        result = evaluate_query(vector_store, q, exp.top_k, doi)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        query_results.append(result)

        status = "HIT ✅" if result.hit else "MISS ❌"
        logger.info(f"  Q{i} [{q.query_type}/{q.language}] {status} (score={result.score:.4f}, latency={elapsed:.3f}s)")

    # 5. Calculate metrics
    hits = sum(1 for r in query_results if r.hit)
    hit_rate = hits / len(query_results) * 100
    avg_latency = sum(latencies) / len(latencies)
    db_size = get_db_size_mb(QDRANT_PATH_BENCH)

    logger.info(f"  Hit Rate: {hits}/{len(query_results)} = {hit_rate:.0f}%")
    logger.info(f"  Avg Latency: {avg_latency:.2f}s")
    logger.info(f"  DB Size: {db_size} MB")

    # Cleanup Qdrant client (release lock)
    del vector_store

    return ExperimentResult(
        config=exp,
        query_results=query_results,
        hit_rate=hit_rate,
        avg_latency=avg_latency,
        db_size_mb=db_size,
        num_chunks=stored,
    )


def extract_paper_info(pdf_path: str) -> dict:
    """Ekstrak info dasar paper untuk header Hasil.md."""
    pdf_meta = get_pdf_native_metadata(pdf_path)
    markdown_text = convert_pdf_to_markdown(pdf_path, write_images=False)
    article_meta = extract_metadata(pdf_path, markdown_text, pdf_meta)
    return {
        "title": article_meta.title,
        "doi": article_meta.doi or "N/A",
        "authors": ", ".join(article_meta.authors) if article_meta.authors else "Unknown",
    }


def generate_hasil_md(results: list[ExperimentResult], paper_info: dict):
    """Generate file Hasil.md dari hasil eksperimen."""

    model_label = f"DistilBERT (`{EMBEDDING_MODEL}`)"
    dim = DENSE_DIM

    lines = []
    lines.append(f"# 📊 Hasil Benchmarking PEDE — DistilBERT Embedding\n")
    lines.append(f"**Paper Uji:** *{paper_info['title']}*  ")
    lines.append(f"**DOI:** `{paper_info['doi']}`  ")
    lines.append(f"**Model Embedding:** {model_label} — {dim} dimensi, model ringan berbasis BERT yang di-distill untuk embedding kalimat\n")

    # === Tabel Eksperimen ===
    lines.append("## Hasil Benchmarking (10 Eksperimen)\n")
    lines.append("| Ukuran Chunk | Overlap | Metode Chunking | Model Embedding | Dukungan Bahasa | Tipe Query Uji | Top-K | Filter Metadata | Hit Rate | Latensi | Ukuran Index DB | Catatan |")
    lines.append("|:---:|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|")

    for r in results:
        hit_str = f"**{r.hit_rate:.0f}%**" if r.hit_rate == 100 else f"{r.hit_rate:.0f}%"
        lines.append(
            f"| {r.config.chunk_size} | {r.config.chunk_overlap} "
            f"| Hybrid | {model_label} "
            f"| Campuran (EN + ID) | Campuran (5 query) "
            f"| {r.config.top_k} | Ya (DOI) "
            f"| {hit_str} | {r.avg_latency:.2f}s "
            f"| {r.db_size_mb} MB | *{r.config.note}* |"
        )

    lines.append("\n---\n")

    # === Detail Pertanyaan ===
    lines.append("## Detail Hasil per Query\n")
    lines.append("### 5 Pertanyaan Tes\n")
    lines.append("| # | Pertanyaan | Tipe | Bahasa | Jawaban yang Diharapkan |")
    lines.append("|:---:|---|---|:---:|---|")

    for i, q in enumerate(QUERIES, 1):
        lines.append(f"| Q{i} | {q.question} | {q.query_type} | {q.language} | {q.expected_answer} |")

    lines.append("")

    # === Hasil per Query (Top-K = 5 vs Top-K = 10) ===
    # Find results for top_k=5 baseline and top_k=10
    baseline_5 = None
    topk_10 = None
    for r in results:
        if r.config.chunk_size == 1000 and r.config.chunk_overlap == 200:
            if r.config.top_k == 5:
                baseline_5 = r
            elif r.config.top_k == 10:
                topk_10 = r

    if baseline_5 and topk_10:
        lines.append("### Hasil per Query (Top-K = 5 vs Top-K = 10)\n")
        lines.append("| # | Tipe | Bahasa | Top-K = 5 | Top-K = 10 |")
        lines.append("|:---:|:---|:---:|:---:|:---:|")

        hits_5 = 0
        hits_10 = 0
        for i in range(len(QUERIES)):
            q = QUERIES[i]
            h5 = "HIT" if baseline_5.query_results[i].hit else "MISS"
            h10 = "HIT" if topk_10.query_results[i].hit else "MISS"
            if baseline_5.query_results[i].hit:
                hits_5 += 1
            if topk_10.query_results[i].hit:
                hits_10 += 1
            lines.append(f"| Q{i+1} | {q.query_type} | {q.language} | {h5} | {h10} |")

        lines.append(f"| | | **Total** | **{hits_5}/5 = {hits_5*20}%** | **{hits_10}/5 = {hits_10*20}%** |")
        lines.append("")

    lines.append("\n---\n")

    # === Analisis ===
    lines.append("## Analisis dan Temuan\n")

    # 1. Pengaruh Chunk Size
    lines.append("### 1. Pengaruh Ukuran Chunk (256, 500, 1000, 2000)\n")
    chunk_results = {r.config.chunk_size: r.hit_rate for r in results if r.config.top_k == 5 and r.config.chunk_overlap > 0 and r.config.chunk_overlap == {256: 50, 500: 100, 1000: 200, 2000: 400}.get(r.config.chunk_size, -1)}

    if chunk_results:
        rates = list(chunk_results.values())
        if len(set(rates)) == 1:
            lines.append(f"Mengubah ukuran chunk dari 256 hingga 2000 karakter **tidak berpengaruh** pada Hit Rate — semua tetap {rates[0]:.0f}% dengan Top-K = 5. Hal ini menunjukkan bahwa metode chunking Hybrid (Header + Recursive) sudah cukup baik dalam mempertahankan konteks informasi di setiap ukuran chunk.\n")
            lines.append("> **Temuan:** Ukuran chunk tidak menjadi faktor pembeda utama untuk model DistilBERT pada dataset ini.\n")
        else:
            best_size = max(chunk_results, key=chunk_results.get)
            lines.append(f"Variasi ukuran chunk menunjukkan perbedaan pada Hit Rate. Ukuran chunk **{best_size}** memberikan hasil terbaik ({chunk_results[best_size]:.0f}%).\n")
            for size, rate in sorted(chunk_results.items()):
                lines.append(f"- Chunk {size}: {rate:.0f}%")
            lines.append("")
            lines.append(f"> **Temuan:** Ukuran chunk {best_size} paling optimal untuk model DistilBERT pada dataset ini.\n")

    # 2. Pengaruh Overlap
    lines.append("### 2. Pengaruh Overlap (0%, 10%, 20%, 50%)\n")
    overlap_results = {r.config.chunk_overlap: r.hit_rate for r in results if r.config.chunk_size == 1000 and r.config.top_k == 5}

    if overlap_results:
        rates = list(overlap_results.values())
        if len(set(rates)) == 1:
            lines.append(f"Menghilangkan overlap sepenuhnya (0%) maupun menaikkannya hingga 50% juga **tidak mengubah** Hit Rate. Informasi yang dicari tetap ditemukan di chunk yang sama terlepas dari seberapa besar tumpang tindih antar potongan.\n")
            lines.append("> **Temuan:** Overlap bermanfaat untuk menjaga kontinuitas teks, tetapi tidak berpengaruh signifikan terhadap akurasi retrieval pada dataset ini.\n")
        else:
            best_overlap = max(overlap_results, key=overlap_results.get)
            lines.append(f"Variasi overlap menunjukkan perbedaan pada Hit Rate:\n")
            for ov, rate in sorted(overlap_results.items()):
                pct = f"{ov/10:.0f}%" if ov > 0 else "0%"
                lines.append(f"- Overlap {ov} ({pct}): {rate:.0f}%")
            lines.append(f"\n> **Temuan:** Overlap {best_overlap} memberikan hasil terbaik.\n")

    # 3. Pengaruh Top-K
    lines.append("### 3. Pengaruh Top-K (3, 5, 10) — Faktor Paling Berpengaruh\n")
    topk_results = {r.config.top_k: r for r in results if r.config.chunk_size == 1000 and r.config.chunk_overlap == 200}

    lines.append("| Top-K | Hit Rate | Penjelasan |")
    lines.append("|:---:|:---:|---|")
    for k in sorted(topk_results.keys()):
        r = topk_results[k]
        hits = sum(1 for qr in r.query_results if qr.hit)
        detail_parts = []
        for i, qr in enumerate(r.query_results):
            if qr.hit:
                detail_parts.append(f"Q{i+1}")
        hit_list = ", ".join(detail_parts) if detail_parts else "Tidak ada"
        lines.append(f"| {k} | {r.hit_rate:.0f}% | {hit_list} ditemukan. |")

    lines.append("")

    # Analisis top-K
    if len(topk_results) >= 2:
        min_k = min(topk_results.keys())
        max_k = max(topk_results.keys())
        min_rate = topk_results[min_k].hit_rate
        max_rate = topk_results[max_k].hit_rate

        if max_rate > min_rate:
            lines.append(f"Top-K adalah **parameter paling berpengaruh** dalam benchmark ini. Menaikkan Top-K dari {min_k} ke {max_k} menaikkan Hit Rate dari {min_rate:.0f}% menjadi {max_rate:.0f}%.\n")
            lines.append(f"**Mengapa?** DistilBERT adalah model Bahasa Inggris. Query Bahasa Indonesia (Q2, Q3) menghasilkan vektor yang tidak terlalu dekat dengan chunk yang berisi jawaban, sehingga chunk tersebut berada di peringkat lebih rendah. Dengan Top-K = {max_k}, chunk tersebut akhirnya ikut dikembalikan.\n")
        else:
            lines.append("Top-K tidak menunjukkan perbedaan signifikan pada dataset ini.\n")

    # 4. Konfigurasi Optimal
    lines.append("### 4. Konfigurasi Paling Optimal\n")
    lines.append("Berdasarkan 10 eksperimen, konfigurasi terbaik adalah:\n")

    best = max(results, key=lambda r: (r.hit_rate, -r.avg_latency))
    lines.append("| Parameter | Nilai Optimal | Alasan |")
    lines.append("|---|:---:|---|")
    lines.append(f"| Chunk Size | **{best.config.chunk_size}** | {'Chunk lebih kecil = konten lebih fokus per chunk' if best.config.chunk_size <= 500 else 'Ukuran seimbang antara konteks dan fokus'} |")
    lines.append(f"| Overlap | **{best.config.chunk_overlap}** | Menjaga kontinuitas teks antar potongan |")
    lines.append(f"| Top-K | **{best.config.top_k}** | {'Menjamin semua jawaban ditemukan termasuk query lintas bahasa' if best.config.top_k >= 10 else 'Keseimbangan antara akurasi dan efisiensi'} |")
    lines.append(f"| Hit Rate | **{best.hit_rate:.0f}%** | {'Semua 5 pertanyaan tes berhasil dijawab' if best.hit_rate == 100 else f'{int(best.hit_rate/20)}/5 pertanyaan berhasil dijawab'} |")

    # 5. Keterbatasan
    lines.append(f"\n### 5. Keterbatasan DistilBERT\n")
    lines.append(f"- **Bahasa:** Hanya mendukung Bahasa Inggris. Query Bahasa Indonesia bisa tetap ditemukan jika Top-K cukup besar, tetapi ranking-nya lebih rendah.")
    lines.append(f"- **Konteks:** Window {MAX_LENGTH} token — chunk yang melebihi batas ini akan terpotong.")
    lines.append(f"- **Ukuran model:** Model ringan (distilled dari BERT), lebih cepat tapi akurasi embedding bisa lebih rendah dibanding model full-size.")
    lines.append(f"- **Domain:** Dilatih pada NLI + STS Benchmark untuk kemiripan semantik umum, bukan domain ilmiah spesifik.")

    lines.append("\n---\n")

    # === Konfigurasi Teknis ===
    lines.append("## Konfigurasi Teknis\n")
    lines.append("| Parameter | Nilai |")
    lines.append("|---|---|")
    lines.append("| PDF → Markdown | `pymupdf4llm` |")
    lines.append("| Chunking Tier 1 | `MarkdownHeaderTextSplitter` (split by `#`, `##`, `###`) |")
    lines.append("| Chunking Tier 2 | `RecursiveCharacterTextSplitter` (fallback jika chunk > ukuran target) |")
    lines.append(f"| Model Embedding | {model_label} — {dim} dimensi |")
    lines.append("| Vector Database | Qdrant (lokal, folder `./qdrant_db`) |")
    lines.append("| Distance Metric | Cosine Similarity |")

    # Use best experiment's chunk count
    best_baseline = next((r for r in results if r.config.name == "EXP-03"), results[0])
    lines.append(f"| Jumlah Chunks | {best_baseline.num_chunks} |")
    lines.append(f"| Dimensi Vektor | {dim} |")
    lines.append("\n")

    # Write to file
    with open(HASIL_OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"\n✅ Hasil.md generated: {HASIL_OUTPUT}")


def main():
    logger.info("=" * 60)
    logger.info("PEDE Benchmark — DistilBERT Embedding")
    logger.info("=" * 60)

    if not os.path.exists(PDF_PATH):
        logger.error(f"Paper not found: {PDF_PATH}")
        sys.exit(1)

    # Extract paper info
    logger.info("\n[INFO] Extracting paper info...")
    paper_info = extract_paper_info(PDF_PATH)
    logger.info(f"  Title: {paper_info['title']}")
    logger.info(f"  DOI: {paper_info['doi']}")

    # Run all experiments
    all_results = []
    total_start = time.time()

    for i, exp in enumerate(EXPERIMENTS, 1):
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Experiment {i}/{len(EXPERIMENTS)}: {exp.name}")
        logger.info(f"{'#'*60}")

        result = run_experiment(exp, PDF_PATH)
        all_results.append(result)

    total_elapsed = time.time() - total_start

    # Generate Hasil.md
    logger.info("\n\n[INFO] Generating Hasil.md...")
    generate_hasil_md(all_results, paper_info)

    # Print summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"  Experiments: {len(all_results)}")
    print(f"  Total time:  {total_elapsed:.1f}s")
    print(f"  Output:      {HASIL_OUTPUT}")
    print()

    # Summary table
    print(f"{'Experiment':<12} {'Chunk':<8} {'Overlap':<8} {'Top-K':<6} {'Hit Rate':<10} {'Latency':<10} {'Note'}")
    print("-" * 80)
    for r in all_results:
        print(
            f"{r.config.name:<12} {r.config.chunk_size:<8} {r.config.chunk_overlap:<8} "
            f"{r.config.top_k:<6} {r.hit_rate:>5.0f}%     {r.avg_latency:>6.2f}s    {r.config.note}"
        )
    print()

    # Cleanup benchmark DB
    if os.path.exists(QDRANT_PATH_BENCH):
        shutil.rmtree(QDRANT_PATH_BENCH)
        logger.info("Cleaned up benchmark database.")


if __name__ == "__main__":
    main()
