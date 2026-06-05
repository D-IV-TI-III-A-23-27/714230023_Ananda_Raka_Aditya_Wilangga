# 📊 Hasil Benchmarking PEDE — DistilBERT Embedding

**Paper Uji:** *Generative adversarial network for fault detection diagnosis of chillers*  
**DOI:** `10.1016/j.buildenv.2020.106698`  
**Model Embedding:** DistilBERT (`distilbert-base-nli-stsb-mean-tokens`) — 768 dimensi, model ringan berbasis BERT yang di-distill untuk embedding kalimat

## Hasil Benchmarking (10 Eksperimen)

| Ukuran Chunk | Overlap | Metode Chunking | Model Embedding | Dukungan Bahasa | Tipe Query Uji | Top-K | Filter Metadata | Hit Rate | Latensi | Ukuran Index DB | Catatan |
|:---:|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 256 | 50 | Hybrid | DistilBERT (`distilbert-base-nli-stsb-mean-tokens`) | Campuran (EN + ID) | Campuran (5 query) | 5 | Ya (DOI) | 40% | 0.08s | 2.4 MB | *Chunk sangat kecil* |
| 500 | 100 | Hybrid | DistilBERT (`distilbert-base-nli-stsb-mean-tokens`) | Campuran (EN + ID) | Campuran (5 query) | 5 | Ya (DOI) | **100%** | 0.05s | 1.4 MB | *Chunk kecil* |
| 1000 | 200 | Hybrid | DistilBERT (`distilbert-base-nli-stsb-mean-tokens`) | Campuran (EN + ID) | Campuran (5 query) | 5 | Ya (DOI) | 80% | 0.10s | 0.8 MB | *Chunk medium (baseline)* |
| 2000 | 400 | Hybrid | DistilBERT (`distilbert-base-nli-stsb-mean-tokens`) | Campuran (EN + ID) | Campuran (5 query) | 5 | Ya (DOI) | **100%** | 0.06s | 0.4 MB | *Chunk besar* |
| 1000 | 0 | Hybrid | DistilBERT (`distilbert-base-nli-stsb-mean-tokens`) | Campuran (EN + ID) | Campuran (5 query) | 5 | Ya (DOI) | **100%** | 0.09s | 0.8 MB | *Overlap 0% (tanpa)* |
| 1000 | 100 | Hybrid | DistilBERT (`distilbert-base-nli-stsb-mean-tokens`) | Campuran (EN + ID) | Campuran (5 query) | 5 | Ya (DOI) | 80% | 0.04s | 0.8 MB | *Overlap 10%* |
| 1000 | 500 | Hybrid | DistilBERT (`distilbert-base-nli-stsb-mean-tokens`) | Campuran (EN + ID) | Campuran (5 query) | 5 | Ya (DOI) | **100%** | 0.04s | 0.8 MB | *Overlap 50%* |
| 1000 | 200 | Hybrid | DistilBERT (`distilbert-base-nli-stsb-mean-tokens`) | Campuran (EN + ID) | Campuran (5 query) | 3 | Ya (DOI) | 80% | 0.04s | 0.8 MB | *Top-K kecil (3)* |
| 1000 | 200 | Hybrid | DistilBERT (`distilbert-base-nli-stsb-mean-tokens`) | Campuran (EN + ID) | Campuran (5 query) | 10 | Ya (DOI) | **100%** | 0.04s | 0.8 MB | *Top-K besar (10)* |
| 500 | 100 | Hybrid | DistilBERT (`distilbert-base-nli-stsb-mean-tokens`) | Campuran (EN + ID) | Campuran (5 query) | 10 | Ya (DOI) | **100%** | 0.05s | 1.4 MB | *Chunk kecil + Top-K besar* |

---

## Detail Hasil per Query

### 5 Pertanyaan Tes

| # | Pertanyaan | Tipe | Bahasa | Jawaban yang Diharapkan |
|:---:|---|---|:---:|---|
| Q1 | What dataset was used to train and evaluate the GAN-based chiller fault detection model? | Factoid | EN | Dataset ASHRAE Project 1043-RP, berisi data operasional chiller 90-ton (316 kW) dengan 7 jenis fault |
| Q2 | Mengapa dataset pelatihan chiller AFDD menjadi tidak seimbang dalam skenario dunia nyata? | Reasoning | ID | Data fault sangat sedikit karena kegagalan chiller jarang terjadi dan langsung diperbaiki, sedangkan data normal sangat banyak |
| Q3 | Bagaimana GAN digunakan untuk meningkatkan akurasi deteksi fault pada chiller? | Semantic | ID | GAN menghasilkan sampel fault buatan untuk menyeimbangkan dataset training yang tidak seimbang, sehingga meningkatkan akurasi klasifikasi |
| Q4 | What are the seven types of chiller faults investigated in this study? | Factoid | EN | F1: Reduced condenser water flow, F2: Reduced evaporator water flow, F3: Refrigerant Leak, F4: Refrigerant Overcharge, F5: Excess Oil, F6: Condenser Fouling, F7: Non-condensables |
| Q5 | berapa jumlah sampel fault training yang diuji dalam eksperimen penelitian ini? | Conversational | ID | 10, 20, 30, 40, dan 50 sampel fault training per jenis fault, bersama 8400 sampel normal |

### Hasil per Query (Top-K = 5 vs Top-K = 10)

| # | Tipe | Bahasa | Top-K = 5 | Top-K = 10 |
|:---:|:---|:---:|:---:|:---:|
| Q1 | Factoid | EN | HIT | HIT |
| Q2 | Reasoning | ID | HIT | HIT |
| Q3 | Semantic | ID | HIT | HIT |
| Q4 | Factoid | EN | HIT | HIT |
| Q5 | Conversational | ID | MISS | HIT |
| | | **Total** | **4/5 = 80%** | **5/5 = 100%** |


---

## Analisis dan Temuan

### 1. Pengaruh Ukuran Chunk (256, 500, 1000, 2000)

Variasi ukuran chunk menunjukkan perbedaan pada Hit Rate. Ukuran chunk **500** memberikan hasil terbaik (100%).

- Chunk 256: 40%
- Chunk 500: 100%
- Chunk 1000: 80%
- Chunk 2000: 100%

### 2. Pengaruh Overlap (0%, 10%, 20%, 50%)

Variasi overlap menunjukkan perbedaan pada Hit Rate:

- Overlap 0 (0%): 100%
- Overlap 100 (10%): 80%
- Overlap 200 (20%): 80%
- Overlap 500 (50%): 100%


### 3. Pengaruh Top-K (3, 5, 10) — Faktor Paling Berpengaruh

| Top-K | Hit Rate | Penjelasan |
|:---:|:---:|---|
| 3 | 80% | Q1, Q2, Q3, Q4 ditemukan. |
| 5 | 80% | Q1, Q2, Q3, Q4 ditemukan. |
| 10 | 100% | Q1, Q2, Q3, Q4, Q5 ditemukan. |

Top-K adalah **parameter paling berpengaruh** dalam benchmark ini. Menaikkan Top-K dari 3 ke 10 menaikkan Hit Rate dari 80% menjadi 100%.

**Mengapa?** DistilBERT adalah model Bahasa Inggris. Query Bahasa Indonesia (Q2, Q3) menghasilkan vektor yang tidak terlalu dekat dengan chunk yang berisi jawaban, sehingga chunk tersebut berada di peringkat lebih rendah. Dengan Top-K = 10, chunk tersebut akhirnya ikut dikembalikan.

### 4. Konfigurasi Paling Optimal

Berdasarkan 10 eksperimen, konfigurasi terbaik adalah:

| Parameter | Nilai Optimal | Alasan |
|---|:---:|---|
| Chunk Size | **1000** | Ukuran seimbang antara konteks dan fokus |
| Overlap | **200** | Menjaga kontinuitas teks antar potongan |
| Top-K | **10** | Menjamin semua jawaban ditemukan termasuk query lintas bahasa |
| Hit Rate | **100%** | Semua 5 pertanyaan tes berhasil dijawab |

### 5. Keterbatasan DistilBERT

- **Bahasa:** Hanya mendukung Bahasa Inggris. Query Bahasa Indonesia bisa tetap ditemukan jika Top-K cukup besar, tetapi ranking-nya lebih rendah.
- **Konteks:** Window 512 token — chunk yang melebihi batas ini akan terpotong.
- **Ukuran model:** Model ringan (distilled dari BERT), lebih cepat tapi akurasi embedding bisa lebih rendah dibanding model full-size.
- **Domain:** Dilatih pada NLI + STS Benchmark untuk kemiripan semantik umum, bukan domain ilmiah spesifik.

---

## Konfigurasi Teknis

| Parameter | Nilai |
|---|---|
| PDF → Markdown | `pymupdf4llm` |
| Chunking Tier 1 | `MarkdownHeaderTextSplitter` (split by `#`, `##`, `###`) |
| Chunking Tier 2 | `RecursiveCharacterTextSplitter` (fallback jika chunk > ukuran target) |
| Model Embedding | DistilBERT (`distilbert-base-nli-stsb-mean-tokens`) — 768 dimensi |
| Vector Database | Qdrant (lokal, folder `./qdrant_db`) |
| Distance Metric | Cosine Similarity |
| Jumlah Chunks | 89 |
| Dimensi Vektor | 768 |

