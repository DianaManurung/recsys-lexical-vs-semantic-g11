
# Recommender Systems: Lexical vs. Semantic

Proyek ini membandingkan pendekatan rekomendasi produk berbasis representasi leksikal dan semantik menggunakan dataset Amazon Reviews 2023 kategori Musical Instruments. Fokus utama proyek adalah membandingkan **TF-IDF Content-Based Filtering** dan **Semantic Embedding Content-Based Filtering** dalam menghasilkan rekomendasi Top-10.

Selain kedua model tersebut, proyek ini juga menyediakan baseline popularitas dan **biased matrix factorization** sebagai pembanding.

## Model

- **Popularity** — mengurutkan item berdasarkan jumlah rating positif (rating >= 4) pada data train.
- **Matrix Factorization (MF)** — memprediksi rating menggunakan global mean, bias pengguna dan item, serta faktor laten yang dilatih menggunakan SGD.
- **TF-IDF Content-Based Filtering (CBF)** — membentuk profil pengguna dari rata-rata vektor TF-IDF item yang diberi rating positif, kemudian mengurutkan kandidat berdasarkan cosine similarity.
- **Semantic Embedding CBF** — menggunakan Sentence Transformer `all-MiniLM-L6-v2` untuk mengubah teks item menjadi embedding dan menghitung kemiripan dengan profil pengguna.

Model content-based menggunakan kolom `content_text_basic` dari metadata item. Profil pengguna dibentuk dari item yang memperoleh rating >= 4 pada data train. Pengguna tanpa profil yang dapat digunakan menggunakan popularitas sebagai fallback.

Semua model menggunakan kandidat item warm dan mengeluarkan item yang sudah pernah dilihat pengguna pada data train.

## Eksperimen

Proyek ini memiliki tujuh rancangan eksperimen:

| ID | Eksperimen | Keterangan |
|---|---|---|
| E1 | Popularity | Baseline popularitas. |
| E2 | Biased Matrix Factorization | Baseline collaborative filtering. |
| E3 | TF-IDF CBF | Representasi item secara leksikal. |
| E4 | Semantic Embedding CBF | Representasi item secara semantik. |
| E5 | TF-IDF + Popularity | Rancangan hybrid leksikal dan popularitas. |
| E6 | Embedding + Popularity | Rancangan hybrid semantik dan popularitas. |
| E7 | Perbandingan berdasarkan aktivitas pengguna | Membandingkan E3 dan E4 pada kelompok aktivitas pengguna rendah, sedang, dan tinggi. |

Tidak semua eksperimen sudah dijalankan. Baseline TF-IDF dan MiniLM sudah memiliki hasil evaluasi, sedangkan analisis kelompok pengguna E7 juga sudah dijalankan. Implementasi pencarian untuk MF dan kedua hybrid belum tersedia pada runner saat ini.

Rincian konfigurasi dan status setiap eksperimen dapat dilihat pada [`docs/experiment_matrix.md`](docs/experiment_matrix.md).

## Struktur proyek

```text
.
├── configs/                    # Konfigurasi eksperimen
├── data/                       # Dataset lokal (tidak dilacak Git)
├── docs/
│   ├── experiment_matrix.md
│   └── experiment_matrix_overleaf.tex
├── notebooks/
│   ├── popularity_demo.ipynb
│   ├── collaborative_mf_demo.ipynb
│   ├── content_based_tfidf_demo.ipynb
│   ├── semantic_cbf_demo.ipynb
│   ├── protocol_demo.ipynb
│   └── ...
├── outputs/
│   ├── configs/                 # Konfigurasi run yang tersimpan
│   ├── figures/                 # Visualisasi hasil
│   ├── logs/                    # Log eksperimen dan protokol
│   ├── metrics/                 # Metrik evaluasi
│   └── samples/                 # Contoh rekomendasi
├── results/                     # Log sanity check
├── src/                         # Implementasi model dan evaluasi
├── README.md
└── requirements.txt
```

Embedding yang dihasilkan disimpan secara lokal di `outputs/embeddings/` dan tidak dilacak Git.

## Persiapan

Gunakan Python 3.11 atau lebih baru. Buat virtual environment dari root repository:

```bash
python -m venv .venv
```

Aktifkan environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Pasang dependensi:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data

Letakkan file dataset berikut di folder `data/` pada root repository:

| File | Kolom yang digunakan | Keterangan |
|---|---|---|
| `interactions_train.csv` | `user_id`, `item_id`, `rating`, `timestamp` | Data untuk melatih model dan membentuk profil pengguna. |
| `interactions_validation.csv` | `user_id`, `item_id`, `rating`, `timestamp` | Data untuk evaluasi dan pemilihan konfigurasi. |
| `candidate_items_warm.csv` | `item_id` | Daftar item yang dapat direkomendasikan. |
| `items.csv` | `item_id`, `content_text_basic` | Metadata item dan teks yang digunakan oleh model content-based. |

Data pada folder `data/` tidak dilacak Git. Pastikan file tersedia secara lokal sebelum menjalankan notebook atau script.

## Menjalankan proyek

Jalankan perintah dari root repository.

Untuk membuka notebook:

```bash
jupyter notebook
```

Notebook yang tersedia antara lain:

1. `notebooks/popularity_demo.ipynb` — baseline popularitas.
2. `notebooks/collaborative_mf_demo.ipynb` — matrix factorization.
3. `notebooks/content_based_tfidf_demo.ipynb` — TF-IDF content-based filtering.
4. `notebooks/semantic_cbf_demo.ipynb` — semantic embedding-based filtering.
5. `notebooks/protocol_demo.ipynb` — pemeriksaan protokol evaluasi.
6. `notebooks/evaluate_sanity_check.ipynb` — pemeriksaan metrik evaluasi.
7. `notebooks/sanity_check_topn.ipynb` — pemeriksaan rekomendasi Top-10.

Untuk membuat embedding item menggunakan Sentence Transformer:

```bash
python -m src.semantic_embedding
```

Untuk memeriksa jumlah data dan protokol evaluasi:

```bash
python -m src.protocol
```

Untuk menjalankan pemeriksaan leakage:

```bash
python -m src.leakage_check
```

Untuk melihat kombinasi konfigurasi TF-IDF yang tersedia:

```bash
python -m src.run_grid configs/tfidf_search.json --list
```

Analisis kelompok aktivitas pengguna untuk E7 dapat dijalankan dengan:

```bash
python -m src.user_groups
```

## Evaluasi

Evaluasi menggunakan protokol warm-start dengan pembagian data temporal:

- Data train: **427.957 interaksi**.
- Data validation: **41.284 interaksi**.
- Data test: **42.595 interaksi**.
- Kandidat item warm: **22.753 item**.
- Pengguna yang dapat dievaluasi: **11.850 pengguna**.

Item dianggap relevan jika memperoleh rating **>= 4** pada data validation, termasuk dalam kandidat warm, dan belum pernah dilihat pengguna pada data train.

Metrik evaluasi yang digunakan adalah **Recall@10**, **NDCG@10**, dan **HitRate@10**. Data test disimpan untuk evaluasi akhir setelah konfigurasi model ditetapkan.

Hasil baseline yang tersimpan pada repository:

| Metrik | TF-IDF CBF | Semantic CBF (MiniLM) |
|---|---:|---:|
| Recall@10 | 0.012722 | 0.008860 |
| NDCG@10 | 0.007452 | 0.005361 |
| HitRate@10 | 0.021772 | 0.015696 |
| Pengguna evaluasi | 11.850 | 11.850 |

Nilai tersebut merupakan hasil evaluasi pada data validation, bukan hasil evaluasi akhir pada data test.

Analisis E7 membagi pengguna berdasarkan jumlah interaksi pada data train:
- Rendah: <= 5 interaksi.
- Sedang: 6–8 interaksi.
- Tinggi: >= 9 interaksi.

Hasil evaluasi dan log eksperimen tersimpan pada folder `outputs/metrics/` dan `outputs/logs/`.
