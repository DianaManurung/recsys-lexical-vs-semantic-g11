# Recommender Systems: Lexical vs. Semantic

Proyek ini membandingkan pendekatan rekomendasi produk berbasis interaksi pengguna dan konten item. Implementasi yang tersedia mencakup baseline popularitas, **biased matrix factorization**, dan **TF-IDF content-based filtering**. Model TF-IDF menggunakan representasi teks leksikal item; matrix factorization menjadi baseline kolaboratif.

## Model

- **Popularity** — mengurutkan item warm berdasarkan jumlah rating positif pada data train.
- **Matrix Factorization (MF)** — memprediksi rating dengan global mean, bias pengguna/item, dan faktor laten yang dilatih menggunakan SGD.
- **TF-IDF Content-Based Filtering (CBF)** — membentuk profil pengguna dari rata-rata vektor TF-IDF item yang diberi rating positif, lalu mengurutkan kandidat berdasarkan cosine similarity.

Semua model merekomendasikan item dari daftar kandidat warm dan mengeluarkan item yang sudah dilihat pengguna pada data train. MF dan CBF menggunakan popularitas sebagai fallback untuk pengguna tanpa profil yang dapat digunakan; CBF juga mengisi slot rekomendasi yang skornya nol dengan urutan popularitas.

## Struktur proyek

```text
.
├── data/                       # Dataset lokal (tidak dilacak Git)
├── notebooks/
│   ├── popularity_demo.ipynb
│   ├── collaborative_mf_demo.ipynb
│   ├── content_based_tfidf_demo.ipynb
│   └── sanity_check_topn.ipynb
├── outputs/
│   ├── configs/                # Konfigurasi run
│   ├── figures/                # Visualisasi hasil
│   ├── logs/                   # Metadata run
│   ├── metrics/                # Metrik evaluasi
│   └── samples/                # Contoh rekomendasi
├── results/                    # Log sanity check
└── src/
    ├── content_based.py
    ├── data_loader.py
    ├── matrix_factorization.py
    └── popularity.py
```

## Persiapan

Gunakan Python 3.11 atau lebih baru. Buat virtual environment dan pasang dependensi:

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

Pasang library yang digunakan:

```bash
python -m pip install numpy pandas scipy scikit-learn jupyter
```

> Repository ini belum menyediakan `requirements.txt`; perintah di atas memasang dependensi yang diperlukan oleh implementasi dan notebook.

## Data

Letakkan file dataset berikut di folder `data/` pada root repository:

| File | Kolom yang digunakan | Keterangan |
|---|---|---|
| `interactions_train.csv` | `user_id`, `item_id`, `rating`, `timestamp` | Interaksi untuk melatih model dan menentukan item yang sudah dilihat. |
| `interactions_validation.csv` | `user_id`, `item_id`, `rating`, `timestamp` | Interaksi untuk evaluasi. |
| `candidate_items_warm.csv` | `item_id` | Kandidat item warm yang boleh direkomendasikan. |
| `items.csv` | `item_id`, `title`, `content_text_basic` | Metadata item; kolom teks dipakai oleh CBF. |

Data di folder `data/` tidak dilacak Git. Pastikan file tersedia secara lokal sebelum menjalankan notebook. ID item pada file interaksi, kandidat, dan metadata harus menggunakan format yang konsisten.

## Menjalankan notebook

Dari root repository, jalankan:

```bash
jupyter notebook
```

Buka dan jalankan notebook sesuai kebutuhan:

1. `notebooks/popularity_demo.ipynb` — contoh rekomendasi popularitas.
2. `notebooks/collaborative_mf_demo.ipynb` — pelatihan dan evaluasi matrix factorization.
3. `notebooks/content_based_tfidf_demo.ipynb` — ekstraksi teks item, TF-IDF, dan evaluasi CBF.
4. `notebooks/sanity_check_topn.ipynb` — pemeriksaan contoh Top-10 untuk beberapa model pada subset pengguna.

Notebook sanity check ditujukan untuk memeriksa keluaran rekomendasi pada subset, bukan menggantikan evaluasi penuh.

## Menggunakan model dari Python

Jalankan contoh ini dari root repository setelah data tersedia:

```python
from src.data_loader import load_data
from src.popularity import PopularityRecommender
from src.matrix_factorization import MatrixFactorizationRecommender
from src.content_based import TfidfContentRecommender

train, validation, candidates, seen = load_data()

# Pilih salah satu model.
model = PopularityRecommender().fit(train, candidates)
# model = MatrixFactorizationRecommender().fit(train, candidates)
# model = TfidfContentRecommender().fit(train, candidates)

user_id = train["user_id"].iloc[0]
recommendations = model.recommend(user_id, seen, k=10)
print(recommendations)
```

Untuk CBF, `fit()` secara default membaca `data/items.csv`, kolom `content_text_basic`. `validation` dapat digunakan untuk evaluasi; data validasi tidak digunakan saat model dilatih.

## Evaluasi

Item dianggap relevan jika rating validasi **≥ 4**, termasuk dalam kandidat warm, dan penggunanya muncul di data train. Metrik evaluasi yang disediakan notebook meliputi **Recall@10**, **NDCG@10**, dan **HitRate@10**.

Sebagai contoh, hasil run TF-IDF yang tersimpan di `outputs/metrics/tfidf_cbf_val.json` mencatat:

| Metrik | Nilai |
|---|---:|
| Recall@10 | 0.0127 |
| NDCG@10 | 0.0074 |
| HitRate@10 | 0.0218 |
| Pengguna evaluasi | 11.850 |

Nilai tersebut merupakan hasil run yang tersimpan, bukan jaminan hasil akan sama untuk konfigurasi atau dataset lain.
