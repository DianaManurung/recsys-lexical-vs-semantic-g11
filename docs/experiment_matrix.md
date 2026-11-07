
# Experiment Matrix — Track E (G11)

## 1. Tujuan Eksperimen

Experiment Matrix ini mendokumentasikan tujuh eksperimen (E1–E7) untuk penelitian *Lexical vs. Semantic Item Representations for Content-Based and Hybrid Top-N Recommendation*.

Perbandingan utama penelitian berfokus pada representasi leksikal menggunakan TF-IDF dan representasi semantik menggunakan Sentence Embedding.

Penelitian memiliki dua pertanyaan utama:

**RQ1:** Do sentence embeddings give better Top-10 recommendations than TF-IDF, measured by Recall@10 and NDCG@10, when both use the same item text?

**RQ2:** Is the difference between sentence embeddings and TF-IDF larger for users with short interaction histories?

E3 dan E4 merupakan eksperimen utama untuk menjawab RQ1. E7 menjawab RQ2. E1 dan E2 merupakan baseline pendukung, sedangkan E5 dan E6 merupakan eksperimen hybrid tambahan yang bersifat eksploratif.

## 2. Protokol Evaluasi Bersama

Semua eksperimen menggunakan protokol evaluasi warm-start yang sama.

| Komponen | Nilai |
|---|---:|
| Training interactions | 427,957 |
| Validation interactions | 41,284 |
| Test interactions | 42,595 |
| Warm-start candidate universe | 22,753 items |
| Evaluable users | 11,850 |
| Relevant validation user-item pairs | 24,707 |
| Positive interaction threshold | Rating >= 4 |
| Recommendation cutoff | Top-10 |
| Primary metrics | Recall@10, NDCG@10 |
| Supporting metric | HitRate@10 |
| Random seed | 42 |

### 2.1 Temporal Split

Proyek menggunakan official temporal split tanpa random reshuffling atau pembuatan split baru.

- Training digunakan untuk fitting model dan membangun profil pengguna.
- Validation digunakan untuk evaluasi, hyperparameter tuning, dan model selection.
- Test disimpan untuk satu evaluasi final setelah konfigurasi model dibekukan.

Test tidak digunakan untuk tuning maupun pemilihan model selama CP3.

### 2.2 Candidate Set

Candidate universe terdiri dari 22,753 item yang muncul pada training dan memiliki metadata yang dapat digunakan.

Untuk setiap pengguna, seluruh item yang pernah diinteraksikan pada training dikeluarkan dari candidate set pengguna tersebut, tanpa memperhatikan nilai rating.

Dengan demikian, semua model menghasilkan ranking dari candidate universe yang sama setelah item yang sudah dilihat pengguna dikeluarkan.

### 2.3 Evaluable Users dan Relevance

Pengguna termasuk evaluable users apabila memiliki riwayat interaksi pada training dan setidaknya satu item relevan pada validation.

Item dianggap relevan apabila:
- Rating validation >= 4.
- Item termasuk warm-start candidate universe.
- Item belum pernah muncul dalam riwayat training pengguna.

Protokol ini menghasilkan 11,850 evaluable users dan 24,707 relevant user-item pairs.

Tidak ada threshold minimum interaksi tambahan untuk populasi evaluasi utama.

### 2.4 Evaluation Metrics

Metrik utama:
- Recall@10
- NDCG@10

Metrik pendukung:
- HitRate@10

Semua model menggunakan populasi evaluasi dan cutoff Top-10 yang sama.

## 3. Experiment Matrix

| ID | Model / Variant | Search Space / Configuration | Evaluation Metrics | Purpose / Research Question |
|---|---|---|---|---|
| E1 | Popularity | Fixed baseline; menghitung jumlah interaksi positif pada training (rating >= 4). Tanpa hyperparameter tuning. | Recall@10, NDCG@10, HitRate@10 | Baseline non-personalized pendukung. |
| E2 | Biased Matrix Factorization (MF) | 16 kombinasi: factors {32, 64}; regularization {0.02, 0.05}; learning rate {0.005, 0.01}; epochs {10, 20}. Item-bias weight = 0; seed = 42; K = 10. | Recall@10, NDCG@10, HitRate@10 | Baseline collaborative filtering pendukung. |
| E3 | TF-IDF Content-Based Filtering | 12 kombinasi: max_features {10,000, 20,000, unlimited}; ngram_range {(1,1), (1,2)}; min_df {2, 5}. Fixed text field `content_text_basic`, English stop words, sublinear TF, max_df = 0.8. | Recall@10, NDCG@10, HitRate@10 | Mengevaluasi representasi leksikal untuk RQ1. |
| E4 | Semantic Embedding Content-Based Filtering | Dua encoder yang dikonfigurasi: `sentence-transformers/all-MiniLM-L6-v2` dan `sentence-transformers/all-mpnet-base-v2`. Maximum sequence length = 256; batch size = 32; CPU. | Recall@10, NDCG@10, HitRate@10 | Mengevaluasi representasi semantik untuk RQ1. |
| E5 | TF-IDF + Popularity Hybrid | Content weight alpha in {0.25, 0.50, 0.75}; popularity weight = 1 - alpha. Kedua skor dinormalisasi dengan min-max secara terpisah untuk setiap pengguna. | Recall@10, NDCG@10, HitRate@10 | Eksperimen eksploratif penggabungan relevansi konten leksikal dan popularitas. |
| E6 | Embedding + Popularity Hybrid | Semantic content weight alpha in {0.25, 0.50, 0.75}; popularity weight = 1 - alpha. Kedua skor dinormalisasi dengan min-max secara terpisah untuk setiap pengguna. | Recall@10, NDCG@10, HitRate@10 | Eksperimen eksploratif penggabungan relevansi konten semantik dan popularitas. |
| E7 | TF-IDF vs. Embedding by User Activity | Membandingkan E3 dan E4 pada kelompok low (<= 5), medium (6–8), dan high (>= 9) berdasarkan seluruh interaksi training. | Recall@10, NDCG@10, HitRate@10 per kelompok | Membandingkan representasi leksikal dan semantik berdasarkan aktivitas pengguna untuk RQ2. |

E5 dan E6 menggabungkan content-based relevance dengan popularity, bukan collaborative filtering dengan content-based filtering. Keduanya merupakan eksperimen eksploratif dan bukan bagian dari pertanyaan utama RQ1 maupun RQ2.

## 4. Aturan Skor Hybrid

Untuk E5 dan E6, skor hybrid didefinisikan sebagai:

\[
s(u,i)=\alpha s_{\mathrm{content}}(u,i)
+(1-\alpha)s_{\mathrm{pop}}(i)
\]

Keterangan:
- `s_content` adalah skor TF-IDF untuk E5 atau cosine similarity semantic embedding untuk E6.
- `s_pop` adalah skor popularitas berdasarkan jumlah interaksi positif pada training.
- `alpha` adalah bobot content-based.
- `1-alpha` adalah bobot popularity.

Skor content dan popularity dinormalisasi secara terpisah menggunakan min-max normalization untuk setiap pengguna pada kandidat warm yang belum dilihat.

Jika pengguna tidak memiliki profil konten yang dapat digunakan, skor content ditetapkan menjadi nol dan komponen popularity tetap digunakan.

E5 dan E6 masih berupa rancangan eksperimen. Implementasi recommender hybrid dan hasil evaluasi lengkap belum tersedia pada repository snapshot yang diperiksa.

## 5. Kesetaraan Perbandingan E3 dan E4

Perbandingan TF-IDF dan Semantic Embedding menggunakan:

- Field teks item yang sama, yaitu `content_text_basic`.
- Candidate universe warm-start yang sama.
- Aturan pembentukan profil pengguna yang sama, berdasarkan item training dengan rating >= 4.
- Cosine similarity untuk penilaian kandidat.
- Pengecualian item yang telah dilihat pengguna pada training.
- Populasi evaluasi dan cutoff Top-10 yang sama.

Terdapat perbedaan pada pemrosesan teks.

TF-IDF memproses teks item yang tersedia tanpa batas pemotongan urutan 256 token. Sebaliknya, model embedding menggunakan maximum sequence length 256 token.

Pada run MiniLM, sebanyak 8,276 dari 22,753 teks item mengalami truncation atau sekitar 36.4% dari candidate-item texts. Sebanyak 1,772,380 token tercatat terpotong.

Perbedaan ini perlu dipertimbangkan ketika menginterpretasikan hasil perbandingan representasi leksikal dan semantik.

Encoder semantic yang dikonfigurasi:
- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`

Baseline MiniLM memiliki hasil validation yang tersimpan. Eksekusi MPNet perlu diverifikasi sebelum dinyatakan selesai.

## 6. Popularity Fallback

Model TF-IDF dan Semantic Embedding menggunakan popularitas training sebagai fallback ketika pengguna tidak memiliki profil positif yang dapat digunakan.

Popularity juga digunakan untuk mengisi slot rekomendasi ketika jumlah kandidat dengan skor similarity positif kurang dari sepuluh.

Berdasarkan log run yang tersedia:

| Model | Evaluable Users | Users Without Profile | Percentage | Partial Popularity Fill |
|---|---:|---:|---:|---|
| TF-IDF | 11,850 | 191 | 1.61% | 0 users recorded |
| Semantic MiniLM | 11,850 | 191 | 1.61% | Not separately recorded |

Persentase pengguna tanpa profil dihitung sebagai:

\[
\frac{191}{11,850}\times100\%=1.61\%
\]

Log TF-IDF mencatat nol pengguna yang memerlukan partial popularity filling.

Log Semantic MiniLM mencatat 191 pengguna tanpa profil, tetapi tidak mencatat jumlah pengguna maupun slot rekomendasi yang diisi melalui partial popularity fallback. Oleh karena itu, nilai tersebut belum dapat dilaporkan.

Bukti yang tersedia:
- `outputs/logs/tfidf_cbf_run.json`
- `outputs/logs/semantic_cbf_run.json`
- `outputs/configs/tfidf_cbf.json`
- `outputs/configs/semantic_cbf.json`

## 7. Analisis Aktivitas Pengguna E7

Aktivitas pengguna dihitung berdasarkan jumlah seluruh interaksi per pengguna pada training, bukan hanya interaksi positif.

Batas kelompok didasarkan pada distribusi interaksi training dengan median enam interaksi per pengguna.

- Low activity: maksimal 5 interaksi, berada di bawah median.
- Medium activity: 6–8 interaksi, berada di sekitar median.
- High activity: minimal 9 interaksi, berada di atas median.

Hasil validation berdasarkan 11,850 evaluable users:

| Kelompok | Training Interactions | Evaluable Users | Percentage |
|---|---:|---:|---:|
| Low | <= 5 | 6,295 | 53.1% |
| Medium | 6–8 | 2,365 | 20.0% |
| High | >= 9 | 3,190 | 26.9% |
| **Total** | - | **11,850** | **100%** |

Ukuran kelompok tidak dimaksudkan sama besar. Pembagian ini mengikuti distribusi aktivitas pengguna pada data training.

Pengelompokan digunakan hanya untuk analisis RQ2 dan tidak mengubah populasi evaluasi utama.

Hasil kelompok tersedia pada `outputs/metrics/user_groups_val.json`.

## 8. Uncertainty Analysis

Perbandingan akhir antara TF-IDF dan semantic embedding direncanakan menggunakan paired user-level bootstrap confidence interval berdasarkan skor NDCG@10 per pengguna.

Selisih skor untuk setiap pengguna didefinisikan sebagai:

\[
\Delta_u =
\mathrm{NDCG@10}_{\mathrm{Embedding},u}
-
\mathrm{NDCG@10}_{\mathrm{TF-IDF},u}
\]

Pengguna yang sama digunakan pada kedua metode untuk mempertahankan desain perbandingan berpasangan.

Rancangan analisis:
- Bootstrap resampling: 10,000 kali.
- Confidence level: 95%.
- Random seed: 42.
- Interval: percentile bootstrap confidence interval.

Fungsi `evaluate_per_user` pada `src/evaluate.py` menyediakan metrik per pengguna yang diperlukan untuk analisis ini.

Bootstrap confidence interval merupakan rancangan analisis lanjutan dan belum menjadi hasil evaluasi final pada repository snapshot saat CP3. Analisis dilakukan setelah konfigurasi model dibekukan dan tidak digunakan untuk hyperparameter tuning atau model selection.

Test set tetap disimpan untuk evaluasi final sesuai jadwal proyek.

## 9. Reproducibility Settings

Random seed yang ditetapkan adalah 42 untuk proses yang melibatkan randomness.

Model selection dilakukan menggunakan validation NDCG@10 tertinggi. Jika terdapat nilai NDCG@10 yang sama, konfigurasi dengan Recall@10 lebih tinggi dipilih. Jika masih sama, indeks kombinasi grid yang lebih rendah dipilih.

### 9.1 Lokasi Konfigurasi dan Log

| Lokasi | Fungsi |
|---|---|
| `configs/` | Menyimpan konfigurasi awal dan search space eksperimen. |
| `outputs/configs/` | Menyimpan snapshot konfigurasi run individual. |
| `outputs/logs/experiment_log.csv` | Menyimpan ringkasan hasil grid run. |
| `outputs/logs/` | Menyimpan log individual dan pemeriksaan protokol. |
| `outputs/metrics/` | Menyimpan hasil evaluasi validation dan analisis kelompok pengguna. |

### 9.2 Penamaan Run

Format penamaan run yang ditetapkan pada `configs/reproducibility.json` adalah:

`<model>_<UTC YYYYMMDD-HHMMSS>_<combination index>_s<seed>`

Contoh format yang direncanakan:

- `tfidf_cbf_20261107-103228_08_s42`
- `embedding_cbf_20261031-110440_00_s42`
- `matrix_factorization_20261107-110000_03_s42`

Namun, run ID yang dihasilkan oleh `src/run_grid.py` saat ini belum menambahkan suffix seed. Contoh ID aktual yang tercatat:

- `tfidf_cbf_20261107-103228_08`
- `tfidf_cbf_20261107-103327_00`

Dengan demikian, format suffix seed masih menjadi aturan reproducibility yang perlu distandardisasi pada runner.

### 9.3 Pencatatan Eksperimen

Setiap run perlu dapat ditelusuri berdasarkan:
- Parameter dan konfigurasi yang digunakan.
- File search configuration dan combination index.
- Random seed.
- Git commit dan status working tree.
- UTC timestamp.
- Fingerprint SHA-256 data input.
- Versi Python dan library.
- Metrik validation.
- Jumlah evaluable users.
- Waktu eksekusi.

Grid log saat ini mencatat run ID, timestamp, model, file konfigurasi, combination index, parameter, Git commit, working-tree state, metrik validation, jumlah pengguna, dan elapsed time.

Seed, SHA-256 fingerprint, dan detail environment belum seluruhnya tersedia sebagai kolom tersendiri di CSV grid log. Informasi tambahan tersedia pada konfigurasi dan log individual ketika direkam.

### 9.4 Estimasi Runtime MF

Catatan awal tim menunjukkan satu kali fitting baseline Matrix Factorization membutuhkan sekitar lima menit.

Dengan 16 kombinasi konfigurasi, estimasi awal fitting secara berurutan adalah:

\[
16 \times 5 \text{ menit} \approx 80 \text{ menit}
\]

Angka tersebut merupakan estimasi perencanaan, bukan hasil pengukuran full-grid. Waktu aktual dapat berubah berdasarkan perangkat keras, lingkungan software, dan overhead evaluasi.

## 10. Status Implementasi dan Bukti Eksperimen

Tabel ini membedakan desain eksperimen dari implementasi serta hasil yang telah memiliki bukti.

| ID | Bukti pada Repository | Status |
|---|---|---|
| E1 | `src/popularity.py`, `notebooks/popularity_demo.ipynb` | Implementasi baseline tersedia. |
| E2 | `src/matrix_factorization.py`, `notebooks/collaborative_mf_demo.ipynb`, `configs/mf_search.json` | Baseline tersedia dan konfigurasi 16 kombinasi telah ditetapkan. Grid penuh belum didukung runner saat ini. |
| E3 | `src/content_based.py`, metrik validation, recheck log, experiment log | Baseline TF-IDF telah dievaluasi. Dua kombinasi tercatat dalam log; seluruh 12 kombinasi belum terbukti selesai. |
| E4 | `src/semantic_embedding.py`, `src/semantic_based.py`, metrik MiniLM | Baseline MiniLM telah dievaluasi. Pencarian lengkap kedua encoder belum terbukti. |
| E5 | `configs/hybrid_search.json` | Konfigurasi tersedia, tetapi implementasi recommender hybrid dan hasil run belum tersedia. |
| E6 | `configs/hybrid_embedding_search.json` | Konfigurasi tersedia, tetapi implementasi recommender hybrid dan hasil run belum tersedia. |
| E7 | `src/user_groups.py`, `configs/user_groups.json`, `outputs/metrics/user_groups_val.json` | Analisis aktivitas pengguna telah dijalankan untuk 11,850 evaluable users. |

`src/run_grid.py` saat ini hanya mendaftarkan builder `tfidf_cbf` dan `embedding_cbf`. Oleh karena itu, konfigurasi MF dan hybrid tidak otomatis berarti grid search tersebut sudah dapat dijalankan melalui runner.

## 11. File Repository yang Menjadi Bukti

Konfigurasi:
- `configs/popularity.json`
- `configs/mf_search.json`
- `configs/tfidf_search.json`
- `configs/embedding_search.json`
- `configs/hybrid_search.json`
- `configs/hybrid_embedding_search.json`
- `configs/user_groups.json`
- `configs/reproducibility.json`

Source code:
- `src/run_grid.py`
- `src/protocol.py`
- `src/shared_eval.py`
- `src/evaluate.py`
- `src/user_groups.py`
- `src/content_based.py`
- `src/semantic_based.py`
- `src/semantic_embedding.py`

Hasil dan log:
- `outputs/logs/protocol_counts.json`
- `outputs/logs/experiment_log.csv`
- `outputs/logs/tfidf_cbf_run.json`
- `outputs/logs/semantic_cbf_run.json`
- `outputs/metrics/tfidf_cbf_val.json`
- `outputs/metrics/semantic_cbf_val.json`
- `outputs/metrics/user_groups_val.json`
