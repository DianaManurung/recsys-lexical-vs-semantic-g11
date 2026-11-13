# MF validation: hasil dan analisis awal W10

## Evaluasi baseline resmi

MF dievaluasi pada validation menggunakan `src.protocol.build_protocol` dan `src.evaluate.evaluate_model`. Daftar evaluable users berisi 11.659 user dan cocok persis dengan daftar resmi. Evaluasi mencatat nol pelanggaran leakage.

| Model | Recall@10 | NDCG@10 | HitRate@10 | Item unik di seluruh Top-10 |
|---|---:|---:|---:|---:|
| Popularity | 0.013499 | 0.009222 | 0.031049 | 18 |
| MF | 0.004289 | 0.002762 | 0.009263 | 3.620 |
| TF-IDF | 0.012612 | 0.007342 | 0.021100 | 16.093 |

Pada evaluasi penuh, Recall MF sekitar 68.2% lebih rendah daripada Popularity dan NDCG-nya sekitar 70.1% lebih rendah. Namun MF menghasilkan rekomendasi yang jauh lebih beragam: 3.620 item unik (15.91% dari 22.753 kandidat), dibanding 18 item untuk Popularity (0.079%). Jadi angka lama “hanya sekitar 17 item unik” tidak menggambarkan baseline MF saat ini; angka itu lebih dekat dengan keluaran Popularity atau MF yang diberi item-bias saat inferensi.

## Pilot sensitivitas

Pilot dijalankan pada 1.000 evaluable users yang sama, dipilih secara deterministik dengan seed 42. MF default dilatih dengan faktor, learning rate, regularisasi, dan seed yang sama seperti baseline. Item-bias weight diuji sebagai pengali skor saat inferensi pada model fit yang sama. Epoch 5 dan 20 dilatih ulang dengan item-bias weight 0.

| Setting | Recall@10 | NDCG@10 | HitRate@10 | Item unik di Top-10 sampel |
|---|---:|---:|---:|---:|
| Popularity (referensi sampel) | 0.016172 | 0.010305 | 0.033 | 14 |
| MF, 10 epoch, item-bias weight 0 | 0.007750 | 0.003675 | 0.013 | 1.274 |
| MF, weight 0.25 | 0.000833 | 0.000683 | 0.002 | 25 |
| MF, weight 0.5 | 0.000976 | 0.000762 | 0.003 | 18 |
| MF, weight 1 | 0.000976 | 0.000766 | 0.003 | 16 |
| MF, 5 epoch, weight 0 | 0.001400 | 0.000649 | 0.003 | 5.078 |
| MF, 20 epoch, weight 0 | 0.007083 | 0.003869 | 0.013 | 284 |

### Temuan

- Item-bias weight positif pada pengujian ini membuat MF kembali terkonsentrasi pada sekitar 16–25 item dan menurunkan metrik dengan tajam. Ini mendukung keputusan konfigurasi baseline saat ini untuk memakai `item_bias_weight=0`.
- Lima epoch terlalu sedikit pada pilot ini: metrik ranking lebih rendah daripada 10 epoch, walaupun cakupan item lebih luas.
- Dua puluh epoch menaikkan NDCG pilot sekitar 5.3% dibanding 10 epoch, tetapi Recall turun sekitar 8.6% dan jumlah item unik di sampel turun dari 1.274 menjadi 284. Belum ada dasar untuk mengganti baseline hanya dari hasil sampel ini; pilihan epoch perlu dibandingkan pada populasi penuh jika ingin tuning final.
- Bias global dan user-bias bernilai konstan untuk semua item dalam ranking satu user, sehingga tidak mengubah urutan Top-N. Komponen bias yang relevan untuk ranking item adalah item bias, dan pilot menunjukkan bobotnya perlu diperlakukan hati-hati.
- Hasil memberi indikasi bahwa MF saat ini menukar sebagian akurasi Top-10 dengan keragaman katalog. Hasil ini sendiri belum membuktikan penyebab umum MF tertinggal; model MF dilatih untuk memprediksi rating eksplisit, sedangkan evaluasi menilai ranking item relevan.

## Batas interpretasi dan reproduksibilitas

Metrik tabel baseline berasal dari evaluasi penuh 11.659 user pada run bersama 13 November 2026. Pilot memakai 1.000 user dan environment berbeda (Python 3.11.4, NumPy 1.26.4), sehingga angka absolut pilot tidak boleh dipertukarkan dengan metrik penuh. Perbandingan antar-setting di dalam pilot menggunakan user yang sama dan model fit yang dikendalikan.

Perintah untuk mengulang pilot dari root repo:

```bash
python -m src.mf_bias_sensitivity
```

Hasil rinci tersimpan di `outputs/metrics/mf_bias_sensitivity_pilot.csv`; parameter, hash input, environment, dan waktu run tercatat di `outputs/logs/mf_bias_sensitivity_pilot.json`.

## Teks metode MF yang sesuai implementasi

Untuk rekomendasi, skor kandidat item dihitung sebagai

\[
s(u,i)=\lambda_b b_i + \mathbf{p}_u^\top\mathbf{q}_i,
\]

dengan \(\mathbf{p}_u\) dan \(\mathbf{q}_i\) sebagai faktor laten user dan item, serta \(b_i\) sebagai item bias yang dikalikan bobot \(\lambda_b\). Baseline W10 menetapkan \(\lambda_b=0\), sehingga ranking kandidat menggunakan dot product faktor laten \(\mathbf{p}_u^\top\mathbf{q}_i\). Rekomendasi diurutkan menurun berdasarkan skor tersebut di antara warm candidates yang belum pernah dilihat user pada data training.

Paragraf ini siap dipindahkan ke bagian Method di Overleaf; file sumber Method tersebut tidak tersedia di checkout lokal.
