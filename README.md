# Credit Risk Prediction
**Final Project IDX Partners**
---
## Project Overview
Project ini bertujuan untuk membangun model Machine Learning yang dapat memprediksi risiko kredit nasabah (*Credit Risk Prediction*) pada perusahaan *Fintech*. Model ini mengklasifikasikan nasabah ke dalam dua kategori: *Good Loan* dan *Bad Loan*, berdasarkan data profil serta historis pinjaman.

## Tujuan Bisnis
1. **Problem Statement**: Meminimalkan risiko gagal bayar yang menyebabkan kerugian finansial, serta meminimalkan penolakan nasabah potensial yang berujung pada hilangnya potensi pendapatan.
2. **Objektif**: Membangun model yang menghasilkan skor probabilitas gagal bayar dan mengidentifikasi fitur-fitur yang paling berpengaruh untuk interpretabilitas pemangku kepentingan.
3. **Metrik Utama**: Fokus pada nilai *Recall*, *ROC-AUC*, *F1-Score*, dan *Precision*.

## Struktur Repositori
```text
├── image/                 # Direktori untuk file gambar visualisasi (PNG)
├── notebooks/             # Jupyter notebook untuk EDA dan pemodelan (project.ipynb)
├── README.md              # Dokumentasi utama proyek
```

## Data Understanding
* **Dataset**: `loan_data_2007_2014.csv` (466.285 baris, 75 kolom).
* **Variabel Target**: Kolom `loan_status` dipetakan menjadi kelas biner:
  * **0 (Good Loan)**: Current, Fully Paid, dll.
  * **1 (Bad Loan)**: Default, Charged Off, Late, dll.
* **Kondisi Data**: Ditemukan ketidakseimbangan kelas (*imbalanced data*) yang signifikan, banyak fitur dengan *missing values* hingga 100%, serta kehadiran *outliers* pada fitur numerik.

## Data Preparation & Feature Engineering
* **Data Cleansing**: Identifikasi dan penanganan *missing values* secara menyeluruh, serta membuang fitur yang memiliki persentase data kosong terlalu tinggi.
* **Outlier Handling**: Analisis *skewness* dan deteksi *outlier* menggunakan metode IQR (*Interquartile Range*).
* **Feature Engineering**: Seleksi fitur (*feature selection*) dan transformasi data numerik/kategorikal ke dalam format yang siap digunakan oleh algoritma Machine Learning.

## Model Architecture & Evaluation
* **Eksperimen Model**: Dilakukan perbandingan performa antara Logistic Regression, XGBoost, dan LightGBM.
* **Model Terbaik**: LightGBM

### Hasil Evaluasi (LightGBM):

| Metrik | Skor |
| :--- | :--- |
| **ROC-AUC** | 0.6951 |
| **Recall** | 0.6644 |
| **F1-Score** | 0.2943 |
| **Precision** | 0.1890 |

* **Kesimpulan Model**: LightGBM memberikan performa deteksi *Bad Loan* (Recall) yang paling optimal dengan proses komputasi yang efisien dibandingkan model lainnya.

## Top 5 Visualizations & Insights

### 1. Target Distribution
![Target Distribution](image/Target%20Distribution.png)

* **Insight**: Proporsi kelas target menunjukkan data yang sangat imbalanced (didominasi oleh *Good Loan*). Hal ini menegaskan mengapa akurasi tidak bisa dijadikan metrik utama dan model difokuskan pada optimasi *Recall*.

### 2. Feature Correlation Heatmap
![Feature Correlation Heatmap](image/Top%2015%20Korelasi%20Heatmap.png)
* **Insight**: Menampilkan 15 fitur dengan korelasi tertinggi. Visualisasi ini membantu mengidentifikasi multikolinearitas antar variabel numerik untuk proses reduksi fitur.

### 3. Model Evaluation Comparison
![Model Evaluation](image/Model%20Evaluation.png)

* **Insight**: Perbandingan metrik antar model menunjukkan LightGBM memiliki keunggulan yang konsisten dalam membedakan kelas (*ROC-AUC*) dan mendeteksi risiko tinggi (*Recall*) dibandingkan Logistic Regression dan XGBoost.

### 4. Feature Importance
![Feature Importance](image/Top%2010%20Fitur%20Paling%20Berpengaruh.png)

* **Insight**: Menyoroti 20 variabel terpenting (seperti `recoveries`, `int_rate`, atau histori keterlambatan) yang paling berkontribusi dalam keputusan prediksi model, memberikan dasar bagi kebijakan kredit bisnis.

## Clone Repositori
```bash
# Clone repositori ini
git clone https://github.com/Juliama26/idx-partners-ds-project

# Masuk ke direktori proyek
cd idx-partners-ds-project
```
