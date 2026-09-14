# 🧳 Sentiment Analysis — Review Aplikasi OTA (Play Store)

Proyek _Natural Language Processing_ untuk mengklasifikasikan sentimen ulasan pengguna Play Store terhadap 3 aplikasi **Online Travel Agent (OTA)** — **Traveloka**, **Tiket.com**, dan **Agoda** — ke dalam 3 kelas: **negative**, **neutral**, **positive**.

**Stack:** `Python`, `Transformers` (HuggingFace), `PyTorch`, `scikit-learn`, `Optuna`, `Sastrawi` & `NLTK` (NLP Bahasa Indonesia), `WordCloud`.

---

## 🧠 Ringkasan Proyek

| | |
|---|---|
| 🎯 Task | Klasifikasi sentimen 3 kelas (negative/neutral/positive) |
| 📊 Dataset | 139.903 ulasan Play Store (setelah cleaning) dari 3 aplikasi OTA |
| 🏷️ Metode Pelabelan | **Lexicon-based** — kamus kata positif/negatif Bahasa Indonesia (skor dijumlahkan per ulasan) |
| 🏆 Model Terbaik (akurasi) | **IndoBERT** (`indobert-base-p1`) — 97% |
| ✅ Model Dipakai untuk Inference | **Logistic Regression** (dipilih karena lebih ringan & efisien) |

---

## 📥 Dataset

Ulasan diambil dari Play Store untuk 3 aplikasi OTA populer di Indonesia:

| Aplikasi | Jumlah Ulasan |
|---|---|
| Traveloka | 50.000 |
| Tiket.com | 50.000 |
| Agoda | 39.911 |
| **Total (sebelum cleaning)** | **139.911** |

Setelah menghapus duplikat & baris kosong, tersisa **139.903 ulasan** yang dipakai untuk analisis.

---

## 🏷️ Metode Pelabelan: Lexicon-Based

Berbeda dari proyek sentimen youtube, proyek ini memberi label sentimen secara otomatis menggunakan **pendekatan berbasis kamus (lexicon-based)**:

1. Setiap kata dalam ulasan dicocokkan dengan kamus kata **positif** dan **negatif** Bahasa Indonesia (skor per kata).
2. Skor seluruh kata dalam satu ulasan dijumlahkan.
3. Skor **> 0** → `positive`, skor **< 0** → `negative`, skor **= 0** → `neutral`.

**Distribusi hasil pelabelan:**

| Sentimen | Jumlah | Persentase |
|---|---|---|
| Positive | 65.356 | 46.7% |
| Negative | 44.214 | 31.6% |
| Neutral | 30.333 | 21.7% |

Dari segi aplikasi, **Traveloka menerima proporsi ulasan negatif tertinggi** dibanding 2 aplikasi lain, sementara **Tiket.com menerima proporsi ulasan netral & positif terbanyak**.

---

## 🧹 Alur Preprocessing Teks

`content` (ulasan mentah) → **cleaning** (hapus mention/hashtag/RT/link/tanda baca/angka) → **casefolding** → **slang normalization** → **tokenizing** → **stopword removal** → `text_akhir`

---

## 🤖 Model yang Dibandingkan

| Model | Pendekatan | Test Accuracy | Macro F1 |
|---|---|---|---|
| **IndoBERT** (`indobert-base-p1`) | Fine-tuning, `WeightedTrainer` (cross-entropy + label smoothing 0.1), early stopping | **97%** | 0.96 |
| **Logistic Regression** | TF-IDF (n-gram 1–3) + hyperparameter tuning Optuna (30 trial) | 96% | 0.95 |
| **MLP** | TF-IDF + neural network kecil (PyTorch, 30 epoch) | 95% | 0.94 |

Ketiga model menunjukkan performa yang solid (95–97%), dengan **IndoBERT sebagai yang paling akurat**, diikuti Logistic Regression yang selisihnya tipis (~1%), lalu MLP.

**MLP menunjukkan indikasi overfitting paling jelas**
---

## ✅ Kesimpulan

1. **IndoBERT tetap menjadi model paling akurat** (97%), konsisten dengan temuan di proyek sentimen youtube 
2. **Penggunaan inference menggunakan Logistic Regression karena **efisiensi**, mengingat selisih performanya dengan IndoBERT relatif kecil (~1%).
3. Pendekatan **lexicon-based** untuk pelabelan awal terbukti cukup baik menghasilkan label yang bisa dipelajari model dengan baik (ketiga model mencapai akurasi ≥95%), meski seperti proyek sentimen youtube, label ini tetap rentan noise karena bukan hasil anotasi manual serta susah memahami makna dalam kalimat.

