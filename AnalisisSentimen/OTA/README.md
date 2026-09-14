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

<img width="251" height="71" alt="image" src="https://github.com/user-attachments/assets/7d722f80-325c-4d4d-ad8e-eee2833ba6c1" />

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

Ketiga model menunjukkan performa yang solid (95–97%), dengan **IndoBERT sebagai yang paling akurat**, diikuti Logistic Regression yang selisihnya tipis (~1%), lalu MLP. terlihat bahwa **MLP menunjukkan indikasi overfitting paling jelas**

---
## 📊 Visualisasi 
1. **Wordcloud sentimen negatif**
<img width="627" height="437" alt="image" src="https://github.com/user-attachments/assets/ae39ebe7-34ce-4bec-b23c-918ff007f0f6" />

Pada gambar, terdapat beberapa kata yang mengandung nuansa negatif, meskipun masih didominasi oleh kata-kata umum pada OTA.

2. **Wordcloud sentimen positif**
<img width="630" height="447" alt="image" src="https://github.com/user-attachments/assets/b3222b05-18f9-4b9b-87a7-9ebb85e3e691" />

Pada gambar wordcloud ini didominasi kata positive seperti ucapan terimakasih dan indikasi aplikasi yang baik.

3. **Wordcloud sentimen neutral**
<img width="630" height="433" alt="image" src="https://github.com/user-attachments/assets/a204d0b7-2082-4b1c-a538-8a26ac068b60" />

Pada gambar ini, terdapat campuran kata-kata negatif maupun positive. namun, indikasi netral tersebut dipicu oleh gabungan dari seluruh kalimat yang dilabeli neutral.

4. **Distribusi Sentimen per Travel**
<img width="866" height="417" alt="image" src="https://github.com/user-attachments/assets/aede22f7-c382-4753-bbd2-d76fcd4f8ec9" />

Pada gambar, berdasarkan pelabelan lexicon, terlihat bahwa semua ulasan positif mendominasi di setiap travel serta jumlah ulasan positive terbanyak didapatkan oleh Tiket.com dan ulasan negative terbanyak didapatkan oleh Traveloka.


## ✅ Kesimpulan

1. **IndoBERT tetap menjadi model paling akurat** (97%), konsisten dengan temuan di proyek sentimen youtube 
2. **Penggunaan inference menggunakan Logistic Regression karena **efisiensi**, mengingat selisih performanya dengan IndoBERT relatif kecil (~1%).
3. Pendekatan **lexicon-based** untuk pelabelan awal terbukti cukup baik menghasilkan label yang bisa dipelajari model dengan baik (ketiga model mencapai akurasi ≥95%), meski seperti proyek sentimen youtube, label ini tetap rentan noise karena bukan hasil anotasi manual serta susah memahami makna dalam kalimat.

