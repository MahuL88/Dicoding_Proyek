# 💬 Sentiment Analysis — Komentar YouTube Game

Proyek _Natural Language Processing_ untuk mengklasifikasikan sentimen komentar YouTube (Bahasa Indonesia) terhadap beberapa game, seperti **Mobile Legends: Bang-Bang**, **The Classrooms**, dan **I Am Fish** ke dalam 3 kelas: **negative**, **neutral**, **positive**. 

**Stack:** `Python`, `Transformers` (HuggingFace), `PyTorch`, `scikit-learn`, `Optuna`, `Sastrawi` & `NLTK` (NLP Bahasa Indonesia).

---

## 📥 Dataset

Ulasan diambil dari Youtube untuk 3 jenis video:

| Sentiment | Jumlah  |
|---|---|
| Neutral | 7.064 |
| Positive | 2.580 |
| Negative | 1.373 |

---

## 🧠 Ringkasan Model

| | |
|---|---|
| 🎯 Task     | Klasifikasi sentimen 3 kelas (negative/neutral/positive) |
| 📊 Dataset | 11.017 komentar YouTube |
| 🧠 Model    | **IndoBERT** (`xlm-roberta-base`, fine-tuned), Logistic Regression (TF-IDF + Optuna tuning), MLP (TF-IDF + oversampling) |

---

## 🧹 Alur Preprocessing Teks

Sebelum digunakan untuk analisis sentimen, teks komentar mentah diproses melalui beberapa tahap sebelum dipakai untuk pelabelan/training:

```text
text
  │
  ▼
🧹 Cleaning
  │  • perbaikan encoding
  │  • menghapus mention, hashtag, URL, dan angka
  │  • normalisasi karakter berulang
  │  • normalisasi ekspresi tertawa
  │  • normalisasi tanda baca dan spasi
  ▼
🔡 Casefolding
  │  • mengubah seluruh teks menjadi huruf kecil
  ▼
💬 Slang Normalization
  │  • mengubah kata gaul, singkatan,
  │    dan bentuk tidak baku menjadi bentuk normal
  ▼
🎮 Title Normalization
  │  • mempertahankan penulisan nama game
  │    dengan format title case
  ▼
text_title
  │
  ├──────────────► 🏷️ Sentiment Labeling
  │                 menggunakan IndoRoBERTa
  │
  └──────────────► 🤖 Model Training
```

Karena menggunakan model berbasis Transformer, preprocessing dipertahankan sampai `text_title` untuk menjaga informasi linguistik dan konteks komentar. **Stemming** dapat mengubah kata ke bentuk dasarnya, sedangkan **stopword removal** dapat menghilangkan kata-kata yang meskipun umum, tetap dapat berperan dalam konteks suatu kalimat.

> ⚠️ **Penting:** Kolom yang digunakan sebagai input model adalah **`text_title`**, bukan `text_akhir`. 

Implementasi preprocessing tersebut  dapat digunakan kembali pada inference maupun pengembangan model melalui fungsi:

```python
cleaningText()
casefoldingText()
fix_slangwords()
TitleGame()
```
Fungsi tersebut dapat ditemukan  pada notebook pelabelan 

--- 

Pada tahap inference, input harus berada dalam format hasil preprocessing, yaitu text_title. Pengguna dapat memilih untuk menjalankan kembali fungsi preprocessing yang tersedia pada notebook pelabelan, atau langsung menggunakan teks yang sebelumnya telah melalui preprocessing.

Sebagai contoh, teks yang telah dinormalisasi seperti “yapping” → “ngoceh” dapat langsung digunakan sebagai input inference tanpa perlu diproses kembali. Hal ini memastikan format teks yang diberikan kepada model tetap konsisten dengan data yang digunakan sebelumnya.

---

## 📁 Struktur Direktori
```plaintext
📦 Sentiment-Analysis-YouTube-Comments
 ┣ 📂indobert_sentiment_pt
 ┃ ┣ 📜config.json
 ┃ ┣ 📜model.safetensors
 ┃ ┣ 📜tokenizer.json
 ┃ ┗ 📜tokenizer_config.json
 ┣ 📂Label_komentar_youtube.csv
 ┣ 📜pelabelan.ipynb
 ┣ 📜proyek-sentiment-analysis-on-youtube.ipynb
 ┗ 📜README.md
```

---

## 📊 Hasil
<img width="456" height="101" alt="image" src="https://github.com/user-attachments/assets/5bd3fd43-6f25-4239-96f3-88afa9042038" />

Berdasarkan jenis sentimen, 80.55% dari seluruh sentimen negative yang ada pada dataset berasal dari game MLBB, sedangkan sentimen neutral dan positive pada seluruh dataset didominasi oleh game The Clasrooms

<img width="451" height="101" alt="image" src="https://github.com/user-attachments/assets/4d8c2f8a-ab77-4c95-9023-4876af7bea12" />

Berdasarkan kategori game, semua game didominasi oleh sentiment neutral yaitu 77.76% pada I Am Fish, 57.46% pada game MLBB, dan 61.93 pada game The Classrooms. 

---
| Model | Karakteristik |
|---|---|
| **IndoBERT** (`xlm-roberta-base`) | Fine-tuning dengan **Focal Loss** + **class weight**, sebagian layer awal di-*freeze*, validasi dengan **3-fold Stratified Cross Validation**, **early stopping** berbasis F1-macro. |
| Logistic Regression | TF-IDF (`max_features=15000`, n-gram 1–3) + hyperparameter tuning **Optuna** (30 trial) untuk `C`, `penalty`, `class_weight`. |
| MLP | TF-IDF (`max_features=12000`, n-gram 1–3) + **RandomOverSampler**, hidden layer kecil (64 unit) dengan regularisasi & early stopping. |

**IndoBERT dipilih sebagai model final** karena unggul pada evaluasi F1-macro/classification report per kelas, terutama pada kelas minoritas (`negative`) yang paling sulit diklasifikasikan oleh model berbasis TF-IDF sederhana.

---

## ✅ Kesimpulan

1. **IndoBERT (xlm-roberta-base)** menunjukkan performa terbaik dalam klasifikasi sentimen komentar game berbahasa Indonesia dibandingkan pendekatan **TF-IDF + model klasifikasi klasik**. Hasil ini menunjukkan bahwa model berbasis *Transformer* lebih sesuai untuk menangani karakteristik komentar yang bersifat informal dan memiliki konteks yang beragam.

2. Kombinasi **Focal Loss + class weight** membantu model dalam menangani ketidakseimbangan kelas, khususnya ketika jumlah data pada kelas `negative` jauh lebih sedikit dibandingkan kelas `neutral`. Pendekatan ini membantu memberikan perhatian yang lebih besar terhadap kelas minoritas selama proses pembelajaran.

3. **Preprocessing komentar YouTube memiliki keterbatasan karena karakteristik data yang sangat beragam dan tidak terstruktur.** Komentar masih mengandung *typo*, slang, singkatan, istilah baru, kata bahasa asing yang diadaptasi ke penulisan Indonesia seperti *“real” → “ril”* dan *“my” → “mai”*, serta penggunaan berbagai bahasa dalam satu komentar seperti bahasa Indonesia, Inggris, Portugis/Brazil, dan Mandarin. Selain itu, terdapat komentar yang hanya berupa *emoticon*, simbol, atau kombinasi kata yang sulit dinormalisasi. Kondisi tersebut menyebabkan data teks belum dapat dibersihkan secara sempurna di setiap komentar meskipun telah melalui tahapan preprocessing.

4. **Kualitas pelabelan sentimen juga masih menjadi keterbatasan penelitian.** Variasi bahasa, *typo*, slang, istilah baru, komentar yang sangat singkat, serta penggunaan bahasa campuran dapat menyebabkan model pelabelan kesulitan memahami maksud sebenarnya dari komentar. Dengan demikian, masih terdapat kemungkinan kesalahan pada label yang dihasilkan dan kondisi tersebut dapat turut memengaruhi hasil klasifikasi sentimen.

5. Meskipun preprocessing dilakukan secara konsisten hingga `text_title`, **konsistensi preprocessing tidak secara otomatis menjamin bahwa seluruh data telah memiliki representasi teks dan label yang sempurna**. Hasil penelitian menunjukkan bahwa pada data komentar YouTube yang sangat bervariatif, keterbatasan normalisasi bahasa dan kualitas label tetap menjadi faktor yang perlu diperhatikan dalam penerapan model pada data nyata.

6. Pelabelan sentimen didasarkan pada reaksi penonton terhadap game yang sedang dimainkan, bukan terhadap streamer. Oleh karena itu, ekspresi seperti “seram banget”, “merinding”, “takut”, atau “deg-degan” dapat dikategorikan sebagai positive apabila konteksnya menunjukkan bahwa game berhasil memberikan pengalaman yang menarik atau memicu respons emosional dari penonton. Hal ini menunjukkan bahwa penentuan sentimen perlu mempertimbangkan konteks dan objek yang menjadi sasaran komentar, sehingga tidak selalu dapat ditentukan hanya berdasarkan makna positif atau negatif dari suatu kata.
