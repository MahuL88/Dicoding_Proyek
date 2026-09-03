# 🍽️ Food Image Classification (Food-101)

Proyek _Computer Vision_ untuk mengklasifikasikan gambar makanan ke dalam **12 kategori** menggunakan _transfer learning_ **DenseNet201**. Model final diekspor ke 3 format sekaligus — **SavedModel**, **TensorFlow.js**, dan **TensorFlow Lite** 

---

## 🧠 Ringkasan Model

| | |
|---|---|
| 🏗️ Arsitektur | DenseNet201 |
| 🗂️ Jumlah Kelas | 12 (dipilih dari 101 kelas Food-101) |
| 🖼️ Total Gambar | 12.000 (7.200 train / 1.800 validasi / 3.000 test) |
| 🔌 Input | Gambar mentah RGB (0–255), ukuran 300×300 |

**12 kelas yang dipilih:** `donuts`, `bibimbap`, `edamame`, `french_fries`, `ramen`, `hot_dog`, `ice_cream`, `miso_soup`, `mussels`, `pizza`, `sashimi`, `spaghetti_bolognese`.

---

## 📥 Dataset

Dataset keseluruhan dapat diunduh pada [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (101 kelas, 1.000 gambar/kelas). Notebook ini hanya memakai 12 kelas terpilih.

---

## 🚀 Cara Menjalankan

1. **Training ulang**
Buka `klasifikasi-gambar.ipynb`, jalankan cell dari atas ke bawah (pastikan path dataset sudah sesuai).

2. **Langsung pakai model**

      | Format | Lokasi | Cocok untuk |
      |---|---|---|
      | SavedModel | `saved_model/` | Server / TensorFlow Serving |
      | TensorFlow.js | `tfjs_model/` | Aplikasi web (browser) |
      | TensorFlow Lite | `tflite/food_model.tflite` | Mobile / edge device |

---

## 📁 Struktur Direktori
```plaintext
📦 Food-Image-Classification
 ┣ 📂saved_model
 ┃ ┗ 📂food_model_v1
 ┃    ┣ 📂assets
 ┃    ┣ 📂variables
 ┃    ┗ 📜saved_model.pb
 ┣ 📂tfjs_model
 ┃ ┣ 📜model.json
 ┃ ┗ 📜*.bin (weight shards)
 ┣ 📂tflite
 ┃ ┣ 📜food_model.tflite
 ┃ ┗ 📜labels.txt
 ┣ 📜klasifikasi-gambar.ipynb
 ┣ 📜requirements.txt
 ┗ 📜README.md
```

---

## 📊 Hasil

| Tahap | Val Accuracy |
|---|---|
| Training awal (backbone freeze, ~25 epoch dgn early stopping) | ~94.4% |
| Fine-tuning (unfreeze 30 layer terakhir, 5 epoch) | ~94.1% |

**Test set (3.000 gambar):**
- Accuracy: **~96.6%**
- Precision/Recall/F1 per kelas mayoritas di kisaran **0.94–1.00** — kelas `edamame` sempurna (1.00), kelas `ramen` paling rendah (F1 0.94) karena mirip secara visual dengan `miso_soup` (sama-sama sup dalam mangkuk).

---

## ✅ Kesimpulan dan Saran

- Transfer learning **DenseNet201** sangat efektif untuk klasifikasi 12 kelas makanan ini, mencapai **~96–97% test accuracy**.
- 🧪 Uji performa model di gambar dunia nyata (bukan dari dataset Food-101) untuk memvalidasi generalisasi sebelum benar-benar dipakai di aplikasi produksi.
- 🌍 Pertimbangkan menambah jumlah kelas secara bertahap untuk melihat seberapa jauh arsitektur ini bisa di-scale.
