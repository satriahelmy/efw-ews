# 📌 efw-ews

**Electricity Fault Weather - Early Warning System**  
Proyek machine learning untuk mendeteksi dan memprediksi gangguan listrik akibat cuaca ekstrem di jaringan distribusi listrik.

---

## Ringkasan

Gangguan listrik yang disebabkan oleh cuaca ekstrem berdampak signifikan pada keandalan pasokan energi dan menyebabkan kerugian ekonomi. Proyek ini mengembangkan model pembelajaran mesin untuk mendeteksi dan memprediksi potensi gangguan listrik berdasarkan data historis cuaca dan kejadian gangguan.

---

## Fitur Utama

✅ Prediksi risiko gangguan listrik dengan machine learning  
✅ Integrasi data cuaca dan real-time  
✅ Transformasi fitur dengan Yeo-Johnson dan robust scaling  
✅ Penanganan ketidakseimbangan kelas dengan SMOTE, ADASYN dan SMOTE-ENN
✅ Geospatial enrichment menggunakan jarak ke pusat beban dan clustering lokasi  
✅ Penggunaan model ensemble: Random Forest, XGBoost, AdaBoost, dan LightGBM  
✅ Evaluasi model dengan metrik F1-Score, False Negative, dan Confusion Matrix

---

## 📊 Dataset

- **Data gangguan listrik:** data histori kejadian gangguan pada gardu, trafo, dan jaringan distribusi.
- **Data cuaca:** data dari Open Meteo

---

## ⚙️ Alur Kerja

1. **Pengumpulan Data:** Download data cuaca dan histori gangguan.
2. **Pra-pemrosesan:**  
   - Penggabungan dataset cuaca & gangguan.
   - Penambahan fitur spasial.
   - Transformasi numerik dan penanganan outlier.
   - Penyeimbangan kelas.
3. **Pelatihan Model:** Training beberapa algoritma ensemble learning.
4. **Evaluasi Model:** Pemilihan model terbaik berdasarkan F1-Score & jumlah False Negative.
5. **Prediksi:** Implementasi model terlatih untuk prediksi gangguan dengan data cuaca terbaru.
6. **Visualisasi:** Dashboard monitoring prediksi risiko gangguan.
---

## Cara Deploy di Server

```bash
git clone https://github.com/username/efw-ews.git
cd efw-ews
pip install -r requirements.txt
python app.py
```

---

## 👥 Kontributor

- **Helmy Satria**
- **Alit Ksatria**
- **Intan Jelita Saragih**

---
