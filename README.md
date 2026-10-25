# Praktik Terpadu RET503: Detektor YOLO (Kamera vs Mouse)

## 1. Tujuan
Membuat detektor objek YOLO yang bisa menemukan dan mengenali **kamera saku** dan **mouse** pada gambar. Model dilatih dengan transfer learning dari bobot pretrained COCO, lalu dibandingkan dengan model yang dilatih dari nol dan model hasil satu kali tuning.

## 2. Dataset
- Sumber: 100 foto buatan sendiri (50 kamera saku, 50 mouse) dari tugas klasifikasi ResNet18 sebelumnya.
- Pelabelan: kotak (bounding box) dibuat manual, satu kotak per foto, disimpan dalam format YOLO (`kelas cx cy w h`, nilai 0-1).
- Kelas: `0 = kamera`, `1 = mouse`.
- Pembagian data (seimbang per kelas, seed 42):

| Bagian | Jumlah gambar | Kamera | Mouse |
|---|---|---|---|
| train | 70 | 35 | 35 |
| val | 20 | 10 | 10 |
| test | 10 | 5 | 5 |

- Struktur folder:

```
dataset_v1/
├── data.yaml
├── images/ (train, val, test)
└── labels/ (train, val, test)
```

- Konversi YOLO ke COCO dan kembali ke YOLO sudah diuji, hasilnya sama dengan label asli (selisih 0).

## 3. Langkah Pengerjaan
1. Mengambil foto dari repo tugas sebelumnya dan memberi nama ulang (`kamera_001.jpg`, `mouse_001.jpg`, dst).
2. Memberi kotak pada semua foto dan menyimpannya dalam format YOLO.
3. Memeriksa label (setiap gambar punya label, nomor kelas benar, nilai kotak 0-1) dan menguji konversi YOLO <-> COCO.
4. Membagi data menjadi train/val/test dan membuat `data.yaml`.
5. Melatih tiga eksperimen (E1, E2, E3) di Google Colab dengan GPU T4.
6. Memilih model terbaik berdasarkan mAP50-95 di data val.
7. Menguji model terpilih di data test **satu kali saja**.

Seluruh langkah ada di notebook `RET503_YOLO_Colab.ipynb`.

## 4. Pengaturan Eksperimen
Data, jumlah epoch, batch, dan seed sama di semua eksperimen. Hanya satu hal yang berbeda.

| Eksperimen | Model | Perbedaan | epochs | imgsz | batch | seed |
|---|---|---|---|---|---|---|
| E1_pretrained | yolo26n.pt | baseline (transfer learning dari COCO) | 30 | 416 | 16 | 0 |
| E2_scratch | yolo26n.yaml | bobot acak, tanpa transfer learning | 30 | 416 | 16 | 0 |
| E3_tuning | yolo26n.pt | satu perubahan: imgsz=640 | 30 | 640 | 16 | 0 |

## 5. Hasil

### 5.1 Perbandingan di data val
**[ISI: salin tabel dari hasil_praktik_<NIM>.txt]**

| Eksperimen | Perubahan | mAP50 | mAP50-95 | Precision | Recall | ms/gambar |
|---|---|---|---|---|---|---|
| E1_pretrained | - (baseline) | ... | ... | ... | ... | ... |
| E2_scratch | bobot acak | ... | ... | ... | ... | ... |
| E3_tuning | imgsz=640 | ... | ... | ... | ... | ... |

Model terpilih (mAP50-95 val tertinggi): **[ISI: nama eksperimen]**

### 5.2 Hasil di data test (dipakai satu kali)
**[ISI: dari sel "Uji di data TEST"]**

| mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|
| ... | ... | ... | ... |

### 5.3 Grafik
**[ISI: tarik file gambar ke editor GitHub, lalu salin link yang muncul]**

- Kurva pelatihan E1: `![results E1](...)`
- Kurva pelatihan E3: `![results E3](...)`
- Confusion matrix model terpilih: `![confusion matrix](...)`
- Contoh prediksi di data test: `![prediksi](...)`

## 6. Analisis
**[ISI: pilih kalimat yang sesuai dengan hasilmu, lalu lengkapi angkanya]**

**E1 vs E2 (manfaat transfer learning).**
E1 mendapat mAP50-95 sebesar [..] sedangkan E2 sebesar [..]. [E1 lebih tinggi / E2 lebih tinggi / hampir sama] karena E1 memakai bobot hasil belajar dari COCO sehingga sudah mengenali tepi, tekstur, dan bentuk, sedangkan E2 harus belajar semuanya dari awal dengan hanya 70 gambar latih.

**E1 vs E3 (hasil tuning).**
E3 mengubah imgsz dari 416 menjadi 640. mAP50-95 berubah dari [..] menjadi [..], dan waktu inferensi dari [..] ms menjadi [..] ms. Perubahan ini [dipertahankan / dibatalkan] karena [alasan: akurasi naik cukup berarti / naik tipis tapi lebih lambat / turun].

**Kurva loss.**
Pada results.png, loss train dan loss val [sama-sama turun / loss val mulai naik setelah epoch ke-..]. Ini menunjukkan model [belajar dengan sehat / mulai overfitting].

**Confusion matrix.**
Kelas [..] lebih sering [terlewat / tertukar dengan kelas ..]. Dugaan penyebabnya: [misalnya sudut foto tertentu, benda terpotong di tepi foto]. (Tulis sebagai dugaan kalau belum diperiksa.)

**Hasil di data test.**
Model terpilih mendapat mAP50-95 [..] di data test, dibandingkan [..] di data val. [Selisihnya kecil / cukup besar] sehingga [model cukup stabil / perlu hati-hati karena data test hanya 10 gambar].

## 7. Keterbatasan
- Dataset kecil (100 foto), sehingga hasil bisa berubah cukup besar antar eksperimen.
- Data test hanya 10 gambar (5 per kelas), jadi angka test kurang stabil.
- Satu kotak per foto dan satu benda per foto, sehingga model belum diuji untuk gambar dengan banyak objek atau latar yang ramai.
- Kotak dibuat manual dengan ketelitian sekitar beberapa persen dari ukuran foto, bukan pas sampai piksel.
- Tiap eksperimen hanya dijalankan satu kali (seed 0), tanpa pengulangan.

## 8. Kesimpulan
**[ISI 2-3 kalimat]** Contoh kerangka: Transfer learning dari COCO [membuktikan / tidak terlalu terlihat manfaatnya] pada dataset kecil ini, dengan E1 mengungguli E2 sebesar [..] mAP50-95. Tuning imgsz=640 [membantu / tidak membantu]. Model terpilih [..] mencapai mAP50-95 [..] di data test.

## 9. Isi Repo
```
.
├── README.md
├── proyek.zip                  (tugas ResNet18 sebelumnya)
├── dataset_v1.zip              (dataset format YOLO)
├── RET503_YOLO_Colab.ipynb     (notebook Colab)
└── hasil_praktik_<4222401009_Dorotea Dai>/
    ├── E1_pretrained/          (weights, results.png, results.csv, dll)
    ├── E2_scratch/
    ├── E3_tuning/
    ├── TEST_<model terpilih>/
    ├── hasil_praktik_<NIM>.txt
    ├── hasil_praktik_<NIM>.csv
    └── annotations_coco.json
```
