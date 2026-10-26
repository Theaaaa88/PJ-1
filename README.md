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
<img width="371" height="91" alt="image" src="https://github.com/user-attachments/assets/ba709a75-1552-4c03-ab9b-35903a55f80a" />

Model terpilih (mAP50-95 val tertinggi): **E1_pretrained**




### 5.2 Hasil di data test (dipakai satu kali)
*<img width="398" height="104" alt="image" src="https://github.com/user-attachments/assets/f3e1a712-5683-4943-af9e-5c89c01d7335" />


### 5.3 Grafik
E1_pretrained
<img width="539" height="260" alt="image" src="https://github.com/user-attachments/assets/59687546-218b-49b2-a359-712bc7b6259b" />

E3_tuning
<img width="548" height="269" alt="image" src="https://github.com/user-attachments/assets/e7b07960-bf88-4eef-9b55-7867bf885b41" />



## 6. Analisis

**E1 vs E2 (manfaat transfer learning).**
E1 (pretrained COCO) mencapai mAP50-95 sebesar 0.725 di data val, sedangkan E2 (bobot acak) hanya 0.002. E1 sudah punya pengetahuan dasar dari COCO (tepi, tekstur, bentuk benda), sehingga cukup menyesuaikan diri ke dua kelas baru. E2 harus belajar dari nol dengan hanya 70 gambar latih dan 30 epoch, sehingga belum sempat belajar mendeteksi. Hasil ini menunjukkan manfaat transfer learning pada dataset kecil, tetapi bukan bukti bahwa model dari nol "tidak bisa": dengan data dan epoch jauh lebih banyak, hasilnya bisa berbeda.

**E1 vs E3 (hasil tuning).**
E3 mengubah satu hal saja, yaitu imgsz dari 416 menjadi 640. mAP50-95 turun dari 0.725 menjadi 0.680, recall dari 0.994 menjadi 0.900, dan precision dari 0.954 menjadi 0.932. Jadi memperbesar ukuran gambar tidak membantu pada dataset ini. Benda pada foto sudah cukup besar, sehingga resolusi tambahan kemungkinan tidak memberi informasi baru (dugaan, belum diuji). Karena val hanya 20 gambar, selisih sekecil ini juga bisa terjadi secara kebetulan. Perubahan imgsz=640 tidak dipertahankan, dan E1_pretrained dipilih sebagai model akhir.

**Kurva pelatihan.**
Pada E1 maupun E3 ada ketidakstabilan di awal: val box_loss melonjak sekitar epoch 6-8 dan mAP di data val turun tajam di sekitar epoch 8-9 sebelum naik lagi. Dugaan penyebabnya adalah tahap awal pelatihan (warm-up) yang dipadukan dengan data yang sangat sedikit, sehingga evaluasi di 20 gambar val mudah naik-turun (dugaan, belum diuji). Setelah itu, pada E1 loss train dan val sama-sama turun, mAP50 naik ke sekitar 0.97, dan mAP50-95 naik sampai 0.725 serta masih cenderung naik di epoch terakhir. Ini menandakan E1 belum benar-benar konvergen, dan menambah epoch kemungkinan masih bisa memperbaikinya (dugaan). Pada E3, loss train terus turun, tetapi val box_loss naik tipis di beberapa epoch terakhir dan mAP50-95 turun dari sekitar 0.677 menjadi 0.655. Ini bisa jadi tanda awal overfitting, tetapi perubahannya kecil dan data val hanya 20 gambar, sehingga belum bisa disimpulkan dengan pasti.

**Perbedaan antar kelas.**
Kelas kamera lebih sulit daripada mouse. Di data val (E3), kamera mendapat recall 0.779 dan mAP50-95 0.567, sedangkan mouse recall 1.000 dan mAP50-95 0.784. Di data test (E1) polanya sama: kamera 0.688 dan mouse 0.934. Konsistennya pola ini menunjukkan bahwa kamera memang lebih sulit dideteksi dengan tepat. Dugaan penyebabnya: tampilan kamera lebih beragam antar sudut foto (depan, samping, atas, bawah), dan kotak label kamera lebih sulit dibuat pas.

**Selisih mAP50 dan mAP50-95.**
Nilai mAP50 sudah sangat tinggi (0.977 di val dan 0.995 di test), tetapi mAP50-95 lebih rendah (0.725 dan 0.811). Artinya model hampir selalu menemukan bendanya, tetapi posisi kotaknya belum selalu serapi label saat standar kecocokan dinaikkan. Salah satu faktor yang mungkin berpengaruh adalah label yang dibuat manual dengan ketelitian beberapa persen dari ukuran foto (dugaan, belum diuji).

**Hasil di data test.**
Model terpilih (E1_pretrained) mendapat mAP50-95 sebesar 0.811 di data test, lebih tinggi dari 0.725 di data val. Selisih ini tidak boleh dibaca sebagai "model makin bagus di data baru". Data test hanya 10 gambar, sehingga satu gambar saja bisa menggeser angkanya cukup jauh. Yang bisa disimpulkan dengan aman adalah bahwa model tidak menurun di data yang belum pernah dilihat. Recall 1.000 berarti kesepuluh benda terdeteksi, dan precision 0.992 berarti hampir tidak ada deteksi yang salah.

**Kecepatan.**
Kolom ms/gambar di tabel 5.1 tidak stabil. E3 (gambar lebih besar) tampak lebih cepat (9.446 ms) daripada E1 (12.764 ms), padahal secara teori seharusnya sebaliknya, dan pada evaluasi test E1 hanya butuh 5.5 ms per gambar. Pengukuran dilakukan satu kali dan dipengaruhi kondisi GPU saat itu, sehingga kecepatan tidak dipakai sebagai dasar memilih model.

## 7. Keterbatasan
- Dataset kecil (100 foto), sehingga hasil bisa berubah cukup besar antar eksperimen.
- Data test hanya 10 gambar (5 per kelas), jadi angka test kurang stabil.
- Satu kotak per foto dan satu benda per foto, sehingga model belum diuji untuk gambar dengan banyak objek atau latar yang ramai.
- Kotak dibuat manual dengan ketelitian sekitar beberapa persen dari ukuran foto, bukan pas sampai piksel.
- Tiap eksperimen hanya dijalankan satu kali (seed 0), tanpa pengulangan.

## 8. Kesimpulan
Berdasarkan grafik hasil pelatihan model, E1_pretrained dan E3_tuning sama-sama menunjukkan konvergensi yang sangat baik dengan tren nilai loss (box, cls, dan l1 pada data train maupun val) yang menurun secara konsisten serta peningkatan metrik akurasi (precision, recall, dan mAP50) yang stabil mendekati angka 1.0. Namun, jika dibandingkan pada metrik ketat mAP50-95, model E1_pretrained menunjukkan hasil akhir yang sedikit lebih unggul dengan mencapai puncak sekitar 0.72, sedangkan E3_tuning berada di kisaran 0.68 hingga 0.70, yang mengindikasikan bahwa pendekatan pretrained memberikan performa generalisasi yang sedikit lebih optimal pada kasus ini.

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
