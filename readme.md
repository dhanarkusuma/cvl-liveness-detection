# Authors

| Nama | Program Studi | Universitas |
|---|---|---|
| St. Syakirah | Magister Kecerdasan Artifisial | Universitas Gadjah Mada |
| Dhanar Januardhi Kusuma | Magister Kecerdasan Artifisial | Universitas Gadjah Mada |
| Amar Ma'ruf | Magister Kecerdasan Artifisial | Universitas Gadjah Mada |
| Anas Putra Agazy | Magister Kecerdasan Artifisial | Universitas Gadjah Mada |
| Lalu Muhamad Waisul Kuroni | Magister Kecerdasan Artifisial | Universitas Gadjah Mada |

Yogyakarta, Indonesia

# Face Anti-Spoofing menggunakan CDCN

Implementasi sistem *Face Anti-Spoofing* berbasis video menggunakan metode **Central Difference Convolutional Networks (CDCN)** dan *Baseline CNN* untuk mendeteksi wajah asli (*live*) dan spoof (*fake*) dari video RGB.

Project ini dibuat sebagai tugas implementasi *Computer Vision* dan *Deep Learning* pada sistem *face liveness detection* menggunakan PyTorch.

---

# Fitur Utama

- Implementasi **CDCN (Central Difference Convolutional Networks)**
- Implementasi **Baseline Vanilla CNN**
- *Pseudo-depth supervision*
- Evaluasi menggunakan:
  - Accuracy
  - AUC
  - APCER
  - BPCER
  - ACER
- Threshold calibration menggunakan **Youden’s J Statistic**
- Ablation study untuk nilai θ (theta)
- Real-time inference menggunakan webcam
- Visualisasi depth map dan bounding box
- Analisis overfitting
