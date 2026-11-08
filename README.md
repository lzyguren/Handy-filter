## *NOTE: Program ini dibuat untuk mendukung penelitian: "Implementation of MediaPipe Hands for Real-Time Gesture-Based Image Segmentation and Transformation"

## Handy-filter
Real-Time Hand Gesture-Controlled ROI Segmentation & Filter (MediaPipe Hands + OpenCV)

#### INSTALL (sekali saja, di terminal VS Code):
pip install opencv-python mediapipe==0.10.14 numpy

#### JALANKAN:
python gesture_roi.py                # mode kamera
python gesture_roi.py --report       # hanya buat ulang laporan tabel dari CSV
python gesture_roi.py --reset        # hapus data eval/benchmark lama
python gesture_roi.py --cam 1        # pakai kamera lain (bila ada)
python gesture_roi.py --fpg 250      # frame per gesture saat evaluasi (4x250 ~ 1000 frame)

#### GESTUR (jempol + telunjuk selalu menentukan ukuran ROI, jari lain memilih filter):
Gesture 1: jari tengah, manis, kelingking terlipat  -> ROI saja (crop/resize)
Gesture 2: hanya jari tengah lurus                  -> Gaussian Blur 15x15
Gesture 3: hanya jari manis lurus                   -> Canny Edge
Gesture 4: hanya kelingking lurus                   -> Grayscale (tekan H untuk HSV)

#### TOMBOL:
S = simpan screenshot (bahan Gambar 3; setelah 4 gesture tertangkap -> fig3_composite.png)
K = tampilkan/sembunyikan 21 landmark + P4, P8 (bahan Gambar 2)
L = ganti label kondisi cahaya (bright 300 / moderate 150 / dim 30 lux)
E = evaluasi akurasi otomatis (isi Tabel 1)
B = benchmark latency & FPS (isi Tabel 2 & baris "Proposed" Tabel 3)
H = ganti Grayscale <-> HSV untuk Gesture 4
X = batalkan evaluasi/benchmark yang sedang berjalan
R = buat laporan tabel      Q / ESC = keluar

### Semua hasil ada di folder ./hasil
========================================
