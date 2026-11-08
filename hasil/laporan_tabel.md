## Tabel 1. Recognition accuracy across gesture modes

| Gesture State | Active Finger Combination | Target ROI Action | Total Trials | Correct Classification | Accuracy (%) |
| --- | --- | --- | --- | --- | --- |
| Gesture 1 | Thumb + Index Pinch | Dynamic ROI Resizing / Crop | 250 | 248 | 99.20 |
| Gesture 2 | Middle Extended | Gaussian Blur Filter | 250 | 250 | 100.00 |
| Gesture 3 | Ring Extended | Canny Edge Detection | 250 | 228 | 91.20 |
| Gesture 4 | Pinky Extended | Color Space Conversion (HSV/Gray) | 250 | 230 | 92.00 |
| **Overall Average** | **--** | **--** | **1000** | **956** | **95.60** |

### Akurasi per kondisi cahaya (untuk subbab 3.3)

| Lighting | Total Trials | Correct | Accuracy (%) |
| --- | --- | --- | --- |
| bright_300lux | 1000 | 956 | 95.60 |

### Confusion (baris = gesture sebenarnya, kolom = prediksi; 0 = tidak terdeteksi/tidak valid)

| True \ Pred | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 248 | 0 | 0 | 0 |
| 2 | 0 | 0 | 250 | 0 | 0 |
| 3 | 7 | 4 | 0 | 228 | 11 |
| 4 | 0 | 20 | 0 | 0 | 230 |

## Tabel 2. Frame rate performance across system execution modules

| Execution Module | Pipeline Operations | Average Latency (ms) | Average Frame Rate (FPS) |
| --- | --- | --- | --- |
| Landmark Tracking | MediaPipe 21 keypoint estimation | 14.173 | 70.6 |
| ROI Extraction | Euclidean distance & cropping | 0.028 | 35831.1 |
| Filter 1 Application | Gaussian Blur (15x15 Kernel) | 0.635 | 1574.5 |
| Filter 2 Application | Canny Edge Gradient Filter | 0.737 | 1356.2 |
| Filter 3 Application | BGR to Grayscale / HSV Conversion | 0.317 | 3159.1 |
| **Overall Pipeline** | Full End-to-End Processing Feed | **33.197** | **30.1** |

## Tabel 3. Baris 'Proposed Framework'

| Method / Framework | Hardware Requirement | Tracking Mechanism | ROI Dynamism | Recognition Accuracy (%) | Average FPS |
| --- | --- | --- | --- | --- | --- |
| Proposed Framework | Intel i5-11400H CPU | Landmark Vector Geometry | Fully Dynamic | 95.6% | 30.1 |

## Pengecekan angka yang tertulis di naskah (sesuaikan teks jika berbeda)

| Klaim di naskah | Hasil program |
| --- | --- |
| Akurasi keseluruhan 96.2% | 95.6% |
| Pinch (Gesture 1) tertinggi 97.6% | G1 = 99.2%, tertinggi: G2 |
| Canny (Gesture 3) 94.8% | 91.2% |
| Cahaya redup (30 lux) 91.2% | -% |
| Rata-rata 29.4 FPS | 30.1 FPS |
| Overhead ROI + filter < 3.1 ms | 0.765 ms (ROI + filter terberat) |
