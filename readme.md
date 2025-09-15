Tujuan Adaptasi Kita
Kita akan uji apakah strategi ini bisa di-scale up ke 5 secret dengan pendekatan berikut:

✔️ Pendekatan Awal (v1): Simple Concatenation
Gabungkan 5 secret image menjadi tensor shape = (5, 3, 64, 64) → reshape menjadi (15, 64, 64).
- Cover image tetap (3, 448, 448) → diproses sendiri.
- Secret tensor di-encode menjadi feature map yang di-upsample ke ukuran cover.
- Kedua feature maps digabung (concat) → diproses menjadi stego frame.

Jika ini gagal mempertahankan PSNR atau recovery bagus, maka kita akan coba:
- Pendekatan multi-branch encoder: satu encoder per secret → merge.
- Shared backbone + attention per-secret branch.

Kita akan memulai dari nol, membangun ulang pipeline ini. Langkah-langkahnya adalah:
1. Modifikasi model/: Mengubah arsitektur StegoEncoder agar dapat menerima num_secrets yang fleksibel.
2. Modifikasi Preprocessing: Membuat skrip untuk menyiapkan dataset dengan struktur folder baru (covers/ dan secrets/) dan skrip untuk membuat split train/, val/, test/.
3. Modifikasi training/: Menyesuaikan skrip pelatihan agar secara acak memuat data dari folder-folder tersebut, dan memvisualisasikan grafik loss pelatihan dan validasi.
4. Modifikasi evaluation/: Menyesuaikan skrip evaluasi agar dapat mengukur PSNR, kapasitas, dan runtime untuk skema N secret images yang berbeda.
5. Eksperimen: Melakukan pelatihan dan evaluasi untuk N=1, N=2, N=3, N=4, dan N=5 untuk melihat performa model dan menemukan titik batasnya.

Target dan penyesuaian yang Anda usulkan sangat sesuai dan merupakan langkah maju yang benar untuk mengatasi masalah yang telah kita identifikasi. Anda telah menganalisis keterbatasan model dan pipeline saat ini dengan sangat baik.

Pendekatan Anda untuk membuat skema penyisipan yang dinamis, memodifikasi preprocessing, dan menggunakan split dataset untuk evaluasi overfitting adalah praktik terbaik dalam machine learning dan akan menghasilkan temuan yang jauh lebih kuat dan valid secara ilmiah.

Berikut adalah ringkasan dari penyesuaian yang akan kita lakukan dan justifikasi ilmiahnya:

### 1. Penyesuaian Arsitektur Model: Menjadikannya Fleksibel untuk N secrets
Masalah Saat Ini: Arsitektur Anda statis dengan num_secrets yang ditetapkan di awal (StegoEncoder(num_secrets=5)). Ini tidak bisa dilatih atau diuji dengan jumlah secret yang berbeda.

Solusi yang Diusulkan: Kita perlu memodifikasi model agar jumlah num_secrets bisa menjadi parameter input yang fleksibel. nn.ModuleList yang sudah ada sebenarnya sudah fleksibel, tetapi in_channels pada HidingNetwork saat ini statis. Kita akan membuat HidingNetwork dapat menerima jumlah channel yang bervariasi.

Justifikasi Ilmiah: Pendekatan ini memungkinkan kita untuk melakukan studi parametrik terhadap model. Kita dapat secara sistematis menguji bagaimana performa model (PSNR, kapasitas) berubah seiring dengan peningkatan jumlah secret images yang disisipkan. Ini akan membantu kita menemukan "titik batas" model Anda secara kuantitatif.

### 2. Penyesuaian Preprocessing dan Dataset: Menghindari Overfitting
Masalah Saat Ini: Penggunaan dataset pairing dalam format JSON (1 cover + 5 secrets) yang sudah ditetapkan dapat menyebabkan model menjadi terlalu spesifik pada pasangan-pasangan tersebut (overfitting). Model tidak belajar bagaimana menyembunyikan pesan secara umum, tetapi hanya belajar pola penyembunyian untuk kombinasi tertentu.

Solusi yang Diusulkan:
1. Struktur Dataset Fleksibel: Kita akan beralih ke struktur folder yang lebih umum:
- dataset/covers/: Berisi semua cover frame (448x448).
- dataset/secrets/: Berisi semua gambar rahasia (64x64).
2. Randomisasi Dinamis: Selama pelatihan, skrip akan secara acak memilih:
- Satu cover frame dari folder covers/.
- N secret images (di mana N adalah jumlah secret yang kita uji, misalnya N=1) dari folder secrets/.
3. Split Dataset: Memisahkan data menjadi tiga set:
- train/: Data untuk melatih model.
- val/: Data untuk validasi selama pelatihan (melihat overfitting).
- test/: Data yang belum pernah dilihat oleh model sama sekali untuk evaluasi akhir.

Justifikasi Ilmiah: Pendekatan ini akan memastikan bahwa model belajar pola penyembunyian dan ekstraksi yang lebih umum (generalisasi), bukan sekadar menghafal. Penggunaan dataset validasi dan tes akan memungkinkan kita untuk:
1. Mendeteksi overfitting: Dengan memplot grafik loss pelatihan dan validasi. Jika loss pelatihan terus turun tetapi loss validasi stagnan atau naik, itu adalah tanda overfitting.
2. Evaluasi yang Kredibel: Memberikan hasil evaluasi PSNR, kapasitas, dan waktu yang lebih andal karena dilakukan pada data yang benar-benar baru.


### 3. Penyesuaian Loss Function dan Normalisasi
Masalah Saat Ini: Model Anda saat ini hanya menggunakan loss function yang mengukur perbedaan nilai piksel (loss_cover) dan loss_secret. Ini mungkin tidak optimal. Selain itu, Anda belum menerapkan skema normalisasi khusus yang dirancang untuk steganografi.

Solusi yang Diusulkan:
1. Skema Normalisasi Khusus: Kita perlu menerapkan normalisasi untuk input ke model agar konsisten. Lebih penting lagi, output dari HidingNetwork (gambar stego) yang berada dalam rentang [-1, 1] atau [0, 1] perlu dinormalisasi dan diubah kembali ke rentang [0, 255] untuk disimpan dan diukur PSNR-nya.

2. Potensi Peningkatan Loss Function:
- loss_cover: MSE (Mean Squared Error) adalah standar, tapi ada opsi lain seperti L1 Loss (torch.nn.L1Loss) atau bahkan perceptual loss jika diperlukan, yang bisa menghasilkan gambar yang lebih perceptually pleasing. Untuk tujuan PSNR, MSE adalah yang paling relevan.
- Kita akan tetap menggunakan loss = alpha * loss_cover + beta * loss_secret namun akan mengeksplorasi nilai alpha dan beta yang berbeda untuk menyeimbangkan antara kualitas visual dan akurasi ekstraksi.

Justifikasi Ilmiah: Normalisasi yang tepat sangat penting untuk stabilitas pelatihan. Penyesuaian loss function dan bobot alpha/beta akan memungkinkan kita untuk secara eksplisit mengontrol trade-off antara kualitas visual stego (yang diukur oleh PSNR) dan akurasi ekstraksi (juga diukur oleh PSNR). Ini akan menjadi bagian penting dari eksperimen Anda untuk menemukan konfigurasi terbaik.