# Pastikan gdown terinstal. Jika belum, jalankan: !pip install gdown
import gdown
import os

# Meminta input dari pengguna untuk ID file Google Drive
google_drive_file_id = input("Masukkan ID File Google Drive: ")

# Meminta input dari pengguna untuk nama file output
output_filename = input("Masukkan nama file output (contoh: my_file.pkl): ")

# Memastikan ROOT_DIR sudah didefinisikan (diasumsikan dari setup notebook)
# Jika ROOT_DIR belum ada, definisikan di sini atau sesuaikan jalur
if 'ROOT_DIR' not in globals():
    ROOT_DIR = '/home/bimachasin86/VARX_REGRESION/' # Ganti dengan jalur root default Anda

# Jalur lengkap untuk menyimpan file
output_path = os.path.join(ROOT_DIR, output_filename)

try:
    print(f"\n\u23F3 Mengunduh file dengan ID: {google_drive_file_id} ke: {output_path}...")
    # Unduh file
    gdown.download(id=google_drive_file_id, output=output_path, quiet=False)
    print(f"\u2705 File berhasil diunduh ke: {output_path}")
except Exception as e:
    print(f"\u274c Gagal mengunduh file: {e}")
    print("Pastikan ID File Google Drive benar dan file memiliki izin akses (berbagi publik atau memiliki akses link).")
