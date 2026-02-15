import os
import subprocess
import sys

# ===== KONFIGURASI =====
TARGET_DIR = "/home/bimachasin86/VARX_REGRESION/"

FILES = {
    "18W02IYbmM8lmPKH1b3PSVzWjcoNhcl6T": "fitted_models.pkl",
    "1_KEgF5kcuDA1q0raGpg3hE9uVxa4owCX": "colab_ngrok_url.txt",
    "1j6riUUZT0TiIxOEtOACqKSSJROfb8RYD": "final_fred_data.pkl",
    "1sa6gdcq9fqIj6dZd_xWiSFsPS8efN1k7" : "restored_forecasts.pkl",
}

# ===== PASTIKAN gdown TERINSTALL =====
try:
    import gdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "gdown"])
    import gdown

# ===== BUAT FOLDER JIKA BELUM ADA =====
os.makedirs(TARGET_DIR, exist_ok=True)

# ===== DOWNLOAD & OVERWRITE =====
for file_id, filename in FILES.items():
    # Tangani ID duplikat
    real_id = file_id.replace("_dup", "")
    output_path = os.path.join(TARGET_DIR, filename)

    print(f"Downloading {real_id} -> {output_path}")
    gdown.download(
        id=real_id,
        output=output_path,
        quiet=False
    )

print("✅ Semua file berhasil di-download dan ditimpa.")

