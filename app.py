import os
import requests
from flask import Flask, render_template, request, url_for
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import time
import json

# --- KONFIGURASI ---
load_dotenv()
app = Flask(__name__)

# Konfigurasi HANYA untuk Hugging Face
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Kamus Model Hugging Face yang sudah kita verifikasi
MODEL_MAP_HF = {
    "stable-diffusion-xl": "stabilityai/stable-diffusion-xl-base-1.0",
    "stable-diffusion-1-5": "runwayml/stable-diffusion-v1-5",
    "realistic-vision": "SG161222/Realistic_Vision_V5.1_no_vae"
}

# Kamus Preset Gaya
STYLE_PRESETS = {
    "klasik-sogan": {
        "gaya": "gaya batik sogan klasik khas keraton solo dan yogyakarta, sangat detail dan agung",
        "warna": "palet warna coklat sogan, krem gading, dan hitam pekat"
    },
    "mega-mendung": {
        "gaya": "gaya mega mendung cirebon dengan sentuhan modern dan garis yang bersih",
        "warna": "palet warna gradasi biru langit, biru tua, dan putih awan bersih"
    },
    "pesisir-ceria": {
        "gaya": "gaya batik pesisiran yang ceria, berani, dan terinspirasi alam bahari seperti dari pekalongan",
        "warna": "palet warna-warni cerah seperti merah fanta, kuning kunyit, hijau toska, dan biru laut"
    }
}


# --- Fungsi-Fungsi Helper ---

def construct_prompt(elemen, gaya, warna):
    """Membangun prompt teks sederhana namun efektif."""
    # Menambahkan lebih banyak "kata kunci ajaib" untuk hasil yang lebih baik
    return (f"masterpiece, best quality, 4k, high resolution, seamless, tileable pattern, "
            f"Indonesian batik, in the style of {gaya}, featuring {elemen}, "
            f"with a {warna} color palette. Intricate details, elegant composition, vector art.")

def generate_image_with_huggingface(model_id, prompt):
    """Mengirim permintaan ke model yang dipilih di Hugging Face."""
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    payload = {"inputs": prompt}
    try:
        response = requests.post(api_url, headers=HF_HEADERS, json=payload)
        if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
            return response.content, None
        else:
            error_message = f"Error dari Hugging Face (Status: {response.status_code}). "
            try:
                # Coba dapatkan pesan error spesifik dari server
                error_details = response.json()
                if "is currently loading" in error_details.get("error", ""):
                    error_message += f"Model '{model_id}' sedang 'pemanasan'. Mohon tunggu 1-2 menit lalu coba lagi."
                else:
                    error_message += error_details.get("error", response.text)
            except json.JSONDecodeError:
                error_message += f"Respons mentah: {response.text}"
            return None, error_message
    except Exception as e:
        return None, f"Terjadi kesalahan saat menghubungi Hugging Face: {e}"

# --- Rute-Rute Aplikasi ---

@app.route("/")
def index():
    # Menggunakan create.html sebagai halaman utama untuk menyederhanakan
    return render_template("create.html")

@app.route("/create")
def create():
    return render_template("create.html")

@app.route("/generate", methods=["POST"])
def generate():
    elemen_input = request.form.get("elemen")
    style_choice = request.form.get("style_choice")
    model_choice = request.form.get("model_choice")

    if not all([elemen_input, style_choice, model_choice]):
        return render_template("results.html", error="Input tidak lengkap. Harap isi semua kolom.")
    
    preset = STYLE_PRESETS.get(style_choice)
    model_id = MODEL_MAP_HF.get(model_choice)
    if not preset or not model_id:
        return render_template("results.html", error="Pilihan gaya atau model tidak valid.")

    gaya_input = preset["gaya"]
    warna_input = preset["warna"]
    prompt = construct_prompt(elemen_input, gaya_input, warna_input)
    print(f"INFO: Mengirim prompt ke model '{model_id}': {prompt}")

    image_data, error_message = generate_image_with_huggingface(model_id, prompt)
    
    if error_message:
        print(f"ERROR: {error_message}")
        return render_template("results.html", error=error_message)

    try:
        image = Image.open(BytesIO(image_data))
        timestamp = int(time.time())
        filename = f'output_{timestamp}.png'
        output_path = os.path.join('static', 'images', filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        
        image_url = url_for('static', filename=f'images/{filename}')
        print(f"INFO: Gambar berhasil dibuat di {output_path}")
        return render_template("results.html", image_url=image_url, error=None)
    except Exception as e:
        print(f"ERROR: Gagal menyimpan atau memproses gambar: {e}")
        return render_template("results.html", error="Terjadi kesalahan saat memproses gambar.")

if __name__ == "__main__":
    app.run(debug=True)