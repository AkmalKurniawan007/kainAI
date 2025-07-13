import os
import requests
from flask import Flask, render_template, request, url_for, redirect
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import time
import json
import google.generativeai as genai

# --- KONFIGURASI ---
load_dotenv()
app = Flask(__name__)

# --- KONFIGURASI DUA API ---
# 1. Kunci API untuk Hugging Face (untuk menggambar)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# 2. Kunci API untuk Google (untuk membuat prompt)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Kamus Model Hugging Face (tetap sama)
MODEL_MAP_HF = {
    "stable-diffusion-xl": "stabilityai/stable-diffusion-xl-base-1.0",
    "stable-diffusion-1-5": "runwayml/stable-diffusion-v1-5",
    "realistic-vision": "SG161222/Realistic_Vision_V5.1_no_vae"
}

# Kamus Preset Gaya (tetap sama)
STYLE_PRESETS = {
    "klasik-sogan": {
        "nama": "Klasik Sogan",
        "gaya": "gaya batik sogan klasik khas keraton solo dan yogyakarta, sangat detail dan agung",
        "warna": "palet warna coklat sogan, krem gading, dan hitam pekat"
    },
    "mega-mendung": {
        "nama": "Mega Mendung Modern",
        "gaya": "gaya mega mendung cirebon dengan sentuhan modern dan garis yang bersih",
        "warna": "palet warna gradasi biru langit, biru tua, dan putih awan bersih"
    },
    "pesisir-ceria": {
        "nama": "Pesisir Ceria",
        "gaya": "gaya batik pesisiran yang ceria, berani, dan terinspirasi alam bahari seperti dari pekalongan",
        "warna": "palet warna-warni cerah seperti merah fanta, kuning kunyit, hijau toska, dan biru laut"
    }
}

# --- Fungsi-Fungsi Helper ---

def generate_enhanced_prompt_with_gemini(elemen, gaya, warna):
    """Menggunakan Gemini untuk membuat prompt yang menekankan elemen utama."""
    if not GOOGLE_API_KEY:
        return None, "Kunci API Google (GOOGLE_API_KEY) tidak ditemukan."
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # --- INSTRUKSI BARU YANG LEBIH TEGAS UNTUK GEMINI ---
        meta_prompt = (
            "Anda adalah seorang ahli penulis prompt untuk model AI text-to-image. "
            "Tugas Anda adalah membuat prompt bahasa Inggris yang sangat deskriptif untuk menghasilkan gambar motif batik Indonesia.\n\n"
            "**Instruksi Paling Penting:** Buatlah prompt yang memastikan elemen di bawah ini menjadi **subjek utama yang paling dominan, besar, dan jelas** dalam gambar. Elemen ini harus menjadi pusat perhatian mutlak.\n\n"
            "Detail untuk prompt:\n"
            f"- **Elemen Utama (Jadikan Fokus Utama):** {elemen}\n"
            f"- Gaya Batik: {gaya}\n"
            f"- Palet Warna: {warna}\n\n"
            "Struktur Prompt yang diinginkan:\n"
            "1. Mulai dengan kata kunci kualitas: 'masterpiece, best quality, 4k, ultra-detailed, intricate details'.\n"
            "2. Deskripsikan sebagai 'Indonesian batik seamless pattern'.\n"
            "3. **Tekankan dengan sangat jelas bahwa fokus utamanya adalah '{elemen}' yang besar dan mendominasi komposisi.** (Contoh: 'featuring a large, majestic, and dominant {elemen} as the central focus').\n"
            "4. Gabungkan dengan gaya dan palet warna yang telah ditentukan.\n"
            "5. Akhiri dengan kata kunci teknis: 'elegant composition, tileable, repeating pattern'.\n\n"
            "**Hanya berikan teks promptnya saja dalam satu paragraf tanpa penjelasan atau judul apa pun.**"
        )
        
        response = model.generate_content(meta_prompt)
        return response.text.strip(), None
    except Exception as e:
        return None, f"Terjadi kesalahan saat menghubungi Google Gemini API: {e}"


def generate_image_with_huggingface(model_id, prompt):
    """Mengirim permintaan ke model yang dipilih di Hugging Face."""
    if not HF_API_TOKEN:
        return None, "Token API Hugging Face (HF_API_TOKEN) tidak ditemukan."
        
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    payload = {"inputs": prompt}
    try:
        response = requests.post(api_url, headers=HF_HEADERS, json=payload, timeout=120) # Tambah timeout
        if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
            return response.content, None
        else:
            error_message = f"Error dari Hugging Face (Status: {response.status_code}). "
            try:
                error_details = response.json()
                if "is currently loading" in error_details.get("error", ""):
                    est_time = error_details.get('estimated_time', 20)
                    error_message += f"Model '{model_id}' sedang 'pemanasan'. Mohon tunggu {int(est_time)} detik lalu coba lagi."
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
    return redirect(url_for('create'))

@app.route("/create")
def create():
    return render_template("create.html", styles=STYLE_PRESETS, models=MODEL_MAP_HF)

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

    # Langkah 1: Buat prompt canggih menggunakan Gemini
    print(f"INFO: Meminta prompt dari Gemini untuk elemen '{elemen_input}'...")
    enhanced_prompt, error_gemini = generate_enhanced_prompt_with_gemini(elemen_input, preset["gaya"], preset["warna"])

    if error_gemini:
        print(f"ERROR: {error_gemini}")
        return render_template("results.html", error=error_gemini)
    
    print(f"INFO: Prompt dari Gemini: {enhanced_prompt}")

    # Langkah 2: Gunakan prompt dari Gemini untuk membuat gambar di Hugging Face
    print(f"INFO: Mengirim prompt ke Hugging Face model '{model_id}'...")
    image_data, error_hf = generate_image_with_huggingface(model_id, enhanced_prompt)
    
    if error_hf:
        print(f"ERROR: {error_hf}")
        return render_template("results.html", error=error_hf, elemen=elemen_input, enhanced_prompt=enhanced_prompt)

    # Langkah 3: Simpan dan tampilkan gambar
    try:
        image = Image.open(BytesIO(image_data))
        filename = f'output_{int(time.time())}.png'
        output_dir = os.path.join('static', 'images')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        image.save(output_path)
        
        image_url = url_for('static', filename=f'images/{filename}')
        print(f"INFO: Gambar berhasil dibuat di {output_path}")
        return render_template("results.html", image_url=image_url, elemen=elemen_input, enhanced_prompt=enhanced_prompt, error=None)
    except Exception as e:
        error_msg = f"Gagal menyimpan atau memproses gambar: {e}"
        print(f"ERROR: {error_msg}")
        return render_template("results.html", error=error_msg)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
