import os
import requests
import base64
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

# Muat environment variable (pastikan file .env ada)
load_dotenv()

# Konfigurasi
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def test_model():
    """Fungsi untuk mengirim satu permintaan tes ke model SDXL."""
    prompt = "A majestic dragon, Indonesian batik style, seamless pattern"
    payload = {"inputs": prompt}
    
    print(f"Mengirim permintaan ke model: {MODEL_ID}...")
    
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        
        print(f"Server merespons dengan status: {response.status_code}")

        if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
            print("Berhasil! Server mengembalikan gambar.")
            image = Image.open(BytesIO(response.content))
            image.save("test_output.jpg")
            print("Gambar tes berhasil disimpan sebagai 'test_output.jpg'")
        else:
            print("Gagal! Respons dari server:")
            # Coba print sebagai teks, mungkin ada pesan error
            print(response.text)
            
    except Exception as e:
        print(f"Terjadi error saat mencoba menghubungi API: {e}")

if __name__ == "__main__":
    test_model()