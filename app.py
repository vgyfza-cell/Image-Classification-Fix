import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Mengatur konfigurasi halaman web Streamlit
st.set_page_config(page_title="Aplikasi Klasifikasi Gambar", layout="centered")
st.title("🐾 Aplikasi Klasifikasi Gambar (Custom Model .h5)")
st.write("Silakan unggah model `.h5` Anda dan gambar hewan yang ingin diklasifikasikan.")

# --- BAGIAN 1: UPLOAD MODEL (.h5) ---
st.header("1. Upload Model Klasifikasi")
model_file = st.file_uploader("Pilih file model dengan format (.h5)", type=["h5"])

# Menggunakan Session State agar model tidak perlu di-load ulang setiap kali user klik tombol
if 'model' not in st.session_state:
    st.session_state.model = None

if model_file is not None:
    try:
        # Menyimpan file h5 sementara di server Streamlit
        with open("temp_model.h5", "wb") as f:
            f.write(model_file.getbuffer())
        
        # Memuat model menggunakan TensorFlow/Keras
        st.session_state.model = tf.keras.models.load_model("temp_model.h5")
        st.success("🎉 Model .h5 Anda berhasil dimuat!")
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file .h5 Anda valid. Error: {e}")

# --- BAGIAN 2: UPLOAD GAMBAR UNTUK DIUJI ---
st.header("2. Upload Gambar Hewan")
image_file = st.file_uploader("Pilih gambar yang ingin diuji (Kucing, Ikan, dll.)", type=["jpg", "jpeg", "png"])

if image_file is not None:
    # Menampilkan gambar yang diunggah oleh user
    image = Image.open(image_file)
    st.image(image, caption='Gambar yang akan diuji', use_column_width=True)
    
    # Tombol untuk memicu proses prediksi/klasifikasi
    if st.button("Klasifikasikan Gambar Sekarang"):
        if st.session_state.model is None:
            st.error("Wajib mengunggah model `.h5` terlebih dahulu di langkah 1!")
        else:
            with st.spinner('Sedang menganalisis gambar...'):
                try:
                    # --- PREPROCESSING GAMBAR ---
                    # PENTING: Ubah (224, 224) ini sesuai dengan ukuran input model kamu saat training dahulu
                    target_size = (224, 224) 
                    
                    # Mengubah ukuran gambar & memastikan format warnanya RGB
                    img_resized = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
                    img_array = np.asarray(img_resized)
                    
                    # Normalisasi nilai pixel jika model kamu ditraining dengan skala [0,1]
                    img_scaled = img_array.astype(np.float32) / 255.0
                    
                    # Menambah dimensi batch (contoh: dari [224, 224, 3] menjadi [1, 224, 224, 3])
                    img_reshape = np.expand_dims(img_scaled, axis=0)
                    
                    # --- PROSES PREDIKSI ---
                    predictions = st.session_state.model.predict(img_reshape)
                    
                    st.write("### Hasil Klasifikasi:")
                    
                    # Cek jika model berbentuk Binary Classification (Hanya ada 1 output node)
                    if predictions.shape[1] == 1:
                        score = predictions[0][0]
                        if score > 0.5:
                            st.info(f"Hasil: **Kelas 1 (Misal: Kucing)** dengan Akurasi {score:.2%}")
                        else:
                            st.info(f"Hasil: **Kelas 0 (Misal: Ikan)** dengan Akurasi {(1-score):.2%}")
                    
                    # Cek jika model berbentuk Multi-class Classification (Output node lebih dari 1)
                    else:
                        class_index = np.argmax(predictions)
                        confidence = predictions[0][class_index]
                        st.success(f"Hasil: **Model memprediksi gambar ini adalah indeks kelas ke-{class_index}**")
                        st.info(f"Tingkat Keyakinan (Confidence Score): {confidence:.2%}")
                        
                        # Menampilkan grafik probabilitas untuk semua kelas hewan
                        st.write("Detail Probabilitas Semua Kelas:")
                        st.bar_chart(predictions[0])
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

print("\n=== File app.py berhasil dibuat! ===")

# SCRIPT OTOMATIS UNTUK DOWNLOAD FILE LANGSUNG KE KOMPUTER KAMU
from google.colab import files
files.download('app.py')
