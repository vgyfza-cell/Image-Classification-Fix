import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tflite_runtime.interpreter as tflite
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klasifikasi Ringan TFLite", page_icon="🐾")

st.title("🐟 Klasifikasi Ikan vs Kucing (Versi Ringan) 🐈")
st.write("Aplikasi ini menggunakan format .tflite yang super cepat dan ringan!")

# --- SIDEBAR: UPLOAD MODEL ---
st.sidebar.header("Pengaturan Model")
uploaded_model = st.sidebar.file_uploader("1. Upload Model (.tflite)", type=["tflite"])

# --- HALAMAN UTAMA: UPLOAD GAMBAR ---
uploaded_image = st.file_uploader("2. Upload Gambar (Ikan/Kucing)", type=["jpg", "png", "jpeg"])

if uploaded_model is not None:
    # Simpan file model sementara ke local disk server Streamlit
    with open("temp_model.tflite", "wb") as f:
        f.write(uploaded_model.read())
        
    try:
        # Load model menggunakan tflite_runtime
        interpreter = tflite.Interpreter(model_path="temp_model.tflite")
        interpreter.allocate_tensors()
        st.sidebar.success("Model .tflite Berhasil Dimuat!")
        
        # Ambil informasi struktur input dan output model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        if uploaded_image is not None:
            # Tampilkan Gambar
            image = Image.open(uploaded_image)
            st.image(image, caption='Gambar yang diupload', width=300)
            
            # Tombol Eksekusi Prediksi
            if st.button("Jalankan Klasifikasi", type="primary"):
                with st.spinner('Menganalisis gambar...'):
                    # Preprocessing: Ubah ukuran ke 150x150 dan normalisasi (bagi 255.0)
                    size = (150, 150)
                    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                    image_array = np.asarray(image_resized).astype(np.float32) / 255.0
                    image_array = np.expand_dims(image_array, axis=0) # Ubah bentuk jadi (1, 150, 150, 3)
                    
                    # Proses Prediksi via TFLite Interpreter
                    interpreter.set_tensor(input_details[0]['index'], image_array)
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details[0]['index'])
                    
                    # Logika Klasifikasi (Output Sigmoid tunggal)
                    # Jika nilai > 0.5 maka Kucing (1), jika kurang maka Ikan (0)
                    hasil = 1 if prediction[0][0] > 0.5 else 0
                    class_names = ["Ikan", "Kucing"]
                    
                    st.write("---")
                    st.subheader(f"Hasil Prediksi: **{class_names[hasil]}**")
                    st.write(f"Skor Keyakinan Model: {prediction[0][0]:.4f}")
                    
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
else:
    st.info("Silahkan upload file model (.tflite) kamu di sidebar sebelah kiri untuk memulai.")
