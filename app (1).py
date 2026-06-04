import streamlit as st
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Klasifikasi Ringan", page_icon="🐾")
st.title("🐟 Klasifikasi Ikan vs Kucing (Versi Ringan) 🐈")

# --- UPLOAD MODEL .TFLITE ---
st.sidebar.header("Pengaturan Model")
uploaded_model = st.sidebar.file_uploader("1. Upload Model (.tflite)", type=["tflite"])

# --- UPLOAD GAMBAR ---
uploaded_image = st.file_uploader("2. Upload Gambar", type=["jpg", "png", "jpeg"])

if uploaded_model is not None:
    # Load TFLite Model secara dinamis dari file uploader
    model_bytes = uploaded_model.read()
    
    try:
        # Inisialisasi Interpreter TFLite
        interpreter = np.ctypeslib.load_library if hasattr(np, 'ctypeslib') else None # Cek dependensi internal
        interpreter = np.core.multiarray._multiarray_umath if hasattr(np, 'core') else None
        
        # Cara standar streamlit load bytes tflite
        import platform
        # Untuk keandalan runtime di cloud:
        with open("temp_model.tflite", "wb") as f:
            f.write(model_bytes)
            
        import os
        # Kita gunakan binding tflite runtime dari standar numpy/python jika memungkinkan
        # Namun alternatif paling aman tanpa tensorflow adalah menggunakan tflite_runtime
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            from tensorflow import lite as tflite # fallback jika dijalankan lokal
            
        interpreter = tflite.Interpreter(model_path="temp_model.tflite")
        interpreter.allocate_tensors()
        
        st.sidebar.success("Model .tflite Berhasil Dimuat!")
        
        # Ambil detail input dan output model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Gambar yang diupload', width=300)
            
            if st.button("Klasifikasikan Gambar", type="primary"):
                with st.spinner('Menganalisis...'):
                    # Preprocessing sesuai input model (150x150)
                    size = (150, 150)
                    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                    image_array = np.asarray(image).astype(np.float32) / 255.0
                    image_array = np.expand_dims(image_array, axis=0)
                    
                    # Jalankan Prediksi dengan TFLite
                    interpreter.set_tensor(input_details[0]['index'], image_array)
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details[0]['index'])
                    
                    # Logika hasil
                    hasil = 1 if prediction[0][0] > 0.5 else 0
                    class_names = ["Ikan", "Kucing"]
                    
                    st.write("---")
                    st.subheader(f"Hasil Prediksi: **{class_names[hasil]}**")
                    st.write(f"Skor Model: {prediction[0][0]:.4f}")
                    
    except Exception as e:
        st.error(f"Gagal memuat interpreter TFLite. Tambahkan 'tflite-runtime' ke requirements.txt jika di cloud.")
else:
    st.info("Silahkan upload file model (.tflite) di sidebar untuk memulai.")
