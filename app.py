import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Jenis Serangan", layout="wide")
st.title("Dashboard Prediksi Jenis Serangan")

#Fungsi untuk memuat model dan label encoder
@st.cache_resource
def load_artifacts():
    pipe = joblib.load('attack_type_pipeline.pkl')
    le   = joblib.load('label_encoder.pkl')
    return pipe, le

pipeline, le = load_artifacts()

#Up file
st.write("Silakan upload file CSV berisi data fitur serangan jaringan.")
uploaded = st.file_uploader("Upload CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()

    expected = [
        'Source Port', 'Destination Port', 'Protocol', 'Packet Length',
        'Packet Type', 'Traffic Type', 'Anomaly Scores', 'Attack Signature',
        'Action Taken', 'Severity Level', 'Network Segment',
        'Geo-location Data', 'Log Source'
    ]

    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.error(f"Kolom berikut tidak ditemukan dalam file CSV: {missing}")
    else:
        #Pred
        X = df[expected]
        y_num = pipeline.predict(X)
        y_lbl = le.inverse_transform(y_num)
        df['Predicted Attack Type'] = y_lbl

        #Hasil pred
        st.subheader("Tabel Hasil Prediksi")
        st.dataframe(df)

        # Visual
        st.subheader("Distribusi Jenis Serangan yang Diprediksi")
        counts = df["Predicted Attack Type"].value_counts().sort_index()
        st.bar_chart(counts, use_container_width=True)

        st.subheader("Kesimpulan")
        top = counts.idxmax()
        pct = counts.max() / counts.sum() * 100
        st.markdown(
            f"Berdasarkan hasil prediksi, jenis serangan yang paling dominan adalah "
            f"**{top}**, sebanyak **{counts.max()} kali**, "
            f"atau sekitar **{pct:.2f}%** dari total **{counts.sum()}** entri data."
        )
else:
    st.info("⬆Silakan upload file CSV terlebih dahulu untuk melihat hasil prediksi.")
