import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Jenis Serangan", layout="wide")
st.title("Dashboard Prediksi Jenis Serangan")

#kolom sesuai pipeline
expected = [
    'Source Port', 'Destination Port', 'Protocol', 'Packet Length',
    'Packet Type', 'Traffic Type', 'Anomaly Scores', 'Attack Signature',
    'Action Taken', 'Severity Level', 'Network Segment',
    'Geo-location Data', 'Log Source'
]

#model & encoder
@st.cache_resource #sekali load
def load_artifacts():
    pipe = joblib.load('model_pipeline.pkl')
    le   = joblib.load('label_encoder.pkl')
    return pipe, le

pipeline, le = load_artifacts()

st.subheader("Contoh Format CSV Input")
st.write(pd.DataFrame(columns=expected).head())

st.write("Silakan upload file CSV berisi data fitur serangan jaringan:")
uploaded = st.file_uploader("Upload CSV", type="csv")
if not uploaded:
    st.info("⬆ Silakan upload file CSV terlebih dahulu.")
    st.stop()

df = pd.read_csv(uploaded)
df.columns = df.columns.str.strip()

#valid kolom
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Kolom berikut tidak ditemukan: {missing}")
    st.stop()

#pred
X = df[expected]
with st.spinner("Melakukan prediksi…"):
    try:
        y_num = pipeline.predict(X)
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
        st.stop()

#inverse transform ke label asli
y_lbl = le.inverse_transform(y_num)
df['Predicted Attack Type'] = y_lbl

st.subheader("Tabel Hasil Prediksi")
st.dataframe(df, use_container_width=True)

st.subheader("Distribusi Jenis Serangan yang Diprediksi")
counts = df["Predicted Attack Type"].value_counts().sort_index()
st.bar_chart(counts, use_container_width=True)

top = counts.idxmax()
pct = counts.max() / counts.sum() * 100
st.markdown(
    f"Berdasarkan hasil prediksi, jenis serangan yang paling dominan adalah "
    f"**{top}**, sebanyak **{counts.max()} kali**, "
    f"atau sekitar **{pct:.2f}%** dari total **{counts.sum()}** entri data."
)

csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Hasil Prediksi",
    data=csv,
    file_name='prediksi_attack_type.csv',
    mime='text/csv'
)
