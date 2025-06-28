from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from scipy.stats import mode

app = Flask(__name__)

# -----------------------------
# Load model dan scaler/transformer
# -----------------------------
with open('model/knn_cluster_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
balltree = knn_model['tree']
y_train_cluster = np.array(knn_model['y_train'])

with open('model/power_transformer.pkl', 'rb') as f:
    object_transformer = pickle.load(f)
    pt = object_transformer['power_transformer']

with open('model/robust_scaler.pkl', 'rb') as f:
    object_scaler = pickle.load(f)
    scaler = object_scaler['scaler']

with open('model/lgbm_smote.pkl', 'rb') as f:
    lgbm_model = pickle.load(f)

selected_columns = [
    'wind_direction_10m_dominant',
    'shortwave_radiation_sum',
    'et0_fao_evapotranspiration',
    'daylight_duration',
    'sunshine_duration',
    'precipitation_sum',
    'rain_sum',
    'precipitation_hours',
    'jarak_ke_beban',
    'cluster'
]

yeojohnson_cols = ['precipitation_sum', 'rain_sum']

@app.route('/', methods=['GET'])
def dashboard():
    tanggal_str = request.args.get('tanggal')
    status_filter = request.args.get('status', 'semua')

    df = pd.read_csv('data/dataset_2025_final.csv')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    X_coords = np.radians(df[['latitude', 'longitude']].values)
    dist, ind = balltree.query(X_coords, k=3)
    predicted_cluster = mode(y_train_cluster[ind], axis=1)[0].flatten()
    df['cluster'] = predicted_cluster

    if tanggal_str:
        # Filter hanya data pada tanggal yang dipilih
        df = df[df['date'] == pd.to_datetime(tanggal_str)]
        
        # 🚨 Cek jika tidak ada data setelah filter
        if df.empty:
            return render_template(
                'dashboard.html',
                tanggal=tanggal_str,
                tanggal_terfilter="Tidak ada data",
                status=status_filter,
                jumlah_gangguan=0,
                locations=[]
            )
        
        df_selected = df[selected_columns].copy()
        df_selected[yeojohnson_cols] = pt.transform(df_selected[yeojohnson_cols])
        X_scaled = scaler.transform(df_selected)
        y_pred = lgbm_model.predict(X_scaled)
        df['Prediction'] = y_pred

        tanggal_terfilter = pd.to_datetime(tanggal_str).strftime('%d %B %Y')
    else:
        # Prediksi semua data dulu untuk cari bulan terbaru dengan gangguan
        df_selected_all = df[selected_columns].copy()
        df_selected_all[yeojohnson_cols] = pt.transform(df_selected_all[yeojohnson_cols])
        X_scaled_all = scaler.transform(df_selected_all)
        y_pred_all = lgbm_model.predict(X_scaled_all)
        df['Prediction'] = y_pred_all

        df_gangguan = df[df['Prediction'] == 1]
        if df_gangguan.empty:
            max_date = df['date'].max()
        else:
            max_date = df_gangguan['date'].max()

        max_year, max_month = max_date.year, max_date.month
        df = df[(df['date'].dt.year == max_year) & (df['date'].dt.month == max_month)]

        # 🚨 Cek jika tidak ada data setelah filter bulan
        if df.empty:
            return render_template(
                'dashboard.html',
                tanggal=None,
                tanggal_terfilter="Tidak ada data",
                status=status_filter,
                jumlah_gangguan=0,
                locations=[]
            )

        df_selected = df[selected_columns].copy()
        df_selected[yeojohnson_cols] = pt.transform(df_selected[yeojohnson_cols])
        X_scaled = scaler.transform(df_selected)
        y_pred = lgbm_model.predict(X_scaled)
        df['Prediction'] = y_pred

        tanggal_terfilter = max_date.strftime('%d %B %Y')

    if status_filter == 'gangguan':
        filtered_df = df[df['Prediction'] == 1]
    elif status_filter == 'normal':
        filtered_df = df[df['Prediction'] == 0]
    else:
        filtered_df = df

    jumlah_gangguan = filtered_df[filtered_df['Prediction'] == 1].shape[0]
    locations = filtered_df.to_dict(orient='records')

    return render_template(
        'dashboard.html',
        tanggal=tanggal_str,
        tanggal_terfilter=tanggal_terfilter,
        status=status_filter,
        jumlah_gangguan=jumlah_gangguan,
        locations=locations
    )

if __name__ == '__main__':
    app.run(debug=True)
