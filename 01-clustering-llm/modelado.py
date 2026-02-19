import pandas as pd
import numpy as np
import kagglehub
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# --- 1. Cargar Datos ---
print("Descargando dataset...")
path = kagglehub.dataset_download("thedevastator/spotify-tracks-genre-dataset")
csv_path = os.path.join(path, "train.csv")
df_raw = pd.read_csv(csv_path).dropna()

# --- 2. Definición de Funciones ---
def ingenieria_basica(df):
    X = df.copy()
    if 'explicit' in X.columns:
        X['explicit'] = X['explicit'].astype(int)
    if 'time_signature' in X.columns:
        X['is_time_signature_4'] = np.where(X['time_signature'] == 4, 1, 0)
        # No hacemos drop aquí para no perder info en el DB local, lo haremos en el pipeline
    if 'mode' in X.columns:
        X['mode'] = X['mode'].astype(int)
    return X

def clipear_outliers_especificos(df):
    X = df.copy()
    cols_to_clip = ['duration_ms', 'speechiness']
    for col in cols_to_clip:
        if col in X.columns:
            limite = X[col].quantile(0.99)
            X[col] = X[col].clip(upper=limite)
    return X

def seleccionar_columnas_modelo(df):
    """Limpia columnas que no usa el modelo antes de pasar al transformador"""
    X = df.copy()
    if 'time_signature' in X.columns:
        X = X.drop(columns=['time_signature'])
    return X

# --- 3. Construcción del Pipeline ---
columnas_cuantitativas = ['popularity', 'duration_ms', 'danceability', 'energy', 
                          'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
                          'liveness', 'valence', 'tempo']
columnas_directas = ['explicit', 'mode', 'is_time_signature_4']

pipe_numerico = Pipeline([
    ('clipping', FunctionTransformer(clipear_outliers_especificos, validate=False)),
    ('yeo_johnson', PowerTransformer(method='yeo-johnson', standardize=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', pipe_numerico, columnas_cuantitativas),
        ('cat_key', OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore'), ['key']),
        ('pass', 'passthrough', columnas_directas)
    ], verbose_feature_names_out=False
).set_output(transform="pandas")

pipeline_global = Pipeline([
    ('ingenieria', FunctionTransformer(ingenieria_basica, validate=False)),
    ('selector', FunctionTransformer(seleccionar_columnas_modelo, validate=False)), # Paso extra de limpieza
    ('transformacion', preprocessor)
])

# --- 4. Entrenamiento ---
print("Entrenando Pipeline Global...")
df_scaled = pipeline_global.fit_transform(df_raw)

print("Entrenando PCA...")
n_pca_components = 15 
pca = PCA(n_components=n_pca_components, random_state=42)
pca.fit(df_scaled)
pca_data = pca.transform(df_scaled)

# Dataframe PCA
pca_cols = [f'PC{i+1}' for i in range(n_pca_components)]
df_pca = pd.DataFrame(pca_data, columns=pca_cols)

# Seleccionamos componentes para GMM (12)
cols_clustering = [f'PC{i+1}' for i in range(12)] 
X_clustering = df_pca[cols_clustering].values

print("Entrenando GMM (K=9)...")
gmm_final = GaussianMixture(n_components=9, covariance_type='full', random_state=42)
gmm_final.fit(X_clustering)

# --- 5. Guardar Modelos y BASE DE DATOS LOCAL ---
print("Guardando modelos...")
joblib.dump(pipeline_global, 'model_pipeline.pkl')
joblib.dump(pca, 'model_pca.pkl')
joblib.dump(gmm_final, 'model_gmm.pkl')
joblib.dump(cols_clustering, 'model_cols.pkl')

print("Generando base de datos ligera para la App...")
# Guardamos solo lo necesario para identificar la canción y predecir
cols_to_keep = ['track_id', 'artists', 'album_name', 'track_name', 'key'] + columnas_cuantitativas + ['explicit', 'mode', 'time_signature']
df_db = df_raw[cols_to_keep].copy()

# Limpiamos duplicados (misma canción en varios álbumes) para ahorrar espacio
df_db = df_db.drop_duplicates(subset=['artists', 'track_name'])

# Guardamos como CSV comprimido para GitHub (Github tiene limite de 100MB)
df_db.to_csv('songs_database.csv', index=False)
print(f"Base de datos guardada: {len(df_db)} canciones listas.")