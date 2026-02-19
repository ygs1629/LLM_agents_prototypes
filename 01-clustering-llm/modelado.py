import pandas as pd
import numpy as np
import os
import joblib
import toml
import spotipy
import kagglehub
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


# ---  Cargar Datos desde la API de Spotify ---
print("Conectando a la API de Spotify...")
secrets = toml.load("secrets.toml")
client_id = secrets["spotify"]["client_id"]
client_secret = secrets["spotify"]["client_secret"]

auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

def obtener_datos_spotify(sp, playlist_ids, target_size=80000):
    """
    Extrae canciones de Spotify. Si la API falla o extrae menos del objetivo,
    completa el dataset descargando y muestreando la base de datos de Kaggle.
    """
    tracks_data = []
    
    # Extracción desde la API
    for pid in playlist_ids:
        print(f"Consultando playlist: {pid}")
        try:
            results = sp.playlist_tracks(pid)
            tracks = results['items']

            while results['next']:
                results = sp.next(results)
                tracks.extend(results['items'])
                
            for item in tracks:
                track = item.get('track')
                if not track or track['id'] is None:
                    continue
                    
                track_info = {
                    'track_id': track['id'],
                    'artists': ", ".join([artist['name'] for artist in track['artists']]),
                    'album_name': track['album']['name'],
                    'track_name': track['name'],
                    'popularity': track['popularity'],
                    'explicit': track['explicit'] }
                tracks_data.append(track_info)
                
        except spotipy.exceptions.SpotifyException as e:
            print(f"  -> Playlist inaccesible ({e.http_status}). Saltando...")
            continue
        except Exception as e:
            print(f"  -> Error inesperado: {e}. Saltando...")
            continue

    # Convertir lo extraído a df
    df_dinamico = pd.DataFrame()
    if len(tracks_data) > 0:
        df_tracks = pd.DataFrame(tracks_data).drop_duplicates(subset=['track_id'])
        print(f"Obteniendo audio features para {len(df_tracks)} canciones nuevas...")
        
        audio_features_list = []
        track_ids = df_tracks['track_id'].tolist()
        
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i+100]
            try:
                features = sp.audio_features(batch)
                audio_features_list.extend([f for f in features if f is not None])
            except:
                continue
                
        if audio_features_list:
            df_features = pd.DataFrame(audio_features_list)
            df_dinamico = pd.merge(df_tracks, df_features, left_on='track_id', right_on='id', how='inner')
            # Limpiar duplicadas
            if 'id' in df_dinamico.columns:
                df_dinamico = df_dinamico.drop(columns=['id'])
            df_dinamico = df_dinamico.dropna()

    obtenidas_api = len(df_dinamico)
    faltantes = target_size - obtenidas_api

    # --- Fallback estático con Kaglle ---
    if faltantes > 0:
        print(f"\n--- FASE 2: Fallback Estático ---")
        print(f"Se obtuvieron {obtenidas_api} canciones de la API.")
        print(f"Conectando con Kaggle para rellenar las {faltantes} canciones faltantes")
        
        try:
            ruta_kaggle = kagglehub.dataset_download("thedevastator/spotify-tracks-genre-dataset")
            archivos_csv = [f for f in os.listdir(ruta_kaggle) if f.endswith('.csv')]
        
            if archivos_csv:
                ruta_csv = os.path.join(ruta_kaggle, archivos_csv[0])
                df_kaggle = pd.read_csv(ruta_csv, engine='python', on_bad_lines='skip')
                
                if not df_dinamico.empty:
                    df_kaggle = df_kaggle[~df_kaggle['track_id'].isin(df_dinamico['track_id'])]
                
                muestra_tamano = min(faltantes, len(df_kaggle))
                df_muestra = df_kaggle.sample(n=muestra_tamano, random_state=42)
                
                if not df_dinamico.empty:
                    df_final = pd.concat([df_dinamico, df_muestra], ignore_index=True)
                else:
                    df_final = df_muestra
                    
                print(f"Dataset listo con {len(df_final)} canciones para el GMM.")
                return df_final
        except Exception as e:
            print(f"Falló la conexión con Kaggle: {e}")
            print("Entrenando solo con los datos de la API")
            
    else:
        print(f"Se llegó al nº objetivo ({obtenidas_api}  de canciones) solo con la API")
        
    return df_dinamico

# Selección de playlists para el GMM 
# Basado en experimentos con la DB: https://www.kaggle.com/datasets/thedevastator/spotify-tracks-genre-dataset
playlists_entrenamiento = [
    "32NxKs6NTtf61tP8fOW106", # C0: Hype
    "3rbOxUsB1PKIB8AZ4dAZ82", # C1: Festividad
    "2fMhc2p5LEqEqJf2IyZmqS", # C2: Instrumental tranquila
    "37i9dQZF1DXcBWIGoYBM5M", # C3: Today's Top Hits
    "37i9dQZF1DXdDh4h59PJIQ", # C4: Electronic
    "51YToRmnFC3U4iPj6LgHgF", #  C5: Jazz n Soul
    "37i9dQZF1DWWQRwui0ExPn", # C6: lofi beats
    "37i9dQZF1DX58gKmCfKS2T", # C7: Rap n Dancehall
    "37i9dQZF1DWYMvTygsLWlG"  # C8: Baladas
]

df_raw = obtener_datos_spotify(sp, playlists_entrenamiento)
print(f"Dataset creado con {len(df_raw)} canciones.")


# --- Definición de funciones ---
def ingenieria_basica(df):
    X = df.copy()
    if 'explicit' in X.columns:
        X['explicit'] = X['explicit'].astype(int)
    if 'time_signature' in X.columns:
        X['is_time_signature_4'] = np.where(X['time_signature'] == 4, 1, 0)
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

# --- Construcción del pipeline ---

columnas_cuantitativas = ['popularity', 'duration_ms', 'danceability', 'energy', 
                        'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
                        'liveness', 'valence', 'tempo']
columnas_directas = ['explicit', 'mode', 'is_time_signature_4']

pipe_numerico = Pipeline([
    ('clipping', FunctionTransformer(clipear_outliers_especificos, validate=False)),
    ('yeo_johnson', PowerTransformer(method='yeo-johnson', standardize=True))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', pipe_numerico, columnas_cuantitativas),
        ('cat_key', OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore'), ['key']),
        ('pass', 'passthrough', columnas_directas)], verbose_feature_names_out=False).set_output(transform="pandas")

pipeline_global = Pipeline([
    ('ingenieria', FunctionTransformer(ingenieria_basica, validate=False)),
    ('selector', FunctionTransformer(seleccionar_columnas_modelo, validate=False)),
    ('transformacion', preprocessor)])


# --- Entrenamiento ---
if df_raw.empty:
    raise ValueError("El dataset está completamente vacío. No se puede entrenar el modelo.")

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


# --- 5. Guardar modelos y db en local ---

# Creacion si no existe de las carpetas
os.makedirs('models', exist_ok=True)
os.makedirs('db', exist_ok=True)

print("Guardando modelos y db")
joblib.dump(pipeline_global, os.path.join('models', 'model_pipeline.pkl'))
joblib.dump(pca, os.path.join('models', 'model_pca.pkl'))
joblib.dump(gmm_final, os.path.join('models', 'model_gmm.pkl'))
joblib.dump(cols_clustering, os.path.join('models', 'model_cols.pkl'))

etiquetas_cluster = gmm_final.predict(X_clustering)

cols_to_keep = ['track_id', 'artists', 'album_name', 'track_name', 'key'] + columnas_cuantitativas + ['explicit', 'mode', 'time_signature']
df_db = df_raw[cols_to_keep].copy()
df_db['cluster'] = etiquetas_cluster
df_db = df_db.drop_duplicates(subset=['artists', 'track_name'])

ruta_csv = os.path.join('db', 'songs_database.csv')
df_db.to_csv(ruta_csv, index=False)

print(f"Base de datos guardada exitosamente en '{ruta_csv}': {len(df_db)} canciones listas con sus clústeres asignados.")