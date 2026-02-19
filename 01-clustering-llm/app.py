import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import google.generativeai as genai

# =============================================================================
# 1. CONFIGURACIÓN Y ESTILOS
# =============================================================================
st.set_page_config(page_title="AI Music Horoscope", page_icon="🔮", layout="wide")

st.markdown("""
<style>
    /* FONDO GRIS OSCURO */
    .stApp {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    
    /* Input de texto visible  */
    .stTextInput input {
        color: #00000 !important;
    }
    
    /* TITULOS */
    h1 {
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        text-align: center;
    }
    h2, h3 {
        color: #92FE9D !important;
    }
    
    /* TARJETAS DE CANCIONES (BUSCADOR) */
    .song-result {
        background-color: #000000;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #00C9FF;
        margin-bottom: 5px;
    }
    
    /* BOTONES */
    div.stButton > button {
        background-color: transparent;
        color: #92FE9D;
        border: 1px solid #92FE9D;
        border-radius: 20px;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #92FE9D;
        color: #1E1E1E;
        font-weight: bold;
        border-color: #92FE9D;
    }
    
    /* TEXTO DE BARRAS DE PROGRESO */
    .progress-label {
        font-weight: bold;
        color: #FFFFFF; 
        margin-bottom: 2px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. FUNCIONES CRÍTICAS DEL MODELO 
# =============================================================================

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
    X = df.copy()
    if 'time_signature' in X.columns:
        X = X.drop(columns=['time_signature'])
    return X

# =============================================================================
# 3. CARGA DE DATOS
# =============================================================================

CLUSTER_MAP = {
    0: "Aggressive / Heavy 🎸 (Intenso)",
    1: "Live Spirit 🎉 (Festivo)",
    2: "Pure Acoustic 🎻 (Orgánico)",
    3: "Radio Mainstream 📻 (Pop/Hits)",
    4: "Electronic Focus 🎹 (Sintético)",
    5: "Hidden Gems 💎 (Nicho)",
    6: "Downtempo 🌙 (Atmosférico)",
    7: "Urban / Spoken 🎤 (Rítmico)",
    8: "Vocal Acoustic 🗣️ (Romántico)"
}

@st.cache_resource
def load_resources():
    try:
        pipe = joblib.load('model_pipeline.pkl')
        pca = joblib.load('model_pca.pkl')
        gmm = joblib.load('model_gmm.pkl')
        cols = joblib.load('model_cols.pkl')
        db = pd.read_csv('songs_database.csv')
        return pipe, pca, gmm, cols, db
    except Exception as e:
        return None, None, None, None, None

pipeline, pca_model, gmm_model, pca_cols, songs_db = load_resources()

# =============================================================================
# 4. GESTIÓN DE SESIÓN Y SIDEBAR (RESTAURADO)
# =============================================================================

if 'playlist' not in st.session_state:
    st.session_state.playlist = []

def add_song(song_data):
    # Evita duplicados en la lista visual
    ids_actuales = [s['track_id'] for s in st.session_state.playlist]
    if song_data['track_id'] not in ids_actuales:
        st.session_state.playlist.append(song_data)

def custom_bar(label, value, color):
    # Barra HTML personalizada
    st.markdown(f"""
    <div style="margin-bottom:10px;">
        <div class="progress-label">{label} &nbsp; <span style="color:{color}">{int(value*100)}%</span></div>
        <div style="background-color:#440; height:8px; border-radius:4px; width:100%;">
            <div style="background-color:{color}; height:8px; border-radius:4px; width:{value*100}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR (PANEL DESLIZANTE) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # API Key status
    try:
        google_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ API Conectada")
    except FileNotFoundError:
        google_key = st.text_input("🔑 Gemini API Key", type="password")
        if google_key:
            st.success("✅ API Key introducida")
        else:
            st.warning("⚠️ Introduce tu API Key")
    
    st.divider()
    
    # Playlist en Sidebar
    st.subheader(f"🎵 Playlist ({len(st.session_state.playlist)})")
    
    if len(st.session_state.playlist) > 0:
        for s in st.session_state.playlist:
            st.caption(f"▫️ {s['track_name']}")
        
        st.write("") # Espacio
        if st.button("🗑️ Limpiar Todo"):
            st.session_state.playlist = []
            st.rerun()
    else:
        st.info("Añade canciones desde el buscador.")

# =============================================================================
# 5. INTERFAZ PRINCIPAL
# =============================================================================

st.title("🔮 Horóscopo Musical IA")
st.markdown("<p style='text-align:center; color:#AAA;'>Dime qué escuchas y los astros digitales te dirán tu destino.</p>", unsafe_allow_html=True)

if songs_db is None:
    st.error("🚨 Faltan los archivos del modelo. Asegúrate de haber ejecutado 'train_model.py'.")
    st.stop()

# --- BUSCADOR ---
c1, c2 = st.columns([3, 1])
with c1:
    search = st.text_input("🔍 Buscar canción:", placeholder="Ej: Bad Bunny, Queen...")

if search:
    mask = songs_db['track_name'].str.contains(search, case=False, na=False) | \
        songs_db['artists'].str.contains(search, case=False, na=False)
    results = songs_db[mask].head(5)
    
    if not results.empty:
        for i, (index, row) in enumerate(results.iterrows()):
            with st.container():
                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    st.markdown(f"""
                    <div class="song-result">
                        <b style="color:#000000">{row['track_name']}</b><br>
                        <span style="color:#440; font-size:0.9em">{row['artists']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with col_btn:
                    # Key única para evitar conflicto de botones
                    unique_key = f"btn_{row['track_id']}_{i}"
                    if st.button("Añadir", key=unique_key):
                        add_song(row.to_dict())
                        st.rerun()
    else:
        st.info("No encontré esa canción en la base de datos offline de la que dispongo :( ")

st.divider()

# --- BOTÓN DE ANÁLISIS ---
if len(st.session_state.playlist) > 0:
    if st.button("🔮 LEER MI FUTURO AHORA", use_container_width=True):
        if not google_key:
            st.warning("⚠️ Necesitas introducir tu API Key de Google en la barra lateral.")
            st.stop()
            
        # Barra de carga falsa
        progress_text = "Cargando..."
        bar = st.progress(0, text=progress_text)
        for p in range(100):
            time.sleep(0.01)
            bar.progress(p + 1, text="Analizando frecuencias...")
        bar.empty()

        # ---------------------------------------------------------------------
        # CORRECCIÓN DEL ERROR DE COLUMNAS FALTANTES
        # ---------------------------------------------------------------------
        # 1. Crear DataFrame de la playlist
        df_play = pd.DataFrame(st.session_state.playlist)
        
        # 2. Asegurarse de que explicit sea numérico para poder promediarlo (o sacar moda)
        if 'explicit' in df_play.columns:
            df_play['explicit'] = df_play['explicit'].astype(int)
            
        # 3. Calcular promedios solo de las numéricas
        numeric_cols = df_play.select_dtypes(include=np.number).columns
        avg_features = df_play[numeric_cols].mean().to_frame().T
        
        # 4. RELLENAR COLUMNAS FALTANTES (explicit, mode, time_signature, key)
        # El pipeline NECESITA estas columnas aunque no sean promedio exacto
        required_cols = ['explicit', 'mode', 'time_signature', 'key']
        
        for col in required_cols:
            if col in df_play.columns:
                # Usamos el valor más frecuente (moda) de la playlist para estos datos categóricos
                avg_features[col] = df_play[col].mode()[0]
            else:
                # Valor por defecto si por alguna razón no existiera en la DB
                avg_features[col] = 0 
        
        # ---------------------------------------------------------------------
        
        # Predicción
        try:
            df_scaled = pipeline.transform(avg_features)
            pca_data = pca_model.transform(df_scaled)
            n_comps = pca_model.n_components_
            df_pca = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(n_comps)])
            
            cluster_id = gmm_model.predict(df_pca[pca_cols].values)[0]
            cluster_name = CLUSTER_MAP.get(cluster_id, "Desconocido")
        except Exception as e:
            st.error(f"Error técnico en el modelo: {e}")
            st.stop()

        # Prompt para Gemini
        prompt = f"""
        INSTRUCCIONES:
        Eres un experto en psicología musical y crítica cultural.
        Explica qué dice esta canción sobre la personalidad de quien la escucha hoy.Sé ingenioso, usa metáforas y habla directamente al usuario.
        
        DATOS DEL USUARIO:
        - Aura Musical: {cluster_name}
        - Danceability: {avg_features['danceability'].iloc[0]:.2f}
        - Energy: {avg_features['energy'].iloc[0]:.2f}
        - Valence: {avg_features['valence'].iloc[0]:.2f}
        - Acousticness: {avg_features['acousticness'].iloc[0]:.2f}
        - Instrumentalness: {avg_features['instrumentalness']}
        
        TAREA: Explica qué dice esta canción sobre la personalidad de quien la escucha hoy y a que grupo social actual pertenece.
        TONO: Sarcástico, cruel, ingenioso,metafórico y directo al usuario.
        IDIOMA: Español.
        """

        # Mostrar Resultados
        c_izq, c_der = st.columns([1, 2])
        
        with c_izq:
            st.markdown(f"### 🧬 Tipo: **{cluster_name}**")
            st.markdown("---")
            # Barras personalizadas
            custom_bar("Energía ⚡", avg_features['energy'].iloc[0], "#FFD700")
            custom_bar("Bailabilidad 💃", avg_features['danceability'].iloc[0], "#FF00FF")
            custom_bar("Positividad 😊", avg_features['valence'].iloc[0], "#00FF00")
            custom_bar("Acústica 🎻", avg_features['acousticness'].iloc[0], "#00C9FF")

        with c_der:
            client = genai.Client(api_key=google_key)
            with st.spinner("Cargando..."):
                try:
                    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                    with st.container(border=True):
                        st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error conectando con Gemini: {e}")

else:
    st.info("👆 Usa el buscador para añadir canciones a tu lectura.")