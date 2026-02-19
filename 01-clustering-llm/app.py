import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import time
import google.generativeai as genai

# =============================================================================
# 1. CONFIGURACIÓN Y ESTILOS
# =============================================================================
st.set_page_config(page_title="AI Musical Psychoanalyst", page_icon="🔮", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* FONDO MÍSTICO (Nebulosa oscura) */
    .stApp {
        background: radial-gradient(circle at top, #1E152A 0%, #0A0A0A 80%);
        color: #E0E0E0;
    }
    
    /* TITULOS PRINCIPALES */
    h1 {
        background: linear-gradient(45deg, #FF00FF, #00C9FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding-bottom: 0.5rem;
    }
    
    /* SUBTITULOS */
    h2, h3 {
        color: #00C9FF !important;
        font-weight: 300 !important;
    }
    
    /* TARJETAS DE CANCIONES (Efecto Glassmorphism) */
    .song-result {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 4px solid #FF00FF;
        transition: transform 0.2s ease, background 0.2s ease;
    }
    .song-result:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateX(5px);
    }
    
    /* TEXTO DE LAS BARRAS DE PROGRESO */
    .progress-label {
        font-weight: 600;
        color: #FFFFFF; 
        margin-bottom: 5px;
        font-size: 0.95rem;
        letter-spacing: 1px;
    }
    
    /* CONTENEDOR DE LA LECTURA DEL HORÓSCOPO */
    .reading-container {
        background: rgba(20, 10, 30, 0.6);
        border: 1px solid #FF00FF;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.1);
        font-size: 1.1rem;
        line-height: 1.6;
    }

    /* ==========================================================
        COLORES DE LOS BOTONES AÑADIR/BORRAR SIN RATÓN
       ========================================================== */
    div[data-testid="stButton"] > button[kind="secondary"] {
        background-color: rgba(20, 10, 30, 0.8) !important;
        color: #00C9FF !important;
        border: 1px solid #00C9FF !important;
    }
    
    /* ==========================================================
        ÚNICO CAMBIO: Texto de la barra de carga en blanco
       ========================================================== */
    [data-testid="stProgress"] p, 
    [data-testid="stProgress"] span, 
    [data-testid="stProgress"] div {
        color: #FFFFFF !important;
        font-weight: bold !important;
    }
    
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        background-color: transparent !important;
        color: #777777 !important; 
        border: 1px solid #555555 !important;
    }
    
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        color: #ff4b4b !important;
        border-color: #ff4b4b !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. FUNCIONES CRÍTICAS DEL MODELO (INTACTAS)
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
# 3. CARGA DE DATOS (INTACTA)
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
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ruta_pipe = os.path.join(base_dir, 'models', 'model_pipeline.pkl')
        ruta_pca = os.path.join(base_dir, 'models', 'model_pca.pkl')
        ruta_gmm = os.path.join(base_dir, 'models', 'model_gmm.pkl')
        ruta_cols = os.path.join(base_dir, 'models', 'model_cols.pkl')
        ruta_db = os.path.join(base_dir, 'db', 'songs_database.csv')
        
        pipe = joblib.load(ruta_pipe)
        pca = joblib.load(ruta_pca)
        gmm = joblib.load(ruta_gmm)
        cols = joblib.load(ruta_cols)
        songs_db = pd.read_csv(ruta_db)
        
        return pipe, pca, gmm, cols, songs_db
        
    except Exception as e:
        st.error(f"Error cargando archivos: {e}")
        return None, None, None, None, None

# =============================================================================
# 4. GESTIÓN DE SESIÓN Y COMPONENTES VISUALES
# =============================================================================

if 'playlist' not in st.session_state:
    st.session_state.playlist = []

def add_song(song_data):
    ids_actuales = [s['track_id'] for s in st.session_state.playlist]
    if song_data['track_id'] not in ids_actuales:
        st.session_state.playlist.append(song_data)

def custom_bar(label, value, color):
    # Barra rediseñada con brillo neón
    st.markdown(f"""
    <div style="margin-bottom: 18px;">
        <div class="progress-label">{label} <span style="float:right; color:{color}; text-shadow: 0 0 5px {color};">{int(value*100)}%</span></div>
        <div style="background-color: rgba(255,255,255,0.1); height: 10px; border-radius: 5px; width: 100%; overflow: hidden;">
            <div style="background-color: {color}; height: 10px; border-radius: 5px; width: {value*100}%; box-shadow: 0 0 10px {color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR (PANEL DESLIZANTE) ---
with st.sidebar:
    st.header("⚙️ Configuración")

    google_key = st.text_input("🔑 Gemini API Key", type="password", help="Necesaria para el análisis.")
    if google_key:
        st.success("Conexión Establecida")
    else:
        st.warning("⚠️ Introduce tu API Key para continuar")
    
    st.divider()
    st.subheader(f"🎵 Tu lista: ({len(st.session_state.playlist)} tracks)")
    
    if len(st.session_state.playlist) > 0:
        with st.container(border=True):
            for i, s in enumerate(st.session_state.playlist):
                st.markdown(f"<span style='color:black;'>**{i+1}.** {s['track_name']}</span> <br><span style='color:black; font-size:0.8em;'>{s['artists']}</span>", unsafe_allow_html=True)
                st.markdown("---")
        
        # Botón sutil abajo del todo
        if st.button("🗑️ Borrar lista", use_container_width=True):
            st.session_state.playlist = []
            st.rerun()
    else:
        st.info("El silencio absoluto...")


# =============================================================================
# 5. INTERFAZ PRINCIPAL
# =============================================================================

st.title("𝚿 IA psicológica musical 𝚿 ")
st.markdown("<h4 style='text-align:center; color:#AAA; font-weight: 300; margin-bottom: 2rem;'>Dime qué escuchas y te diré (o adivinaré más bien) quién eres.</h4>", unsafe_allow_html=True)

if songs_db is None:
    st.error("🚨 Faltan los archivos del modelo. Asegúrate de haber ejecutado 'modelado.py'.")
    st.stop()

# --- BUSCADOR OPTIMIZADO ---

with st.container(border=True):
    c1, c2, c3 = st.columns([1, 6, 1]) # Centrado
    with c2:
        search = st.text_input("🔍 Escibe una canción o artista:.", placeholder="Buscador case sensitive. Ej: Bad Bunny, Queen, Rosalía...", label_visibility="collapsed")

# =========================================================
# BOTÓN DE ANÁLISIS
# =========================================================
iniciar_lectura = False
if len(st.session_state.playlist) > 0:
    st.markdown("<br>", unsafe_allow_html=True)
    col_space1, col_btn, col_space2 = st.columns([1, 2, 1])
    with col_btn:
        iniciar_lectura = st.button("LECTURA PROFUNDA", type="primary", use_container_width=True)

# RESULTADOS QUE SE DESPLIEGAN Y SE OCULTAN
if search and not iniciar_lectura:
    mask = songs_db['track_name'].str.contains(search, case=False, na=False) | \
            songs_db['artists'].str.contains(search, case=False, na=False)
    results = songs_db[mask].head(5)
    
    if not results.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        for i, (index, row) in enumerate(results.iterrows()):
            col_info, col_btn = st.columns([5, 1], vertical_alignment="center")
            with col_info:
                st.markdown(f"""
                <div class="song-result">
                    <span style="font-size: 1.1em; color: white; font-weight: bold;">{row['track_name']}</span><br>
                    <span style="color: #00C9FF; font-size: 0.9em;">🎤 {row['artists']} &nbsp; | &nbsp; 💿 {row['album_name']}</span>
                </div>
                """, unsafe_allow_html=True)
            with col_btn:
                if st.button("➕ Añadir", key=f"btn_{row['track_id']}_{i}", use_container_width=True):
                    add_song(row.to_dict())
                    st.toast(f"¡{row['track_name']} añadida al ritual!")
                    st.rerun()
    else:
        st.warning("""No tenemos esa canción en la base de datos :(
                    \n✨¡Agredecemos la colaboración (nula) de Spotify por romper (intenciondamente) su API y tener una app con (cero) dinamismo!✨""")

st.markdown("<br><br>", unsafe_allow_html=True)

# --- EJECUCIÓN DEL ANÁLISIS ---
if iniciar_lectura:
    if not google_key:
        st.error("⚠️ Necesitas introducir tu API Key en la barra lateral.")
        st.stop()
        
    # Barra de carga inmersiva
    bar = st.progress(0, text="Pensando...")
    for p in range(100):
        time.sleep(0.015)
        if p > 0 : bar.progress(p, text="Extrayendo conclusiones...")
        elif p > 15: bar.progress(p, text="Revisando...")
        else: bar.progress(p + 1)
    bar.empty()

    st.markdown("<div style='text-align:center; padding: 12px; background-color: rgba(0, 201, 255, 0.15); border-radius: 8px; border: 1px solid #00C9FF; margin-bottom: 20px; color: white; font-size: 1.1em;'> <b>¡Desliza hacia abajo para ver tu resultado!</b> 👇</div>", unsafe_allow_html=True)
    
    # ---------------------------------------------------------------------
    # LÓGICA DE DATOS
    # ---------------------------------------------------------------------
    df_play = pd.DataFrame(st.session_state.playlist)
    
    if 'explicit' in df_play.columns:
        df_play['explicit'] = df_play['explicit'].astype(int)
        
    numeric_cols = df_play.select_dtypes(include=np.number).columns
    avg_features = df_play[numeric_cols].mean().to_frame().T
    
    required_cols = ['explicit', 'mode', 'time_signature', 'key']
    for col in required_cols:
        if col in df_play.columns:
            avg_features[col] = df_play[col].mode()[0]
        else:
            avg_features[col] = 0 
        
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

    # ---------------------------------------------------------------------
    # CONEXIÓN CON GEMINI
    # ---------------------------------------------------------------------
    prompt = f"""
        INSTRUCCIONES:
        Eres un experto en psicología musical y crítica cultural implacable.
        Tu objetivo es destrozar y analizar la personalidad de quien escucha esta música hoy. Sé ingenioso, usa metáforas y habla directamente al usuario.
        
        CONTEXTO OCULTO (Úsalo para tu análisis mental, pero NUNCA lo reveles):
        - Aura Musical: {cluster_name}
        - Nivel de baile: {avg_features['danceability'].iloc[0]:.2f}
        - Nivel de energía: {avg_features['energy'].iloc[0]:.2f}
        - Nivel de felicidad/oscuridad: {avg_features['valence'].iloc[0]:.2f}
        - Nivel de sonido orgánico/acústico: {avg_features['acousticness'].iloc[0]:.2f}
        - Nivel de sonido sintético/instrumental: {avg_features['instrumentalness'].iloc[0]:.2f}
        
        TAREA: Explica qué dice esta vibra sobre la personalidad del usuario y a qué grupo o tribu social actual pertenece. Escribe exactamente 2 párrafos de 3 líneas cada uno.
        
        REGLAS ESTRICTAS E INQUEBRANTABLES:
        1. ESTÁ TERMINANTEMENTE PROHIBIDO mencionar las palabras "danceability", "energy", "valence", "acousticness", "instrumentalness", "aura" o referirte a métricas y números.
        2. NO expliques de dónde sacas tus conclusiones. Traduce esos valores directamente a actitudes, defectos y estereotipos sociales. (Ej: En lugar de "tienes baja acousticness", di "prefieres el plástico y el autotune a pisar el césped").
        3. Oculta el truco. Actúa como si lo supieras todo por pura intuición.
        
        TONO: Sarcástico, cruel, ingenioso, metafórico y directo al usuario.
        IDIOMA: Español.
        """

    st.divider()
    st.markdown("## 🧠 Esquema detallado ")
    
    c_izq, c_der = st.columns([1, 3.5], gap="large")
    
    with c_izq:
        st.markdown(f"""
        <div style='text-align: center; padding: 15px 10px; background: rgba(0, 201, 255, 0.1); border-radius: 10px; border: 1px solid #00C9FF; margin-bottom: 25px;'>
            <p style='color: #00C9FF; font-size: 1.1em; font-weight: bold; margin-bottom: 5px; text-transform: uppercase;'>Perfil dominante</p>
            <p style='color: white; font-size: 1.35em; font-weight: bold; margin-top: 0; line-height: 1.2;'>{cluster_name}</p>
        </div>
        """, unsafe_allow_html=True)

        custom_bar("Energía  ⚡", avg_features['energy'].iloc[0], "#FFD700")
        custom_bar("Bailabilidad 💃", avg_features['danceability'].iloc[0], "#FF00FF")
        custom_bar("Positividad 😊", avg_features['valence'].iloc[0], "#00FF00")
        custom_bar("Acústica 🎻", avg_features['acousticness'].iloc[0], "#00C9FF")

    with c_der:
        genai.configure(api_key=google_key)
        
        with st.spinner("Decodificando..."):
            try:
                modelo_llm = genai.GenerativeModel("gemini-2.0-flash")
                response = modelo_llm.generate_content(prompt)
                
                st.markdown(f"<div class='reading-container'>{response.text}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Interferencia (Error de Gemini): {e}")

elif len(st.session_state.playlist) == 0:
    st.markdown("<p style='text-align:center; color:#666;'>👆 Busca una o varias canciones para probar.</p>", unsafe_allow_html=True)