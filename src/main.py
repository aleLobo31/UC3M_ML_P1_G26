import streamlit as st
import pandas as pd
import os
import random
from model_strategy import JoblibModelStrategy
import __main__

# --- Funciones de preprocesado requeridas por el Pipeline ---
def pdays_transform(x):
    if x == -1:
        return 0
    elif x < 100:
        return 1
    elif x < 200:
        return 2
    elif x < 400:
        return 3
    else:
        return 4

def process_pdays(X: pd.DataFrame) -> pd.DataFrame:
    X_out = X.copy()
    X_out['wasContacted'] = X_out['pdays'].map(lambda x: 0 if x == -1 else 1)
    X_out['pdaysTransformed'] = X_out['pdays'].map(pdays_transform)
    return X_out

def calculate_propensity(row):
    if row['poutcome'] != 'success':
        return 0
    elif row['pdays'] < 0:
        return 0
    else:
        return 1/(1 + row['pdays'])

def process_poutcome(X: pd.DataFrame) -> pd.DataFrame:
    X_out = X.copy()
    X_out['DepositPropensity'] = X_out[['poutcome', 'pdays']].apply(calculate_propensity, axis=1)
    return X_out

def process_marital(X):
    X_out = X.copy()
    X_out['marital'] = X_out['marital'].fillna('unknown')
    return X_out

def drop_columns(X):
    return X.drop(['pdays'], axis=1, errors='ignore')

# Vincularlas al módulo __main__ explícitamente para que joblib las encuentre
__main__.pdays_transform = pdays_transform
__main__.process_pdays = process_pdays
__main__.calculate_propensity = calculate_propensity
__main__.process_poutcome = process_poutcome
__main__.process_marital = process_marital
__main__.drop_columns = drop_columns

# --- Configuración de página y estilos ---
st.set_page_config(page_title="Predicción de Depósitos Bancarios", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    /* Estilos globales y paleta de colores */
    .stApp {
        background-color: #E7EFFE;
        color: #000000;
    }
    h1, h2, h3, p, label, .stMarkdown, .stText {
        color: #000000 !important;
    }
    .header-box h1, .header-box p {
        color: #FFFFFF !important;
    }

    /* =====================================================
       FIX 1: Botones Deploy y menú arriba a la izquierda
       El header de Streamlit tiene fondo claro (#E7EFFE),
       por lo que los iconos/texto deben ser oscuros para contrastar.
    ===================================================== */
    header[data-testid="stHeader"] {
        background-color: #1E3A8A !important;
    }
    header[data-testid="stHeader"] * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }
    /* Botón Deploy específico */
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] a,
    [data-testid="stToolbar"] button,
    [data-testid="stToolbar"] svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }
    /* Forzar visibilidad del toolbar de deploy */
    [data-testid="stDeployButton"],
    [data-testid="stDeployButton"] * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    /* Estilos del Header banner */
    .header-banner {
        background-color: #6397F8;
        width: 100vw;
        position: relative;
        left: 50%;
        right: 50%;
        margin-left: -50vw;
        margin-right: -50vw;
        padding: 40px 0;
        margin-top: -60px;
        margin-bottom: 30px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .header-box {
        background-color: #0d2c6c;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        color: #FFFFFF !important;
        width: 70%;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
    }
    .header-box h1 {
        margin-bottom: 10px;
        font-size: 2.2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        color: #FFFFFF !important;
    }
    .header-box p {
        font-size: 1.2rem;
        margin-bottom: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        color: #FFFFFF !important;
    }
    .title-centered {
        text-align: center;
        font-weight: 600;
        margin-bottom: 10px;
        font-size: 2.5rem;
        color: #000000 !important;
    }
    .col-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 10px;
        color: #1E3A8A !important;
        text-align: center;
    }
    .field-desc {
        font-size: 0.9rem;
        color: #4A5568 !important;
        margin-top: -15px;
        margin-bottom: 5px;
    }

    /* =====================================================
       FIX 4: Cuadro resultado verde / rojo
       — ocupa todo el alto disponible, texto en una línea
    ===================================================== */
    .result-box-si {
        border: 4px solid #10B981;
        border-radius: 15px;
        padding: 0 20px;
        text-align: center;
        background-color: #D1FAE5;
        color: #065F46 !important;
        font-weight: bold;
        font-size: 1.4rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        min-height: 140px;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        white-space: nowrap;
        gap: 12px;
    }
    .result-box-no {
        border: 4px solid #EF4444;
        border-radius: 15px;
        padding: 0 20px;
        text-align: center;
        background-color: #FEE2E2;
        color: #991B1B !important;
        font-weight: bold;
        font-size: 1.4rem;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
        min-height: 140px;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        white-space: nowrap;
        gap: 12px;
    }

    /* =====================================================
       FIX 3: Cajas de probabilidad (columnas Yes / No)
    ===================================================== */
    .prob-box-si {
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background-color: #ECFDF5;
        margin-bottom: 10px;
    }
    .prob-box-si .prob-label {
        font-size: 1rem;
        font-weight: 600;
        color: #065F46 !important;
        margin-bottom: 6px;
    }
    .prob-box-si .prob-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #059669 !important;
    }
    .prob-box-no {
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background-color: #FFF5F5;
        margin-bottom: 10px;
    }
    .prob-box-no .prob-label {
        font-size: 1rem;
        font-weight: 600;
        color: #991B1B !important;
        margin-bottom: 6px;
    }
    .prob-box-no .prob-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #DC2626 !important;
    }

    /* Divisores */
    hr {
        border-color: #000000 !important;
        background-color: #000000 !important;
    }

    /* Botón Predecir */
    .stButton>button {
        display: block;
        margin: 0 auto;
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px;
        height: 50px;
        width: 250px;
        font-size: 1.2rem;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1D4ED8 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }
    .stButton>button p {
        color: #FFFFFF !important;
    }
    /* Botón secundario: Rellenar aleatoriamente */
    [data-testid="stBaseButton-secondary"] > button,
    .btn-random button {
        background-color: #64748B !important;
        color: #FFFFFF !important;
        border: 2px dashed #94A3B8 !important;
        border-radius: 8px;
        font-size: 0.95rem !important;
    }
    .btn-random button:hover {
        background-color: #475569 !important;
    }

    /* =====================================================
       FIX 2: Modal de error centrado y visible
    ===================================================== */
    div[data-testid="stDialog"] {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border-radius: 14px;
        padding: 10px;
        border: 2px solid #F59E0B;
    }
    div[data-testid="stDialog"] p,
    div[data-testid="stDialog"] h2,
    div[data-testid="stDialog"] .stMarkdown,
    div[data-testid="stDialog"] span {
        color: #FFFFFF !important;
        text-align: center;
    }
    /* Centrar el botón Entendido dentro del modal */
    div[data-testid="stDialog"] .stButton {
        display: flex;
        justify-content: center;
    }
    div[data-testid="stDialog"] .stButton > button {
        width: 180px !important;
        margin: 0 auto !important;
    }
    /* Overlay del modal: fondo semitransparente oscuro para centrado */
    div[data-testid="stModal"] {
        background-color: rgba(0, 0, 0, 0.55) !important;
        backdrop-filter: blur(2px) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div[data-testid="stModal"] > div {
        background-color: transparent !important;
    }

    /* Métricas centradas */
    div[data-testid="stMetric"] {
        text-align: center;
        margin: 0 auto;
    }
    div[data-testid="stMetricValue"] {
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# FIX 2: Modal de error — centrado, con campo faltante
# =====================================================
@st.dialog("⚠️ Campo obligatorio incompleto")
def mostrar_modal_error(mensaje):
    st.markdown(
        f"""
        <div style="text-align:center; padding: 10px 0;">
            <div style="font-size:3rem; margin-bottom:12px;">🚫</div>
            <p style="font-size:1.1rem; color:#FFFFFF; font-weight:500; margin-bottom:8px;">
                Por favor, completa el siguiente campo antes de continuar:
            </p>
            <div style="
                background-color:#F59E0B;
                color:#1C1917;
                border-radius:8px;
                padding:12px 20px;
                font-size:1.2rem;
                font-weight:bold;
                display:inline-block;
                margin-top:8px;
            ">
                {mensaje}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    # Centrar el botón Entendido con st.columns
    col_ok1, col_ok2, col_ok3 = st.columns([1, 1, 1])
    with col_ok2:
        if st.button("Entendido", use_container_width=True, key="btn_modal_ok"):
            st.rerun()

# --- Carga del modelo ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_final.joblib")
    if not os.path.exists(model_path):
        return None
    return JoblibModelStrategy(model_path)

modelo = load_model()

# --- Valores posibles para relleno aleatorio ---
_opciones_random = {
    'job':      ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'unknown', 'self-employed', 'student'],
    'marital':  ['married', 'divorced', 'single'],
    'education':['secondary', 'tertiary', 'primary', 'unknown'],
    'default':  ['no', 'yes'],
    'housing':  ['yes', 'no'],
    'loan':     ['no', 'yes'],
    'contact':  ['unknown', 'cellular', 'telephone'],
    'month':    ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'sep'],
    'poutcome': ['unknown', 'other', 'failure', 'success']
}
# Mapa de posición en la lista original (con '') para cada valor random
_opciones_full = {
    'job':      ['', 'admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'unknown', 'self-employed', 'student'],
    'marital':  ['', 'married', 'None', 'divorced', 'single'],
    'education':['', 'secondary', 'tertiary', 'primary', 'unknown'],
    'default':  ['', 'no', 'yes'],
    'housing':  ['', 'yes', 'no'],
    'loan':     ['', 'no', 'yes'],
    'contact':  ['', 'unknown', 'cellular', 'telephone'],
    'month':    ['', 'may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'sep'],
    'poutcome': ['', 'unknown', 'other', 'failure', 'success']
}

def rellenar_aleatorio():
    for var, vals in _opciones_random.items():
        valor = random.choice(vals)
        st.session_state[f"cat_{var}"] = _opciones_full[var].index(valor)
    st.session_state["num_age"]      = random.randint(18, 75)
    st.session_state["num_balance"]  = round(random.uniform(-500, 15000), 2)
    st.session_state["num_day"]      = random.randint(1, 28)
    st.session_state["num_duration"] = random.randint(30, 600)
    st.session_state["num_campaign"] = random.randint(1, 10)
    st.session_state["num_pdays"]    = random.choice([0] + list(range(0, 400)))
    st.session_state["num_previous"] = random.randint(0, 5)


# --- Header ---
st.markdown("""
<div class="header-banner">
    <div class="header-box">
        <h1>Anticípate al Éxito Financiero</h1>
        <p>Potencia tu estrategia comercial identificando con precisión la probabilidad de que un cliente contrate nuestro depósito bancario a plazo fijo.</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<h2 class="title-centered">Introduce los datos del cliente</h2>', unsafe_allow_html=True)
st.divider()

# Listas de variables categóricas
opciones_cat = {
    'job': ['', 'admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'unknown', 'self-employed', 'student'],
    'marital': ['', 'married', 'None', 'divorced', 'single'],
    'education': ['', 'secondary', 'tertiary', 'primary', 'unknown'],
    'default': ['', 'no', 'yes'],
    'housing': ['', 'yes', 'no'],
    'loan': ['', 'no', 'yes'],
    'contact': ['', 'unknown', 'cellular', 'telephone'],
    'month': ['', 'may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'sep'],
    'poutcome': ['', 'unknown', 'other', 'failure', 'success']
}

desc_cat = {
    'job': 'Tipo de trabajo del cliente',
    'marital': 'Estado civil actual',
    'education': 'Nivel educativo alcanzado',
    'default': '¿Tiene crédito en mora preexistente?',
    'housing': '¿Mantiene un préstamo hipotecario?',
    'loan': '¿Mantiene algún préstamo personal?',
    'contact': 'Medio de comunicación empleado',
    'month': 'Último mes de contacto en el año',
    'poutcome': 'Resultado de campañas de marketing previas'
}

# --- Layout en dos columnas ---
col_cat, col_num = st.columns(2, gap="large")

with col_cat:
    st.markdown('<div class="col-header">Variables Categóricas</div>', unsafe_allow_html=True)
    st.divider()
    datos_cat = {}
    for var, opciones in opciones_cat.items():
        display_options = ["Seleccione una opción..."] + opciones[1:]
        st.markdown(f"**{var.capitalize()}** *", unsafe_allow_html=True)
        st.markdown(f'<div class="field-desc">{desc_cat[var]}</div>', unsafe_allow_html=True)
        val_idx = st.selectbox(f"{var}", range(len(display_options)), format_func=lambda x, o=display_options: o[x], key=f"cat_{var}", label_visibility="collapsed")
        st.write("")
        datos_cat[var] = opciones[val_idx]

with col_num:
    st.markdown('<div class="col-header">Variables Numéricas</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**Age** *", unsafe_allow_html=True)
    st.markdown('<div class="field-desc">Edad del cliente en años</div>', unsafe_allow_html=True)
    age = st.number_input("Age", value=st.session_state.get("num_age", 0), step=1, format="%d", min_value=0, key="num_age", label_visibility="collapsed")
    st.write("")

    st.markdown("**Balance** *", unsafe_allow_html=True)
    st.markdown('<div class="field-desc">Saldo medio anual en la cuenta (puede tener decimales)</div>', unsafe_allow_html=True)
    balance = st.number_input("Balance", value=st.session_state.get("num_balance", 0.0), step=10.0, format="%.2f", key="num_balance", label_visibility="collapsed")
    st.write("")

    st.markdown("**Day** *", unsafe_allow_html=True)
    st.markdown('<div class="field-desc">Día del mes del último contacto</div>', unsafe_allow_html=True)
    day = st.number_input("Day", value=st.session_state.get("num_day", 0), step=1, format="%d", min_value=0, max_value=31, key="num_day", label_visibility="collapsed")
    st.write("")

    st.markdown("**Duration** *", unsafe_allow_html=True)
    st.markdown('<div class="field-desc">Duración del último contacto realizado en segundos</div>', unsafe_allow_html=True)
    duration = st.number_input("Duration", value=st.session_state.get("num_duration", 0), step=10, format="%d", min_value=0, key="num_duration", label_visibility="collapsed")
    st.write("")

    st.markdown("**Campaign** *", unsafe_allow_html=True)
    st.markdown('<div class="field-desc">Número de contactos realizados durante esta campaña (incluye último)</div>', unsafe_allow_html=True)
    campaign = st.number_input("Campaign", value=st.session_state.get("num_campaign", 0), step=1, format="%d", min_value=0, key="num_campaign", label_visibility="collapsed")
    st.write("")

    st.markdown("**Pdays** *", unsafe_allow_html=True)
    st.markdown('<div class="field-desc">Días transcurridos desde el último contacto (0 o más)</div>', unsafe_allow_html=True)
    pdays = st.number_input("Pdays", value=st.session_state.get("num_pdays", 0), step=1, format="%d", min_value=0, key="num_pdays", label_visibility="collapsed")
    st.write("")

    st.markdown("**Previous** *", unsafe_allow_html=True)
    st.markdown('<div class="field-desc">Número de contactos realizados antes de esta campaña</div>', unsafe_allow_html=True)
    previous = st.number_input("Previous", value=st.session_state.get("num_previous", 0), step=1, format="%d", min_value=0, key="num_previous", label_visibility="collapsed")
    st.write("")

st.write("---")

# Dos botones centrados: Rellenar aleatoriamente | Predecir
col_btn1, col_btn_rand, col_btn_sep, col_btn_pred, col_btn5 = st.columns([1, 2, 0.3, 2, 1])

with col_btn_rand:
    # Usamos on_click para evitar el error de modificación de session_state cuando los widgets ya están instanciados
    st.button("🎲 Rellenar aleatoriamente", use_container_width=True, key="btn_random", on_click=rellenar_aleatorio)

with col_btn_pred:
    btn_predict = st.button("🔍 Predecir", use_container_width=True, key="btn_predict")

# =====================================================
# Validación y predicción
# =====================================================
if btn_predict:
    error_detectado = None

    # Comprobar vacíos en categóricas
    for k, v in datos_cat.items():
        if v == '':
            # Nombre amigable del campo que falta
            nombre_campo = desc_cat.get(k, k.capitalize())
            error_detectado = f"{k.capitalize()} — {nombre_campo}"
            break

    # Comprobar numéricas
    if not error_detectado:
        if age <= 0:
            error_detectado = "Age — La edad debe ser mayor a 0"

    if error_detectado:
        mostrar_modal_error(error_detectado)
    else:
        if modelo is None:
            st.error("No se ha encontrado el modelo en 'models/modelo_final.joblib'. Asegúrese de entrenarlo y guardarlo.")
        else:
            input_dict = {
                **datos_cat,
                'age': age,
                'balance': balance,
                'day': day,
                'duration': duration,
                'campaign': campaign,
                'pdays': pdays,
                'previous': previous
            }
            if input_dict['marital'] == 'None':
                input_dict['marital'] = None

            df_input = pd.DataFrame([input_dict])

            try:
                pred = modelo.predict(df_input)[0]

                # =====================================================
                # FIX 3: Probabilidades reales del modelo
                # =====================================================
                try:
                    # En Scikit-Learn pipelines, si el estimador final es un clasificador lineal 
                    # como SVM o Ridge sin probability=True, predict_proba devuelve AttributeError.
                    # En su lugar, usamos decision_function si existe y lo escalamos, o miramos el estimador real.
                    
                    estimador_final = modelo.model.steps[-1][1] if hasattr(modelo.model, "steps") else modelo.model

                    if hasattr(estimador_final, "predict_proba"):
                        proba = modelo.predict_proba(df_input)
                        if proba.shape[1] == 2:
                            prob_no = proba[0][0] * 100
                            prob_si = proba[0][1] * 100
                        else:
                            prob_si = max(proba[0]) * 100
                            prob_no = 100 - prob_si
                    elif hasattr(estimador_final, "decision_function"):
                        import scipy.special
                        decision = estimador_final.decision_function(modelo.model[:-1].transform(df_input))
                        # Aplicar softmax o sigmoide
                        prob_si = scipy.special.expit(decision[0]) * 100
                        prob_no = 100 - prob_si
                    else:
                        raise AttributeError("No hay métodos probabilísticos disponibles en el pipeline")

                except Exception as e:
                    import traceback
                    # Opcionalmente, imprimir e para depuración: print("Error en predict_proba:", e)
                    prob_si = 100.0 if pred in ['yes', 1] else 0.0
                    prob_no = 100.0 if pred in ['no', 0] else 0.0

                st.markdown("<br><h3 class='title-centered'>Resultados de Predicción</h3>", unsafe_allow_html=True)
                st.divider()

                col_res1, col_res2 = st.columns(2, gap="large")

                # --- Columna izquierda: probabilidades en dos sub-columnas (Yes | No) ---
                with col_res1:
                    st.markdown('<div class="col-header">Probabilidades del Modelo</div>', unsafe_allow_html=True)
                    st.divider()

                    sub_col_si, sub_col_no = st.columns(2, gap="medium")

                    with sub_col_si:
                        st.markdown(
                            f"""
                            <div class="prob-box-si">
                                <div class="prob-label">✅ Probabilidad Sí</div>
                                <div class="prob-value">{prob_si:.1f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with sub_col_no:
                        st.markdown(
                            f"""
                            <div class="prob-box-no">
                                <div class="prob-label">❌ Probabilidad No</div>
                                <div class="prob-value">{prob_no:.1f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                # --- Columna derecha: decisión final enmarcada en verde o rojo ---
                with col_res2:
                    st.markdown('<div class="col-header">Decisión del Modelo</div>', unsafe_allow_html=True)
                    st.divider()

                    resultado_positivo = pred in ['yes', 1]

                    # FIX 4: cuadro con borde verde/rojo bien visible
                    if resultado_positivo:
                        st.markdown("""
                            <div class="result-box-si">
                                <span style="font-size:2rem; margin-right:10px;">✅</span>
                                <span>El cliente <strong>CONTRATARÁ</strong> el depósito.</span>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="result-box-no">
                                <span style="font-size:2rem; margin-right:10px;">❌</span>
                                <span>El cliente <strong>NO contratará</strong> el depósito.</span>
                            </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error al realizar la predicción. Asegúrate de que los campos coinciden con los esperados por el Pipeline del modelo. Detalle: {e}")