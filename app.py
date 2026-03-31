"""
╔══════════════════════════════════════════════════════════════╗
║   Call Center IA – Clasificador de Intención                ║
║   Modelo Híbrido IA + Humano · Streamlit App                ║
║   Deploy: Streamlit Cloud (gratis)                          ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
import pickle
import json
import io
import time
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Call Center IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos CSS personalizados ────────────────────────────────────────────────
st.markdown("""
<style>
  /* Fuente base */
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #1C1C2E;
  }
  [data-testid="stSidebar"] * { color: #E5E7EB !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stTextArea label { color: #E8521A !important; font-weight: 600; }

  /* Métricas */
  [data-testid="metric-container"] {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 14px 18px;
  }
  [data-testid="metric-container"] label { font-size: 11px !important; font-weight: 700; letter-spacing: 0.08em; color: #6B7280 !important; text-transform: uppercase; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 800; color: #E8521A !important; }

  /* Tarjetas resultado */
  .result-card {
    background: #fff;
    border: 1.5px solid #E5E7EB;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 10px 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
  }
  .result-card.ia   { border-left: 5px solid #059669; }
  .result-card.human{ border-left: 5px solid #DC2626; }

  .badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .badge.ia    { background: #D1FAE5; color: #065F46; }
  .badge.human { background: #FEE2E2; color: #991B1B; }
  .badge.cat   { background: #FFF7ED; color: #C2410C; }

  .conf-bar-wrap { background:#F3F4F6; border-radius:6px; height:10px; margin:6px 0 2px; }
  .conf-bar      { height:10px; border-radius:6px; background:linear-gradient(90deg,#E8521A,#C04010); }

  /* Historial */
  .hist-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 8px;
    margin: 6px 0;
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    font-size: 13px;
  }
  .hist-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
  .hist-dot.ia    { background:#059669; }
  .hist-dot.human { background:#DC2626; }

  /* Header */
  .app-header {
    background: linear-gradient(135deg, #1C1C2E 60%, #2D1B0E);
    color: white;
    padding: 28px 32px;
    border-radius: 14px;
    margin-bottom: 24px;
  }
  .app-header h1 { color:#E8521A; font-size:26px; margin:0 0 4px; font-weight:800; }
  .app-header p  { color:#9CA3AF; margin:0; font-size:14px; }

  /* Code block */
  .code-preview {
    background:#0D1117;
    border-radius:10px;
    padding:18px 22px;
    font-family:'DM Mono',monospace;
    font-size:12px;
    color:#A8FF90;
    border:1px solid #30363D;
    white-space:pre;
    overflow-x:auto;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 13px;
  }
  .stTabs [aria-selected="true"] {
    color: #E8521A !important;
    border-bottom-color: #E8521A !important;
  }

  /* Botón primario */
  .stButton > button {
    background: #E8521A;
    color: white;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 14px;
    transition: all 0.2s;
  }
  .stButton > button:hover { background: #C04010; }

  /* Ocultar footer de Streamlit */
  footer { visibility: hidden; }
  #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATOS Y MODELO
# ══════════════════════════════════════════════════════════════════════════════

CONSULTAS_DATA = [
    ('Quiero saber el estado de mi pedido número 12345', 'pedido'),
    ('Cuándo llega mi pedido de la semana pasada', 'pedido'),
    ('No me ha llegado el pedido que hice el lunes', 'pedido'),
    ('Necesito rastrear mi entrega de hoy', 'pedido'),
    ('Dónde está mi orden pendiente de distribución', 'pedido'),
    ('Mi pedido lleva tres días sin llegar', 'pedido'),
    ('Quiero confirmar si mi pedido fue procesado', 'pedido'),
    ('Tengo un pedido en tránsito y no sé cuándo llega', 'pedido'),
    ('Pueden decirme el estatus del pedido PED-8821', 'pedido'),
    ('El pedido que hice el viernes todavía no aparece', 'pedido'),
    ('Seguimiento de mi última orden de compra', 'pedido'),
    ('Me aparece que el pedido fue cancelado pero yo no lo cancelé', 'pedido'),
    ('Cuál es el número de guía de mi pedido', 'pedido'),
    ('Pedí 200 unidades y solo llegaron 150, falta completar', 'pedido'),
    ('Mi pedido está en estado pendiente hace 5 días', 'pedido'),
    ('Necesito saber si ya salió mi pedido del almacén', 'pedido'),
    ('Confirmar llegada del pedido para mañana', 'pedido'),
    ('El transportista no llegó a entregar mi pedido', 'pedido'),
    ('Cambiar la dirección de entrega de mi pedido activo', 'pedido'),
    ('Quiero cancelar un pedido que acabo de hacer', 'pedido'),
    ('No me llegó la factura del mes pasado', 'factura'),
    ('Necesito una copia de mi factura de enero', 'factura'),
    ('Me llegó una factura con un monto incorrecto', 'factura'),
    ('Cuándo me van a enviar la factura electrónica', 'factura'),
    ('Tengo un problema con la facturación de mi cuenta', 'factura'),
    ('Necesito reenviar la factura a otro correo electrónico', 'factura'),
    ('Me cobraron dos veces en la misma factura', 'factura'),
    ('La factura tiene mi RUC mal escrito', 'factura'),
    ('Pueden emitir una nota de crédito por la factura 556', 'factura'),
    ('El XML de la factura electrónica no abre en mi sistema', 'factura'),
    ('Necesito factura con detalle por producto individual', 'factura'),
    ('Quiero que cambien la razón social en mis facturas', 'factura'),
    ('No reconozco un cargo en mi última factura', 'factura'),
    ('Cuándo vence el plazo de pago de mi factura actual', 'factura'),
    ('Puedo pagar la factura en dos partes este mes', 'factura'),
    ('Recibí productos dañados en mi último pedido', 'reclamo'),
    ('Me enviaron productos vencidos y necesito que los recojan', 'reclamo'),
    ('Quiero hacer una queja formal por el servicio recibido', 'reclamo'),
    ('El asesor que me atendió fue muy grosero', 'reclamo'),
    ('Llevo dos semanas esperando respuesta a mi reclamo', 'reclamo'),
    ('Me prometieron una devolución que nunca llegó', 'reclamo'),
    ('Los productos llegaron sin las etiquetas requeridas', 'reclamo'),
    ('Me enviaron una referencia diferente a la que pedí', 'reclamo'),
    ('El embalaje llegó completamente destruido', 'reclamo'),
    ('Quiero escalar mi reclamo porque nadie me responde', 'reclamo'),
    ('El producto tiene defectos de fabricación evidentes', 'reclamo'),
    ('Me cobraron un precio diferente al acordado', 'reclamo'),
    ('Reclamo por incumplimiento del tiempo de entrega prometido', 'reclamo'),
    ('Necesito hablar con un supervisor sobre mi caso', 'reclamo'),
    ('Tengo un reclamo abierto hace 10 días sin resolución', 'reclamo'),
    ('Quiero actualizar mi dirección de entrega principal', 'cuenta'),
    ('Cómo cambio el correo de mi cuenta de distribuidor', 'cuenta'),
    ('Necesito aumentar mi línea de crédito con ustedes', 'cuenta'),
    ('Quiero agregar un nuevo punto de entrega a mi cuenta', 'cuenta'),
    ('Cuál es mi límite de crédito actual', 'cuenta'),
    ('Me bloquearon el acceso a la plataforma de pedidos', 'cuenta'),
    ('Quiero actualizar los datos de mi representante de compras', 'cuenta'),
    ('Cómo puedo ver el historial de mis compras del año', 'cuenta'),
    ('Necesito cambiar mi número de contacto registrado', 'cuenta'),
    ('Quiero registrar una nueva cuenta bancaria para pagos', 'cuenta'),
    ('Mi usuario de la plataforma no funciona desde ayer', 'cuenta'),
    ('Pueden enviarme el estado de cuenta del trimestre', 'cuenta'),
    ('Quiero dar de baja una sucursal de mi cuenta', 'cuenta'),
    ('Cuál es mi saldo pendiente con ustedes actualmente', 'cuenta'),
    ('Necesito el certificado de ser distribuidor autorizado', 'cuenta'),
    ('Cuáles son los horarios de atención del call center', 'general'),
    ('Tienen disponible el producto SKU-2234 en stock', 'general'),
    ('Cuáles son las promociones vigentes para distribuidores', 'general'),
    ('Cómo puedo convertirme en distribuidor autorizado', 'general'),
    ('A qué hora cierran el almacén de despacho', 'general'),
    ('Cuándo es la próxima visita del representante de ventas', 'general'),
    ('Tienen servicio de entrega para la provincia de Arequipa', 'general'),
    ('Cuál es el pedido mínimo para obtener descuento', 'general'),
    ('Dónde puedo descargar el catálogo de productos actualizado', 'general'),
    ('Cuáles son las políticas de devolución de productos', 'general'),
    ('Cuánto tiempo tarda normalmente el proceso de entrega', 'general'),
    ('Tienen número de WhatsApp para hacer pedidos', 'general'),
    ('Qué documentos necesito para abrir una cuenta nueva', 'general'),
    ('Cuáles son las zonas de cobertura de distribución', 'general'),
    ('Tienen algún descuento por volumen de compra mensual', 'general'),
]

RESPUESTAS = {
    'pedido':  '📦 Su pedido está en tránsito. Fecha estimada de entrega: {fecha}. Le enviamos el número de guía por correo.',
    'factura': '🧾 Su factura fue encontrada en el sistema. Le reenviamos una copia al correo registrado en su cuenta.',
    'cuenta':  '👤 Los datos de su cuenta han sido verificados. Línea de crédito vigente: S/ {monto}. ¿Desea actualizar algún dato?',
    'general': '📋 Atención al distribuidor: Lunes a Viernes 8:00–18:00. Para más detalles, su asesor le contactará en breve.',
    'reclamo': '⚠️ Su reclamo requiere revisión por un especialista. Derivando con contexto completo.',
}

CATEGORIAS_ICONS = {
    'pedido':  '📦',
    'factura': '🧾',
    'reclamo': '⚠️',
    'cuenta':  '👤',
    'general': '📋',
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def limpiar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    return re.sub(r'\s+', ' ', texto).strip()


@st.cache_resource(show_spinner=False)
def entrenar_modelo():
    import pickle, os
    if os.path.exists('modelo_callcenter_ia.pkl'):
        with open('modelo_callcenter_ia.pkl', 'rb') as f:
            pipe = pickle.load(f)
        df = pd.DataFrame(CONSULTAS_DATA, columns=['consulta', 'categoria'])
        df['limpia'] = df['consulta'].apply(limpiar_texto)
        X, y = df['limpia'], df['categoria']
        from sklearn.model_selection import train_test_split, cross_val_score
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        cv = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
        acc_test = pipe.score(X_test, y_test)
        return pipe, acc_test, cv.mean(), cv.std(), X_test, y_test
    else:
        df = pd.DataFrame(CONSULTAS_DATA, columns=['consulta', 'categoria'])
        df['limpia'] = df['consulta'].apply(limpiar_texto)
        X, y = df['limpia'], df['categoria']
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000,
                                       sublinear_tf=True, min_df=1)),
            ('clf', CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=3))
        ])
        pipe.fit(X_train, y_train)
        cv = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
        acc_test = pipe.score(X_test, y_test)
        return pipe, acc_test, cv.mean(), cv.std(), X_test, y_test
```

Abajo donde dice **Commit changes**, escribe:
```
actualizo app para cargar modelo real


def clasificar(modelo, texto, umbral=0.65):
    limpio   = limpiar_texto(texto)
    cat      = modelo.predict([limpio])[0]
    scores   = modelo.decision_function([limpio])[0]
    exp_s    = np.exp(scores - scores.max())
    confianza = float((exp_s / exp_s.sum()).max())
    resolvible = cat != 'reclamo'
    ia_ok    = resolvible and confianza >= umbral
    fecha    = (pd.Timestamp.now() + pd.Timedelta(days=np.random.randint(1,4))).strftime('%d/%m/%Y')
    monto    = f'{np.random.randint(1000, 50000):,}'
    respuesta = RESPUESTAS[cat].format(fecha=fecha, monto=monto)
    return {
        'categoria':  cat,
        'confianza':  confianza,
        'ia_resuelve': ia_ok,
        'respuesta':  respuesta,
        'timestamp':  datetime.now().strftime('%H:%M:%S'),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ESTADO DE SESIÓN
# ══════════════════════════════════════════════════════════════════════════════
if 'historial' not in st.session_state:
    st.session_state.historial = []
if 'modelo_listo' not in st.session_state:
    st.session_state.modelo_listo = False


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤖 Call Center IA")
    st.markdown("**Modelo Híbrido IA + Humano**")
    st.markdown("---")

    st.markdown("### ⚙️ Configuración")
    umbral = st.slider(
        "Umbral de confianza",
        min_value=0.40, max_value=0.90, value=0.65, step=0.05,
        help="Si la confianza del modelo es menor a este valor, la consulta escala a un asesor humano."
    )
    st.markdown(f"*Umbral actual: **{umbral:.0%}***")

    st.markdown("---")
    st.markdown("### 📊 Sesión actual")
    total_h = len(st.session_state.historial)
    ia_h    = sum(1 for r in st.session_state.historial if r['ia_resuelve'])
    st.metric("Consultas procesadas", total_h)
    if total_h > 0:
        st.metric("Resueltas por IA", f"{ia_h}/{total_h} ({ia_h/total_h:.0%})")
        st.metric("Escaladas a humano", total_h - ia_h)

    st.markdown("---")
    if st.button("🗑️ Limpiar historial"):
        st.session_state.historial = []
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ Stack tecnológico")
    st.markdown("""
    - 🐍 **Python** + scikit-learn
    - 📝 **TF-IDF** (ngram 1-2)
    - 🔷 **LinearSVC** clasificador
    - 🌐 **Streamlit** frontend
    - ☁️ **Streamlit Cloud** deploy
    """)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
  <h1>🤖 Call Center IA – Clasificador de Intención</h1>
  <p>Modelo Híbrido IA + Humano · Automatización inteligente de consultas de distribuidores · NLP · Machine Learning · RPA</p>
</div>
""", unsafe_allow_html=True)

# ── Cargar modelo ─────────────────────────────────────────────────────────────
with st.spinner("⏳ Inicializando modelo NLP/ML..."):
    modelo, acc_test, cv_mean, cv_std, X_test, y_test = entrenar_modelo()

# ══════════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Clasificador en Vivo",
    "📊 Métricas del Modelo",
    "📋 Historial de Sesión",
    "🔌 API & Producción",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – CLASIFICADOR EN VIVO
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### 💬 Ingresar consulta del distribuidor")

        consulta_texto = st.text_area(
            "Escribe la consulta:",
            height=120,
            placeholder="Ej: No me llegó el pedido de la semana pasada...",
            label_visibility="collapsed"
        )

        st.markdown("**O elige un ejemplo rápido:**")
        ejemplos = {
            "📦 Seguimiento de pedido":         "Necesito saber dónde está mi pedido PED-9921, lleva 3 días sin llegar",
            "🧾 Problema con factura":           "Me llegó la factura FAC-0045 con el monto incorrecto, necesito una corrección",
            "👤 Actualizar cuenta":              "Quiero actualizar mi dirección de entrega y aumentar mi línea de crédito",
            "📋 Consulta general":               "Cuáles son los horarios de atención y zonas de cobertura que tienen",
            "⚠️ Reclamo – siempre a humano":    "Recibí productos completamente dañados y quiero hacer un reclamo formal urgente",
            "❓ Consulta ambigua (baja conf.)": "Tengo una situación con lo que me enviaron que no me cuadra del todo",
        }
        ejemplo_sel = st.selectbox("Ejemplos:", ["— seleccionar —"] + list(ejemplos.keys()), label_visibility="collapsed")
        if ejemplo_sel != "— seleccionar —":
            consulta_texto = ejemplos[ejemplo_sel]

        btn_clasificar = st.button("⚡ Clasificar consulta", use_container_width=True)

    with col_result:
        st.markdown("### 🎯 Resultado del flujo TO-BE")

        if btn_clasificar and consulta_texto.strip():
            with st.spinner("Procesando..."):
                time.sleep(0.4)  # simular latencia realista
                resultado = clasificar(modelo, consulta_texto, umbral)
                st.session_state.historial.append({**resultado, 'consulta': consulta_texto})

            cat  = resultado['categoria']
            conf = resultado['confianza']
            ia   = resultado['ia_resuelve']

            tipo_clase = "ia" if ia else "human"
            accion_txt = "🤖 RESOLUCIÓN AUTOMÁTICA IA" if ia else "👤 ESCALAMIENTO A ASESOR HUMANO"
            accion_color = "#059669" if ia else "#DC2626"

            st.markdown(f"""
            <div class="result-card {tipo_clase}">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
                <span style="font-size:22px;">{CATEGORIAS_ICONS[cat]}</span>
                <div>
                  <span class="badge {tipo_clase}">{accion_txt}</span>
                  &nbsp;
                  <span class="badge cat">{cat.upper()}</span>
                </div>
              </div>

              <p style="font-size:13px;color:#6B7280;margin:0 0 4px;font-weight:600;">CONSULTA RECIBIDA</p>
              <p style="font-size:14px;color:#1F2937;margin:0 0 14px;">"{consulta_texto[:120]}..."</p>

              <p style="font-size:13px;color:#6B7280;margin:0 0 4px;font-weight:600;">CONFIANZA DEL MODELO</p>
              <div class="conf-bar-wrap">
                <div class="conf-bar" style="width:{min(conf*100,100):.0f}%"></div>
              </div>
              <p style="font-size:12px;color:#374151;margin:0 0 14px;">{conf:.1%} &nbsp;{'✅ sobre el umbral' if conf >= umbral else '⚠️ bajo el umbral – escala a humano'}</p>

              <p style="font-size:13px;color:#6B7280;margin:0 0 4px;font-weight:600;">RESPUESTA DE LA IA</p>
              <p style="font-size:14px;color:#1F2937;margin:0 0 14px;padding:10px 14px;background:#F9FAFB;border-radius:8px;border:1px solid #E5E7EB;">{resultado['respuesta']}</p>

              <p style="font-size:11px;color:#9CA3AF;margin:0;">⏱ {resultado['timestamp']} · umbral configurado: {umbral:.0%}</p>
            </div>
            """, unsafe_allow_html=True)

        elif btn_clasificar:
            st.warning("Escribe o selecciona una consulta primero.")
        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#9CA3AF;">
              <div style="font-size:48px;margin-bottom:12px;">🤖</div>
              <p style="font-size:15px;">Ingresa una consulta y presiona <strong>Clasificar</strong></p>
              <p style="font-size:12px;">El modelo NLP analizará la intención y determinará si la IA puede resolver o si debe escalar al equipo humano.</p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – MÉTRICAS DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🎯 Performance del Modelo NLP/ML")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy Test Set",   f"{acc_test:.1%}")
    c2.metric("Cross-Val (5-fold)",  f"{cv_mean:.1%}")
    c3.metric("Desviación estándar", f"±{cv_std:.1%}")
    c4.metric("Categorías cubiertas","5")

    st.markdown("---")

    col_cm, col_bar = st.columns(2, gap="large")

    with col_cm:
        st.markdown("#### Matriz de Confusión")
        y_pred = modelo.predict(X_test.apply(limpiar_texto))
        cats_ord = ['pedido', 'factura', 'reclamo', 'cuenta', 'general']
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred, labels=cats_ord)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=cats_ord, yticklabels=cats_ord,
                    ax=ax, linewidths=0.5, linecolor='white',
                    cbar_kws={'shrink': 0.8})
        ax.set_xlabel('Predicho', fontsize=10, fontweight='bold')
        ax.set_ylabel('Real', fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_bar:
        st.markdown("#### Accuracy por Categoría")
        accs = []
        for i, cat in enumerate(cats_ord):
            total = cm[i].sum()
            accs.append(cm[i][i] / total if total > 0 else 0)

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        colors = ['#059669' if a >= 0.90 else '#D97706' if a >= 0.80 else '#DC2626' for a in accs]
        bars = ax2.barh(cats_ord, [a * 100 for a in accs],
                        color=colors, edgecolor='white', height=0.5)
        ax2.set_xlim(0, 115)
        ax2.axvline(80, color='#9CA3AF', linestyle='--', lw=1, label='Mínimo 80%')
        ax2.axvline(90, color='#059669', linestyle='--', lw=1, label='Objetivo 90%')
        ax2.set_xlabel('Accuracy (%)', fontsize=9)
        for bar, acc in zip(bars, accs):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     f'{acc:.0%}', va='center', fontweight='bold', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.tick_params(labelsize=9)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("#### 📋 Reporte completo de clasificación")
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, target_names=cats_ord, output_dict=True)
    df_report = pd.DataFrame(report).T.round(2).iloc[:-3]
    df_report.index.name = "Categoría"
    df_report.columns = ["Precisión", "Recall", "F1-Score", "Soporte"]
    st.dataframe(df_report, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – HISTORIAL
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📋 Historial de consultas en esta sesión")

    if not st.session_state.historial:
        st.info("Aún no hay consultas procesadas. Ve a la pestaña **Clasificador en Vivo** para comenzar.")
    else:
        # KPIs de sesión
        total_s = len(st.session_state.historial)
        ia_s    = sum(1 for r in st.session_state.historial if r['ia_resuelve'])
        conf_s  = np.mean([r['confianza'] for r in st.session_state.historial])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total procesadas", total_s)
        k2.metric("IA resuelve", f"{ia_s} ({ia_s/total_s:.0%})")
        k3.metric("Escala a humano", total_s - ia_s)
        k4.metric("Confianza promedio", f"{conf_s:.0%}")

        st.markdown("---")

        # Tabla de historial
        for i, r in enumerate(reversed(st.session_state.historial)):
            dot_class = "ia" if r['ia_resuelve'] else "human"
            accion    = "IA Resuelve" if r['ia_resuelve'] else "→ Asesor Humano"
            consulta_corta = r['consulta'][:80] + ('...' if len(r['consulta']) > 80 else '')
            st.markdown(f"""
            <div class="hist-row">
              <div class="hist-dot {dot_class}"></div>
              <span style="font-weight:700;color:#1F2937;min-width:110px;">{CATEGORIAS_ICONS[r['categoria']]} {r['categoria'].upper()}</span>
              <span style="flex:1;color:#374151;">{consulta_corta}</span>
              <span style="font-weight:600;color:{'#059669' if r['ia_resuelve'] else '#DC2626'};min-width:130px;">{accion}</span>
              <span style="color:#9CA3AF;font-size:11px;min-width:55px;">{r['confianza']:.0%}</span>
              <span style="color:#9CA3AF;font-size:11px;">{r['timestamp']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Exportar CSV
        df_hist = pd.DataFrame(st.session_state.historial)[
            ['timestamp','consulta','categoria','confianza','ia_resuelve','respuesta']
        ]
        df_hist.columns = ['Hora','Consulta','Categoría','Confianza','IA Resuelve','Respuesta IA']
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Descargar historial (CSV)",
            data=csv,
            file_name=f"historial_callcenter_ia_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – API & PRODUCCIÓN
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🔌 Pasos para integrar en producción")

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("#### 1️⃣ Endpoint REST (FastAPI)")
        st.markdown("""
        Copia este código en `api.py` y despliégalo en cualquier servidor:
        """)
        st.code("""
# api.py – FastAPI endpoint listo para producción
from fastapi import FastAPI
from pydantic import BaseModel
import pickle, re, unicodedata, numpy as np
from datetime import datetime

app = FastAPI(title="Call Center IA v1.0")

with open("modelo_callcenter_ia.pkl", "rb") as f:
    modelo = pickle.load(f)

class Consulta(BaseModel):
    texto: str
    id_distribuidor: str = "DIST-0000"
    umbral_confianza: float = 0.65

def limpiar(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9\\s]", " ", t).strip()

@app.post("/clasificar")
def clasificar(q: Consulta):
    cat   = modelo.predict([limpiar(q.texto)])[0]
    s     = modelo.decision_function([limpiar(q.texto)])[0]
    exp_s = np.exp(s - s.max())
    conf  = float((exp_s / exp_s.sum()).max())
    ia_ok = cat != "reclamo" and conf >= q.umbral_confianza
    return {
        "id_distribuidor": q.id_distribuidor,
        "categoria": cat,
        "confianza": round(conf, 4),
        "accion": "IA_RESUELVE" if ia_ok else "ESCALAR_HUMANO",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
def health():
    return {"status": "ok"}
        """, language="python")

        st.markdown("#### 2️⃣ Arrancar el servidor")
        st.code("uvicorn api:app --host 0.0.0.0 --port 8000", language="bash")

        st.markdown("#### 3️⃣ Llamar desde el IVR / sistema del Call Center")
        st.code("""
import requests

response = requests.post(
    "http://tu-servidor:8000/clasificar",
    json={
        "texto": "No me llegó el pedido de la semana",
        "id_distribuidor": "DIST-1042",
        "umbral_confianza": 0.65
    }
)
print(response.json())
# {"categoria":"pedido","confianza":0.88,"accion":"IA_RESUELVE",...}
        """, language="python")

    with col_b:
        st.markdown("#### 🚀 Deploy Streamlit Cloud (esta app, gratis)")
        st.markdown("""
        1. **Fork** este repositorio a tu GitHub
        2. Ve a **[share.streamlit.io](https://share.streamlit.io)**
        3. Conecta tu repositorio y selecciona `app.py`
        4. Clic en **Deploy** → URL pública en 2 minutos ✅
        """)

        st.markdown("---")
        st.markdown("#### ☁️ Opciones de deploy en nube")

        opciones = {
            "☁️ Azure App Service": {
                "precio": "Gratis (tier F1)",
                "dificultad": "⭐⭐",
                "ideal": "Si ya usan Microsoft 365",
                "cmd": "az webapp up --name callcenter-ia --runtime PYTHON:3.11"
            },
            "🟠 AWS Lambda": {
                "precio": "~$0 primeros 1M calls/mes",
                "dificultad": "⭐⭐⭐",
                "ideal": "Serverless, escala automático",
                "cmd": "serverless deploy"
            },
            "🌐 Google Cloud Run": {
                "precio": "Gratis hasta 2M req/mes",
                "dificultad": "⭐⭐",
                "ideal": "Si ya usan Google Workspace",
                "cmd": "gcloud run deploy callcenter-ia --source ."
            },
            "🎈 Streamlit Cloud": {
                "precio": "100% Gratis",
                "dificultad": "⭐",
                "ideal": "Demo, POC y presentaciones",
                "cmd": "# Solo subir a GitHub y conectar"
            },
        }

        for nombre, info in opciones.items():
            with st.expander(nombre):
                c1, c2 = st.columns(2)
                c1.markdown(f"**Precio:** {info['precio']}")
                c2.markdown(f"**Dificultad:** {info['dificultad']}")
                st.markdown(f"**Ideal para:** {info['ideal']}")
                st.code(info['cmd'], language="bash")

        st.markdown("---")
        st.markdown("#### 📦 requirements.txt")
        st.code("""streamlit>=1.32.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
fastapi>=0.110.0
uvicorn>=0.27.0
pydantic>=2.0.0
requests>=2.31.0
        """, language="text")
