import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
import pickle
import os
import time
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Call Center IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 14px 18px;
}
.stButton > button {
    background: #E8521A;
    color: white;
    font-weight: 700;
    border: none;
    border-radius: 8px;
}
.stButton > button:hover {
    background: #C04010;
    color: white;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

CONSULTAS_DATA = [
    ("Quiero saber el estado de mi pedido numero 12345", "pedido"),
    ("Cuando llega mi pedido de la semana pasada", "pedido"),
    ("No me ha llegado el pedido que hice el lunes", "pedido"),
    ("Necesito rastrear mi entrega de hoy", "pedido"),
    ("Donde esta mi orden pendiente de distribucion", "pedido"),
    ("Mi pedido lleva tres dias sin llegar", "pedido"),
    ("Quiero confirmar si mi pedido fue procesado", "pedido"),
    ("Tengo un pedido en transito y no se cuando llega", "pedido"),
    ("Pueden decirme el estatus del pedido PED-8821", "pedido"),
    ("El pedido que hice el viernes todavia no aparece", "pedido"),
    ("Seguimiento de mi ultima orden de compra", "pedido"),
    ("Me aparece que el pedido fue cancelado pero yo no lo cancele", "pedido"),
    ("Cual es el numero de guia de mi pedido", "pedido"),
    ("Pedi 200 unidades y solo llegaron 150 falta completar", "pedido"),
    ("Mi pedido esta en estado pendiente hace 5 dias", "pedido"),
    ("Necesito saber si ya salio mi pedido del almacen", "pedido"),
    ("Confirmar llegada del pedido para manana", "pedido"),
    ("El transportista no llego a entregar mi pedido", "pedido"),
    ("Cambiar la direccion de entrega de mi pedido activo", "pedido"),
    ("Quiero cancelar un pedido que acabo de hacer", "pedido"),
    ("No me llego la factura del mes pasado", "factura"),
    ("Necesito una copia de mi factura de enero", "factura"),
    ("Me llego una factura con un monto incorrecto", "factura"),
    ("Cuando me van a enviar la factura electronica", "factura"),
    ("Tengo un problema con la facturacion de mi cuenta", "factura"),
    ("Necesito reenviar la factura a otro correo electronico", "factura"),
    ("Me cobraron dos veces en la misma factura", "factura"),
    ("La factura tiene mi RUC mal escrito", "factura"),
    ("Pueden emitir una nota de credito por la factura 556", "factura"),
    ("El XML de la factura electronica no abre en mi sistema", "factura"),
    ("Necesito factura con detalle por producto individual", "factura"),
    ("Quiero que cambien la razon social en mis facturas", "factura"),
    ("No reconozco un cargo en mi ultima factura", "factura"),
    ("Cuando vence el plazo de pago de mi factura actual", "factura"),
    ("Puedo pagar la factura en dos partes este mes", "factura"),
    ("Recibi productos danados en mi ultimo pedido", "reclamo"),
    ("Me enviaron productos vencidos y necesito que los recojan", "reclamo"),
    ("Quiero hacer una queja formal por el servicio recibido", "reclamo"),
    ("El asesor que me atendio fue muy grosero", "reclamo"),
    ("Llevo dos semanas esperando respuesta a mi reclamo", "reclamo"),
    ("Me prometieron una devolucion que nunca llego", "reclamo"),
    ("Los productos llegaron sin las etiquetas requeridas", "reclamo"),
    ("Me enviaron una referencia diferente a la que pedi", "reclamo"),
    ("El embalaje llego completamente destruido", "reclamo"),
    ("Quiero escalar mi reclamo porque nadie me responde", "reclamo"),
    ("El producto tiene defectos de fabricacion evidentes", "reclamo"),
    ("Me cobraron un precio diferente al acordado", "reclamo"),
    ("Reclamo por incumplimiento del tiempo de entrega prometido", "reclamo"),
    ("Necesito hablar con un supervisor sobre mi caso", "reclamo"),
    ("Tengo un reclamo abierto hace 10 dias sin resolucion", "reclamo"),
    ("Quiero actualizar mi direccion de entrega principal", "cuenta"),
    ("Como cambio el correo de mi cuenta de distribuidor", "cuenta"),
    ("Necesito aumentar mi linea de credito con ustedes", "cuenta"),
    ("Quiero agregar un nuevo punto de entrega a mi cuenta", "cuenta"),
    ("Cual es mi limite de credito actual", "cuenta"),
    ("Me bloquearon el acceso a la plataforma de pedidos", "cuenta"),
    ("Quiero actualizar los datos de mi representante de compras", "cuenta"),
    ("Como puedo ver el historial de mis compras del ano", "cuenta"),
    ("Necesito cambiar mi numero de contacto registrado", "cuenta"),
    ("Quiero registrar una nueva cuenta bancaria para pagos", "cuenta"),
    ("Mi usuario de la plataforma no funciona desde ayer", "cuenta"),
    ("Pueden enviarme el estado de cuenta del trimestre", "cuenta"),
    ("Quiero dar de baja una sucursal de mi cuenta", "cuenta"),
    ("Cual es mi saldo pendiente con ustedes actualmente", "cuenta"),
    ("Necesito el certificado de ser distribuidor autorizado", "cuenta"),
    ("Cuales son los horarios de atencion del call center", "general"),
    ("Tienen disponible el producto SKU-2234 en stock", "general"),
    ("Cuales son las promociones vigentes para distribuidores", "general"),
    ("Como puedo convertirme en distribuidor autorizado", "general"),
    ("A que hora cierran el almacen de despacho", "general"),
    ("Cuando es la proxima visita del representante de ventas", "general"),
    ("Tienen servicio de entrega para la provincia de Arequipa", "general"),
    ("Cual es el pedido minimo para obtener descuento", "general"),
    ("Donde puedo descargar el catalogo de productos actualizado", "general"),
    ("Cuales son las politicas de devolucion de productos", "general"),
    ("Cuanto tiempo tarda normalmente el proceso de entrega", "general"),
    ("Tienen numero de WhatsApp para hacer pedidos", "general"),
    ("Que documentos necesito para abrir una cuenta nueva", "general"),
    ("Cuales son las zonas de cobertura de distribucion", "general"),
    ("Tienen algun descuento por volumen de compra mensual", "general"),
]

RESPUESTAS = {
    "pedido":  "Su pedido esta en transito. Le enviamos el numero de guia al correo registrado.",
    "factura": "Su factura fue localizada. Reenviamos copia al correo de su cuenta.",
    "cuenta":  "Datos de su cuenta verificados. Linea de credito vigente disponible.",
    "general": "Atencion al distribuidor: Lunes a Viernes 8:00 a 18:00 horas.",
    "reclamo": "Su reclamo requiere revision por un especialista. Derivando con contexto completo.",
}

ICONS = {
    "pedido": "📦",
    "factura": "🧾",
    "reclamo": "⚠️",
    "cuenta": "👤",
    "general": "📋",
}


def limpiar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


@st.cache_resource(show_spinner=False)
def cargar_modelo():
    df = pd.DataFrame(CONSULTAS_DATA, columns=["consulta", "categoria"])
    df["limpia"] = df["consulta"].apply(limpiar_texto)
    X, y = df["limpia"], df["categoria"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    if os.path.exists("modelo_callcenter_ia.pkl"):
        with open("modelo_callcenter_ia.pkl", "rb") as f:
            pipe = pickle.load(f)
        fuente = "modelo real (entrenado en Colab)"
    else:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), max_features=5000,
                sublinear_tf=True, min_df=1
            )),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=3
            ))
        ])
        pipe.fit(X_train, y_train)
        fuente = "modelo simulado (sin .pkl)"

    cv = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    acc = pipe.score(X_test, y_test)
    return pipe, acc, cv.mean(), cv.std(), X_test, y_test, fuente


def clasificar(modelo, texto, umbral=0.50):
    limpio = limpiar_texto(texto)
    clases = modelo.classes_
    probs = modelo.predict_proba([limpio])[0]
    idx = probs.argmax()
    cat = clases[idx]
    confianza = float(probs[idx])
    resolvible = cat != "reclamo"
    ia_ok = resolvible and confianza >= umbral
    respuesta = RESPUESTAS[cat]
    if not ia_ok:
        motivo = "Reclamo: siempre a asesor" if not resolvible else f"Confianza {confianza:.0%} bajo umbral"
        respuesta = f"Derivando a asesor humano. Motivo: {motivo}."
    return {
        "categoria": cat,
        "confianza": confianza,
        "ia_resuelve": ia_ok,
        "respuesta": respuesta,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }


if "historial" not in st.session_state:
    st.session_state.historial = []

with st.spinner("Cargando modelo..."):
    modelo, acc_test, cv_mean, cv_std, X_test, y_test, fuente = cargar_modelo()

with st.sidebar:
    st.markdown("## 🤖 Call Center IA")
    st.markdown("**Modelo Hibrido IA + Humano**")
    st.caption(f"Fuente: {fuente}")
    st.markdown("---")
    st.markdown("### Configuracion")
    umbral = st.slider(
        "Umbral de confianza",
        min_value=0.30, max_value=0.90, value=0.50, step=0.05,
        help="Si la confianza es menor a este valor, la consulta escala al asesor humano."
    )
    st.markdown("---")
    st.markdown("### Sesion actual")
    total_h = len(st.session_state.historial)
    ia_h = sum(1 for r in st.session_state.historial if r["ia_resuelve"])
    st.metric("Consultas procesadas", total_h)
    if total_h > 0:
        st.metric("Resueltas por IA", f"{ia_h} ({ia_h/total_h:.0%})")
        st.metric("Escaladas a humano", total_h - ia_h)
    st.markdown("---")
    if st.button("Limpiar historial"):
        st.session_state.historial = []
        st.rerun()

st.markdown("## 🤖 Call Center IA — Clasificador de Intencion")
st.caption("Modelo Hibrido IA + Humano · NLP · Machine Learning · Automatizacion de consultas")

tab1, tab2, tab3 = st.tabs(["💬 Clasificador en vivo", "📊 Metricas del modelo", "📋 Historial"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Ingresar consulta")
        consulta_texto = st.text_area(
            "Escribe la consulta del distribuidor:",
            height=120,
            placeholder="Ej: No me llego el pedido de la semana pasada..."
        )
        st.markdown("**O elige un ejemplo:**")
        ejemplos = {
            "Seleccionar...": "",
            "Pedido — IA resuelve":   "Necesito saber donde esta mi pedido PED-9921, lleva 3 dias sin llegar",
            "Factura — IA resuelve":  "Me llego la factura FAC-0045 con el monto incorrecto",
            "Cuenta — IA resuelve":   "Quiero actualizar mi direccion de entrega y ver mi linea de credito",
            "General — IA resuelve":  "Cuales son los horarios de atencion y zonas de cobertura",
            "Reclamo — escala siempre": "Recibi productos completamente danados y quiero hacer un reclamo formal",
            "Ambigua — baja confianza": "Tengo una situacion con lo que me enviaron que no me cuadra",
        }
        sel = st.selectbox("Ejemplos rapidos:", list(ejemplos.keys()))
        if ejemplos[sel]:
            consulta_texto = ejemplos[sel]

        clasificar_btn = st.button("Clasificar consulta", use_container_width=True)

    with col2:
        st.markdown("### Resultado del flujo TO-BE")

        if clasificar_btn and consulta_texto.strip():
            with st.spinner("Procesando..."):
                time.sleep(0.3)
                resultado = clasificar(modelo, consulta_texto, umbral)
                st.session_state.historial.append({**resultado, "consulta": consulta_texto})

            cat = resultado["categoria"]
            conf = resultado["confianza"]
            ia = resultado["ia_resuelve"]

            if ia:
                st.success(f"🤖 RESOLUCION AUTOMATICA IA")
            else:
                st.error(f"👤 ESCALAMIENTO A ASESOR HUMANO")

            col_a, col_b = st.columns(2)
            col_a.metric("Categoria detectada", f"{ICONS[cat]} {cat.upper()}")
            col_b.metric("Confianza del modelo", f"{conf:.1%}")

            st.markdown("**Consulta recibida:**")
            st.info(consulta_texto)

            st.markdown("**Respuesta de la IA:**")
            if ia:
                st.success(resultado["respuesta"])
            else:
                st.warning(resultado["respuesta"])

            conf_pct = int(conf * 100)
            st.markdown(f"Confianza: **{conf_pct}%** {'✅ sobre el umbral' if conf >= umbral else '⚠️ bajo el umbral'}")
            st.progress(conf_pct)

        elif clasificar_btn:
            st.warning("Escribe o selecciona una consulta primero.")
        else:
            st.markdown("""
            Ingresa una consulta y presiona **Clasificar consulta**.

            El modelo NLP analizara la intencion y determinara si la IA
            puede resolver automaticamente o si debe escalar al equipo humano.
            """)

with tab2:
    st.markdown("### Performance del modelo NLP/ML")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy Test Set", f"{acc_test:.1%}")
    c2.metric("Cross-Val 5-fold", f"{cv_mean:.1%}")
    c3.metric("Desviacion estandar", f"+-{cv_std:.1%}")
    c4.metric("Categorias", "5")

    st.markdown("---")

    col_cm, col_bar = st.columns(2, gap="large")

    with col_cm:
        st.markdown("#### Matriz de Confusion")
        cats_ord = ["pedido", "factura", "reclamo", "cuenta", "general"]
        y_pred = modelo.predict(X_test.apply(limpiar_texto))
        cm = confusion_matrix(y_test, y_pred, labels=cats_ord)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=cats_ord, yticklabels=cats_ord,
                    ax=ax, linewidths=0.5)
        ax.set_xlabel("Predicho", fontsize=10)
        ax.set_ylabel("Real", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_bar:
        st.markdown("#### Accuracy por Categoria")
        accs = [cm[i][i] / cm[i].sum() if cm[i].sum() > 0 else 0 for i in range(len(cats_ord))]
        colors = ["#22c55e" if a >= 0.90 else "#f59e0b" if a >= 0.80 else "#ef4444" for a in accs]
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        bars = ax2.barh(cats_ord, [a * 100 for a in accs], color=colors, height=0.5)
        ax2.set_xlim(0, 115)
        ax2.axvline(80, color="gray", linestyle="--", lw=1, label="Min 80%")
        ax2.axvline(90, color="#22c55e", linestyle="--", lw=1, label="Meta 90%")
        ax2.set_xlabel("Accuracy (%)")
        ax2.legend(fontsize=8)
        for bar, acc in zip(bars, accs):
            ax2.text(bar.get_width() + 1,
                     bar.get_y() + bar.get_height() / 2,
                     f"{acc:.0%}", va="center", fontweight="bold")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("#### Reporte de clasificacion")
    report = classification_report(y_test, y_pred, target_names=cats_ord, output_dict=True)
    df_rep = pd.DataFrame(report).T.round(2).iloc[:-3]
    df_rep.columns = ["Precision", "Recall", "F1-Score", "Soporte"]
    st.dataframe(df_rep, use_container_width=True)

with tab3:
    st.markdown("### Historial de consultas")

    if not st.session_state.historial:
        st.info("Aun no hay consultas procesadas. Ve a la pestana Clasificador en vivo para comenzar.")
    else:
        total_s = len(st.session_state.historial)
        ia_s = sum(1 for r in st.session_state.historial if r["ia_resuelve"])
        conf_s = np.mean([r["confianza"] for r in st.session_state.historial])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total", total_s)
        k2.metric("IA resuelve", f"{ia_s} ({ia_s/total_s:.0%})")
        k3.metric("Escala a humano", total_s - ia_s)
        k4.metric("Confianza promedio", f"{conf_s:.0%}")

        st.markdown("---")

        for r in reversed(st.session_state.historial):
            accion = "IA Resuelve" if r["ia_resuelve"] else "Asesor Humano"
            color = "green" if r["ia_resuelve"] else "red"
            st.markdown(
                f"**:{color}[{accion}]** &nbsp;|&nbsp; "
                f"{ICONS[r['categoria']]} `{r['categoria'].upper()}` &nbsp;|&nbsp; "
                f"Confianza: `{r['confianza']:.0%}` &nbsp;|&nbsp; "
                f"{r['consulta'][:70]}{'...' if len(r['consulta']) > 70 else ''} "
                f"&nbsp;|&nbsp; _{r['timestamp']}_"
            )

        st.markdown("---")
        df_hist = pd.DataFrame(st.session_state.historial)[
            ["timestamp", "consulta", "categoria", "confianza", "ia_resuelve", "respuesta"]
        ]
        df_hist.columns = ["Hora", "Consulta", "Categoria", "Confianza", "IA Resuelve", "Respuesta"]
        csv = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar historial (CSV)",
            data=csv,
            file_name=f"historial_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
