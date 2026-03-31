# 🤖 Call Center IA – Clasificador de Intención
## Modelo Híbrido IA + Humano · Deploy Streamlit Cloud

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📋 Descripción

Aplicación web que clasifica automáticamente las consultas del call center usando NLP + Machine Learning (TF-IDF + LinearSVC), determinando si la IA puede resolverlas o si deben escalarse a un asesor humano.

**Categorías:** pedido · factura · reclamo · cuenta · general

**Flujo TO-BE:**
```
Consulta → NLP (TF-IDF) → ML (LinearSVC) → ¿Confianza ≥ umbral?
                                                ↓ SÍ → Respuesta Automática IA
                                                ↓ NO → Escalamiento a Asesor Humano
```

---

## 🚀 Deploy en Streamlit Cloud (gratis, 5 pasos)

### Paso 1 – Fork del repositorio
1. Ve a este repositorio en GitHub
2. Clic en **Fork** (esquina superior derecha)
3. Confirma el fork en tu cuenta

### Paso 2 – Crear cuenta en Streamlit Cloud
1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Regístrate con tu cuenta de **GitHub** (mismo usuario del fork)

### Paso 3 – Conectar el repositorio
1. Clic en **"New app"**
2. Selecciona tu fork del repositorio
3. Branch: `main`
4. Main file: `app.py`

### Paso 4 – Deploy
1. Clic en **"Deploy!"**
2. Espera 2–3 minutos mientras Streamlit instala las dependencias
3. Tu app estará disponible en: `https://tu-usuario-callcenter-ia.streamlit.app`

### Paso 5 – Compartir
Comparte la URL pública con tu equipo o en la presentación.

---

## 🛠️ Ejecutar localmente

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/callcenter-ia.git
cd callcenter-ia

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la app
streamlit run app.py
```

La app abre automáticamente en `http://localhost:8501`

---

## 📁 Estructura del proyecto

```
callcenter-ia/
├── app.py              ← App principal Streamlit
├── requirements.txt    ← Dependencias Python
└── README.md           ← Este archivo
```

---

## ⚙️ Configuración

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| Umbral de confianza | 0.65 (65%) | Mínimo para que IA resuelva automáticamente |
| Algoritmo | LinearSVC | Clasificador SVM lineal |
| Vectorización | TF-IDF ngram(1,2) | Unigramas + bigramas |
| Dataset | 80 muestras simuladas | Reemplazar con datos reales del CC |

---

## 🔌 API REST (para integración con IVR/ERP)

Ver pestaña **"API & Producción"** dentro de la app para el código completo.

Endpoint: `POST /clasificar`

```json
// Request
{
  "texto": "No me llegó el pedido de la semana pasada",
  "id_distribuidor": "DIST-1042",
  "umbral_confianza": 0.65
}

// Response
{
  "categoria": "pedido",
  "confianza": 0.8823,
  "accion": "IA_RESUELVE",
  "timestamp": "2025-01-15T14:32:11"
}
```

---

## 📊 Métricas del POC

| Métrica | Valor |
|---------|-------|
| Accuracy (CV 5-fold) | ≥ 85% |
| Tasa resolución automática | ~70% |
| Tiempo de inferencia | < 200ms |
| Categorías cubiertas | 5 |

---

## 🧪 Próximos pasos (producción real)

1. **Datos reales:** Reemplazar las 80 muestras con ≥1,500 interacciones etiquetadas del CC
2. **Speech-to-Text:** Integrar Azure/Google STT para procesar audio de llamadas
3. **RPA:** Conectar con ERP/CRM para respuestas automáticas con datos reales
4. **Monitoreo:** Activar logs en producción + dashboard Power BI
5. **Reentrenamiento:** Ciclo mensual con nuevas interacciones etiquetadas

---

## 📄 Licencia

MIT – Libre para uso interno y académico.
