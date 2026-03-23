import io
import os

import requests
import streamlit as st
from PIL import Image

API_BASE_URL = os.getenv("API_BASE_URL", "").strip()

if API_BASE_URL and not API_BASE_URL.startswith("http"):
    API_BASE_URL = f"https://{API_BASE_URL}"

API_URL = f"{API_BASE_URL}/predict" if API_BASE_URL else ""

st.set_page_config(
    page_title="Clasificador de Malaria",
    page_icon="🦠",
    layout="centered"
)

st.title("🦠 Clasificador de Malaria")
st.markdown(
    """
    Esta aplicación permite cargar una imagen de una célula sanguínea y enviarla a una **API de predicción**,
    la cual utiliza un **modelo CNN LeNet** entrenado para clasificarla como:

    - **Parasitized**: célula parasitada
    - **Uninfected**: célula no infectada
    """
)

st.info("Flujo del sistema: Imagen → Streamlit → FastAPI → Modelo LeNet → Predicción")

if not API_BASE_URL:
    st.error(
        "La variable de entorno API_BASE_URL no está configurada. "
        "En Render debe apuntar al servicio malaria-api."
    )
    st.stop()

uploaded = st.file_uploader(
    "Sube una imagen de célula (JPG o PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        # ✅ CORRECCIÓN AQUÍ
        st.image(img, caption="Imagen cargada", use_column_width=True)

    with col2:
        st.markdown("### Información de la imagen")
        st.write(f"**Nombre:** {uploaded.name}")
        st.write(f"**Tipo:** {uploaded.type}")
        st.write(f"**Tamaño:** {round(uploaded.size / 1024, 2)} KB")

    if st.button("Predecir"):
        try:
            with st.spinner("Enviando imagen a la API y generando predicción..."):
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)

                files = {
                    "file": ("imagen.png", buffer, "image/png")
                }

                response = requests.post(API_URL, files=files, timeout=60)

            if response.status_code == 200:
                result = response.json()

                if "error" in result:
                    st.error(f"Error devuelto por la API: {result['error']}")
                    st.json(result)
                else:
                    label = result.get("label", "Sin resultado")
                    score = result.get("score", 0.0)
                    input_shape_model = result.get("input_shape_model", "No disponible")
                    prediction_shape = result.get("prediction_shape", "No disponible")
                    filename = result.get("filename", "No disponible")

                    st.markdown("## Resultado de la predicción")

                    if label == "Parasitized":
                        st.error(f"**Clase predicha:** {label}")
                    else:
                        st.success(f"**Clase predicha:** {label}")

                    st.metric("Confianza del modelo", f"{score:.2f}%")

                    st.markdown("### Detalles técnicos")
                    st.write(f"**Archivo enviado:** {filename}")
                    st.write(f"**Input esperado por el modelo:** {input_shape_model}")
                    st.write(f"**Forma de la salida del modelo:** {prediction_shape}")

                    st.markdown("### Respuesta completa de la API")
                    st.json(result)

            else:
                st.error(f"Error en la API: {response.status_code}")
                st.code(response.text)

        except requests.exceptions.ConnectionError:
            st.error(
                f"No fue posible conectar con la API. Verifica que API_BASE_URL sea correcta: {API_BASE_URL}"
            )
        except requests.exceptions.Timeout:
            st.error("La API tardó demasiado en responder.")
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

st.markdown("---")
st.caption(
    "Aplicación educativa para demostrar el flujo de despliegue de un modelo de Deep Learning con Streamlit + FastAPI."
)