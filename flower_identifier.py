import base64
import io
import json
import os
from typing import Optional

from PIL import Image
import streamlit as st
from openai import OpenAI


# -----------------------------
# Page config
# --------------------------([developers.openai.com](https://developers.openai.com/api/reference/responses/overview/?utm_source=chatgpt.com))wer Name & Species Identifier",
    page_icon="🌼",
    layout="centered",
)

st.title("🌼 Flower Name & Species Identifier")
st.caption("Take a flower photo and let OpenAI identify the flower and its likely species.")


# -----------------------------
# OpenAI setup
# -----------------------------
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise ValueError(
            "Missing OPENAI_API_KEY. Add it as an environment variable or in Streamlit secrets."
        )
    return OpenAI(api_key=api_key)


# -----------------------------
# Helpers
# -----------------------------
def image_to_data_url(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def detect_mime_type(filename: Optional[str]) -> str:
    if not filename:
        return "image/jpeg"
    lower = filename.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


def identify_flower(client: OpenAI, image_bytes: bytes, mime_type: str) -> dict:
    image_data_url = image_to_data_url(image_bytes, mime_type)

    prompt = """
You are a botanical image identification assistant.
Analyze the flower photo and return JSON only.

Required JSON schema:
{
  "common_name": "string",
  "likely_species": "string",
  "confidence": 0.0,
  "description": "short description",
  "carefully_worded_note": "brief uncertainty note if needed"
}

Rules:
- Identify the visible flower as accurately as possible.
- For likely_species, use the scientific name when possible.
- confidence must be a number from 0 to 1.
- If uncertain, say the most likely answer and mention uncertainty briefly.
- Return valid JSON only. No markdown.
""".strip()

    response = client.responses.create(
        model="gpt-5.4",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
    )

    text = response.output_text.strip()
    return json.loads(text)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Setup")
    st.markdown(
        """
1. Install dependencies:
   ```bash
   pip install streamlit openai pillow
   ```
2. Set your API key:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```
3. Run the app:
   ```bash
   streamlit run flower_camera_identifier_app.py
   ```
        """
    )

    st.header("How it works")
    st.write(
        "The app sends your camera image to OpenAI, then asks the model to identify "
        "the flower's common name and likely species."
    )

    st.warning("API usage may incur cost depending on your OpenAI account and model usage.")


# -----------------------------
# Main app
# -----------------------------
try:
    client = get_client()
except Exception as exc:
    st.error(str(exc))
    st.stop()

camera_photo = st.camera_input("Take a flower photo")
uploaded_file = st.file_uploader("Or upload a flower image", type=["jpg", "jpeg", "png", "webp"])

image_source = camera_photo if camera_photo is not None else uploaded_file

if image_source is not None:
    image_bytes = image_source.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    mime_type = detect_mime_type(getattr(image_source, "name", None))

    st.subheader("Image")
    st.image(image, use_container_width=True)

    with st.spinner("Identifying flower..."):
        try:
            result = identify_flower(client, image_bytes, mime_type)
        except json.JSONDecodeError:
            st.error("The model returned an unexpected response format. Please try again.")
            st.stop()
        except Exception as exc:
            st.error(f"OpenAI request failed: {exc}")
            st.stop()

    common_name = result.get("common_name", "Unknown")
    likely_species = result.get("likely_species", "Unknown")
    confidence = float(result.get("confidence", 0))
    description = result.get("description", "")
    note = result.get("carefully_worded_note", "")

    st.success(f"Identified flower: **{common_name}**")
    st.write(f"**Likely species:** {likely_species}")
    st.metric("Confidence", f"{confidence * 100:.1f}%")

    if description:
        st.subheader("Description")
        st.write(description)

    if note:
        st.subheader("Note")
        st.write(note)
else:
    st.info("Use your camera or upload an image to identify a flower.")
