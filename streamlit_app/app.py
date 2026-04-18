from marshal import load

import streamlit as st
import requests
from PIL import Image
import io
import os
from authlib.integrations.requests_client import OAuth2Session
from dotenv import load_dotenv


# load_dotenv()


CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

# ---- Page title ----
st.title("SPE Image Segmentation App")

# ---- LOGIN FLOW ----

if "token" not in st.session_state:

    oauth = OAuth2Session(
        CLIENT_ID,
        CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope="openid email profile"
    )

    authorization_url, state = oauth.create_authorization_url(AUTHORIZATION_ENDPOINT)

    st.markdown(f"[👉 Login with Google]({authorization_url})")

    if "code" in st.query_params:
        code = st.query_params["code"]

        token = oauth.fetch_token(
            TOKEN_ENDPOINT,
            code=code,
            grant_type="authorization_code"
        )

        # ✅ STORE TOKEN
        st.session_state.token = token

        st.success("Authentication successful!")

        st.query_params.clear()
        st.rerun()

    st.stop()


# ---- AFTER LOGIN ----

st.success("You are logged in!")

id_token = st.session_state.token["id_token"]


# ---- File uploader ----
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)


# ---- Predict button ----
if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):

        headers = {
            "Authorization": f"Bearer {id_token}"
        }

        files = {
            "file": uploaded_file.getvalue()
        }
        API_URL=os.getenv("API_URL")
        response = requests.post(
            f"http://{API_URL}:8000/predict/",
            files=files,
            headers=headers
        )

        if response.status_code == 200:

            image=Image.open(io.BytesIO(response.content))

            predicted_class = response.headers.get("X-Predicted-Class", "N/A")
            confidence = response.headers.get("X-Confidence", "N/A")

            st.markdown(f"**Predicted Class:** {predicted_class}")
            st.markdown(f"**Confidence:** {confidence*100}%")


            st.image(image, caption="Prediction Output")

        else:
            st.error(f"Prediction failed: {response.text}")
