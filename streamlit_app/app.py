import io
import os

import requests
import streamlit as st
from authlib.integrations.requests_client import OAuth2Session
from PIL import Image


CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "").strip()
API_URL = os.getenv("API_URL")

AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

print(f"Loaded OAuth settings: CLIENT_ID={'set' if CLIENT_ID else 'missing'}, "
      f"CLIENT_SECRET={'set' if CLIENT_SECRET else 'missing'}, "
      f"REDIRECT_URI={'set' if REDIRECT_URI else 'missing'}")

st.title("SPE Image Segmentation App")


if "token" not in st.session_state:
    missing_oauth_settings = [
        name
        for name, value in {
            "GOOGLE_CLIENT_ID": CLIENT_ID,
            "GOOGLE_CLIENT_SECRET": CLIENT_SECRET,
            "GOOGLE_REDIRECT_URI": REDIRECT_URI,
        }.items()
        if not value
    ]

    if missing_oauth_settings:
        st.error(f"Missing OAuth settings: {', '.join(missing_oauth_settings)}")
        st.stop()

    oauth_error = st.query_params.get("error")
    if oauth_error:
        error_description = st.query_params.get("error_description", oauth_error)
        st.error(f"Google OAuth failed: {error_description}")
        st.stop()

    code = st.query_params.get("code")
    if code:
        callback_oauth = OAuth2Session(
            CLIENT_ID,
            CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope="openid email profile",
        )

        try:
            st.session_state.token = callback_oauth.fetch_token(
                TOKEN_ENDPOINT,
                code=code,
                redirect_uri=REDIRECT_URI,
                grant_type="authorization_code",
            )
        except Exception as exc:
            st.error(f"Google token exchange failed: {exc}")
            st.stop()

        st.query_params.clear()
        st.rerun()

    login_oauth = OAuth2Session(
        CLIENT_ID,
        CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope="openid email profile",
    )
    authorization_url, _ = login_oauth.create_authorization_url(
        AUTHORIZATION_ENDPOINT,
        prompt="select_account",
    )

    st.link_button("Login with Google", authorization_url)
    st.stop()


st.success("You are logged in!")

id_token = st.session_state.token["id_token"]

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"],
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        headers = {
            "Authorization": f"Bearer {id_token}",
        }
        files = {
            "file": uploaded_file.getvalue(),
        }

        response = requests.post(
            f"http://{API_URL}:8000/predict/",
            files=files,
            headers=headers,
        )

        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            predicted_class = response.headers.get("X-Predicted-Class", "N/A")
            confidence = response.headers.get("X-Confidence", "N/A")

            st.markdown(f"**Predicted Class:** {predicted_class}")
            st.markdown(f"**Confidence:** {confidence}%")
            st.image(image, caption="Prediction Output")
        else:
            st.error(f"Prediction failed: {response.text}")
