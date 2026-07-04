import io
import logging
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

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

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
        logger.warning("Google OAuth failed: %s", error_description)
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
            logger.info("Exchanging Google OAuth code for token")
            st.session_state.token = callback_oauth.fetch_token(
                TOKEN_ENDPOINT,
                code=code,
                redirect_uri=REDIRECT_URI,
                grant_type="authorization_code",
            )
            logger.info("Google OAuth token exchange succeeded")
        except Exception as exc:
            logger.exception("Google token exchange failed")
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
    logger.info("Rendered Google login button")
    st.stop()


st.success("You are logged in!")

id_token = st.session_state.token["id_token"]

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"],
)

if uploaded_file is not None:
    logger.info("Image uploaded: name=%s, type=%s, size=%d", uploaded_file.name, uploaded_file.type, uploaded_file.size)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        logger.info("Submitting prediction request for uploaded file: %s", uploaded_file.name)
        headers = {
            "Authorization": f"Bearer {id_token}",
        }
        files = {
            "file": uploaded_file.getvalue(),
        }

        try:
            response = requests.post(
                f"http://{API_URL}:8000/predict/",
                files=files,
                headers=headers,
                timeout=60,
            )
        except requests.RequestException as exc:
            logger.exception("Prediction request failed")
            st.error(f"Prediction request failed: {exc}")
            st.stop()

        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            predicted_class = response.headers.get("X-Predicted-Class", "N/A")
            confidence = response.headers.get("X-Confidence", "N/A")

            st.markdown(f"**Predicted Class:** {predicted_class}")
            st.markdown(f"**Confidence:** {confidence}%")
            st.image(image, caption="Prediction Output")
            logger.info("Prediction succeeded: class=%s, confidence=%s", predicted_class, confidence)
        else:
            logger.error("Prediction failed: status=%d, body=%s", response.status_code, response.text)
            st.error(f"Prediction failed: {response.text}")
