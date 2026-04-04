FROM python:3.12-slim

WORKDIR /app

COPY ./streamlit_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY src/ /app/src/
COPY streamlit_app/ /app/
ENV PYTHONPATH=/app
ENV GOOGLE_CLIENT_ID=1053008146027-e3vi5srg82j69h5tt3bsbcask0bcoq09.apps.googleusercontent.com
ENV GOOGLE_CLIENT_SECRET=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
ENV GOOGLE_REDIRECT_URI=http://localhost:8501
ENV API_URL=http://predict-api:8000/predict

EXPOSE 8501

CMD ["streamlit", "run", "app.py","--server.port=8501", "--server.address=0.0.0.0"]


