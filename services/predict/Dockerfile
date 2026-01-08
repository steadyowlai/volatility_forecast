FROM vf-base

WORKDIR /app

COPY services/predict/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/predict/app.py .

CMD ["python", "app.py"]
