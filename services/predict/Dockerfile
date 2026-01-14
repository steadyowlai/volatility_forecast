FROM vf-base

WORKDIR /app

COPY services/predict/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code (self-contained with experiment_tracker.py, storage.py, and prediction_status.py)
COPY services/predict/app.py .
COPY services/predict/experiment_tracker.py .
COPY services/predict/storage.py .
COPY services/predict/prediction_status.py .

CMD ["python", "app.py"]
