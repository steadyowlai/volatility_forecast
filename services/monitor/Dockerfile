FROM vf-base

WORKDIR /app

COPY services/monitor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code (self-contained with experiment_tracker.py)
COPY services/monitor/app.py .
COPY services/monitor/experiment_tracker.py .

CMD ["python", "app.py"]
