FROM vf-base

WORKDIR /app

# Install dependencies
COPY services/prepare_dataset/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code (self-contained with data_status.py)
COPY services/prepare_dataset/app.py .
COPY services/prepare_dataset/data_status.py .

CMD ["python", "app.py"]
