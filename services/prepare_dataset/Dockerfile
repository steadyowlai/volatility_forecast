FROM vf-base

WORKDIR /app

# Install dependencies
COPY services/prepare_dataset/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/prepare_dataset/app.py .

CMD ["python", "app.py"]
