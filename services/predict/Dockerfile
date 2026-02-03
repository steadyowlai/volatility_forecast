FROM vf-base

WORKDIR /app

COPY services/predict/requirements.txt /app/services/predict/requirements.txt
RUN pip install --no-cache-dir -r /app/services/predict/requirements.txt

# Copy service code
COPY services/predict/ /app/services/predict/

CMD ["python", "services/predict/app.py"]
