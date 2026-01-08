FROM vf-base

WORKDIR /app

COPY services/monitor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/monitor/app.py .

CMD ["python", "app.py"]
