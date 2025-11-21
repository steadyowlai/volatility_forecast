FROM vf-base AS features

#no additional dependencies needed beyond base
COPY services/features/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

# copy libs and service code
COPY libs/ ./libs/
COPY services/features/ ./services/features/
COPY config/ ./config/

ENV PYTHONPATH=/app

CMD ["python", "services/features/app.py"]
