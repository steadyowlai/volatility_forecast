FROM vf-base AS ingest

# ---- Ingest-specific dependencies (base deps already in vf-base) ----
COPY requirements/ingest.txt /tmp/ingest.txt
RUN pip install --no-cache-dir -r /tmp/ingest.txt

# ---- Working directory ----
WORKDIR /app

# ---- Copy project code (relative to build context =) repo root ----
COPY libs/ ./libs/
COPY services/ingest/ ./services/ingest/
COPY config/ ./config/

# ---- Default command ----
CMD ["python", "services/ingest/app.py"]