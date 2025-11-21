# FROM vf-base AS ingest:
# ingest is an internal name. you could do from vf-base import xyz, keep everything same and nothing would break
# we give the actual name to the image [vf-ingest for this] during the docker build command
# look at docs/commands.txt


FROM vf-base AS ingest 
# ---- Ingest-specific dependencies (base deps already in vf-base) ----
COPY services/ingest/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---- Working directory ----
WORKDIR /app

# ---- Copy project code (relative to build context = repo root) ----
COPY libs/ ./libs/
COPY services/ingest/ ./services/ingest/
COPY config/ ./config/

# ---- Set PYTHONPATH so Python can find the libs module ----
ENV PYTHONPATH=/app

# ---- Default command ----
# "When someone runs a container from this image, execute this command by default."
# So if you run: docker run vf-ingest, it will automatically run this command: python services/ingest/app.py
CMD ["python", "services/ingest/app.py"]
