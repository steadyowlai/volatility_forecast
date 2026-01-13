FROM vf-base

WORKDIR /app

#training service requirements
COPY services/train/requirements.txt /app/services/train/requirements.txt

RUN pip install --no-cache-dir -r services/train/requirements.txt

# Copy service code (self-contained with experiment_tracker.py)
COPY services/train/ /app/services/train/
COPY final_model/ /app/final_model/

#default command
CMD ["python", "services/train/app.py"]
