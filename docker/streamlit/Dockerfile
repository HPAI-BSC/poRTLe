FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/

# Copy pre-loaded data (database + JSON results)
# These directories must exist when building the image
COPY results/portle.db ./results/portle.db
COPY results/json/ ./results/json/

# Enable demo mode to disable data modification buttons
ENV PORTLE_DEMO_MODE=true

EXPOSE 8501

CMD ["streamlit", "run", "src/ui/portle_ui.py", "--server.port=8501"]
