# Build the image (from repo root)
docker build -f docker/streamlit/Dockerfile -t portle-ui .

# Run the container (port 8501 or any available port)
docker run -d -p 8502:8501 --name portle-demo portle-ui

# View the app
http://localhost:8502/

# Stop and remove
docker stop portle-demo && docker rm portle-demo

# View logs
docker logs portle-demo