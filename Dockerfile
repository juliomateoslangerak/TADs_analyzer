# Dockerfile for TADs Analyzer
# Compatible with mybinder.org

FROM python:3.12-slim

# Set working directory
WORKDIR /home/jovyan

# Install system dependencies and Poetry
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir poetry==2.2.1

# Copy poetry files and README (required by Poetry for package installation)
COPY pyproject.toml poetry.lock README.md ./

# Configure Poetry to not create virtual environments (not needed in container)
RUN poetry config virtualenvs.create false

# Install dependencies using Poetry
RUN poetry install --only main --no-interaction --no-ansi

# Copy project files
COPY src/ ./src/

# Expose marimo default port
EXPOSE 8080

# Set environment variables for marimo
ENV MARIMO_HOST=0.0.0.0
ENV MARIMO_PORT=8080

# Default command: run find_rois_notebook.py with marimo
CMD ["marimo", "edit", "--host", "0.0.0.0", "--port", "8080", "--no-token", "src/find_rois_notebook.py"]
