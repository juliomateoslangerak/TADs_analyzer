# Dockerfile for TADs Analyzer
# Compatible with mybinder.org

FROM python:3.12-slim

# Create jovyan user (mybinder.org standard)
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER}
ENV HOME=/home/${NB_USER}

RUN adduser --disabled-password --gecos "Default user" --uid ${NB_UID} ${NB_USER}

# Install system dependencies and Poetry
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir poetry==2.2.1

# Copy poetry files and README (as root to ensure proper permissions)
COPY --chown=${NB_USER}:${NB_USER} pyproject.toml poetry.lock README.md ${HOME}/

# Switch to jovyan user
USER ${NB_USER}

# Set working directory
WORKDIR ${HOME}

# Configure Poetry to not create virtual environments (not needed in container)
RUN poetry config virtualenvs.create false

# Install dependencies using Poetry (--no-root to skip installing the project itself)
RUN poetry install --only main --no-root --no-interaction --no-ansi

# Switch back to root to copy files
USER root

# Copy project files
COPY --chown=${NB_USER}:${NB_USER} src/ ${HOME}/src/

# Switch back to jovyan for runtime
USER ${NB_USER}

# Expose port 8888 (mybinder.org standard)
EXPOSE 8888

# Set environment variables for marimo
ENV MARIMO_HOST=0.0.0.0
ENV MARIMO_PORT=8888

# Default command: run find_rois_notebook.py with marimo
CMD ["marimo", "edit", "--host", "0.0.0.0", "--port", "8888", "--no-token", "src/find_rois_notebook.py"]
