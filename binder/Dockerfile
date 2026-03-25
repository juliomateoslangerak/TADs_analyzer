# Dockerfile for TADs Analyzer
# Compatible with mybinder.org
# Using Jupyter base image for faster builds

FROM quay.io/jupyter/base-notebook:ubuntu-22.04

# Switch to root for installing packages
USER root

# Install Poetry
RUN pip install --no-cache-dir poetry==2.2.1

# Switch back to jovyan user
USER ${NB_USER}

# Set working directory
WORKDIR ${HOME}

# Copy poetry files and README (required by Poetry for package installation)
COPY pyproject.toml poetry.lock README.md ./

# Configure Poetry to not create virtual environments (not needed in container)
RUN poetry config virtualenvs.create false

# Install dependencies using Poetry (--no-root to skip installing the project itself)
RUN poetry install --only main --no-root --no-interaction --no-ansi

# Copy project files
COPY src/ ./src/

# Expose port 8888 (mybinder.org standard)
EXPOSE 8888

# Set environment variables for marimo
ENV MARIMO_HOST=0.0.0.0
ENV MARIMO_PORT=8888

# Default command: run find_rois_notebook.py with marimo
CMD ["marimo", "edit", "--host", "0.0.0.0", "--port", "8888", "--no-token", "src/find_rois_notebook.py"]
