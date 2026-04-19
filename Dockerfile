FROM python:3.10

# Create user to comply with Hugging Face Spaces requirements
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy the environment over
COPY --chown=user . $HOME/app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default port for Hugging Face Spaces
EXPOSE 7860

# Start the application
CMD ["uvicorn", "src.rag_api:app", "--host", "0.0.0.0", "--port", "7860"]
