FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Install dependencies
RUN pip install --no-cache-dir transformers peft accelerate runpod

# Copy handler
COPY runpod_handler.py /handler.py

# Set working directory
WORKDIR /

# Command
CMD ["python", "-u", "/handler.py"]
