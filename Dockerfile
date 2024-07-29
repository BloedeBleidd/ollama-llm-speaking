# Dockerfile
FROM python:3.12-slim

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1-dev \
    git \
    portaudio19-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install \
    numpy \
    sounddevice \
    rich \
    torch \
    transformers \
    nltk

# Install Whisper from GitHub
RUN pip install git+https://github.com/openai/whisper.git

# Clone and set up Bark Model
RUN git clone https://github.com/suno-ai/bark.git && cd bark && pip install -e .

# Copy the script to the container
COPY . /app
WORKDIR /app

# Run the speech service script
CMD ["python3", "speech_service.py"]
