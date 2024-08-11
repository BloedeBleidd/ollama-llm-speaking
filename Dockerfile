FROM python:3.12-slim

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1-dev \
    git \
    portaudio19-dev \
    curl \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with specific versions
RUN pip install --upgrade pip &&\
    pip install --no-cache-dir \
    numpy==2.0.1 \
    sounddevice==0.4.7 \
    rich==13.7.1 \
    torch==2.4.0 \
    transformers==4.44.0 \
    tqdm==4.66.5 \
    ollama==0.3.1 \
    openai-whisper==20230306 \
    bark==0.1.5

# Preload the Whisper model and other transformers models during build
RUN python -c "import whisper; whisper.load_model('base')"

# Optionally, preload Bark or other Transformers models if they are large
RUN python -c "from transformers import AutoProcessor, BarkModel; AutoProcessor.from_pretrained('suno/bark-small'); BarkModel.from_pretrained('suno/bark-small')"

# Install Ollama CLI (this will be used to pull models at runtime)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy the script to the container
COPY speech_service.py /app/speech_service.py
WORKDIR /app

# Start the Ollama server and run the speech service script
CMD ["/bin/bash", "-c", "ollama serve > /dev/null 2>&1 & sleep 5 && python speech_service.py"]
