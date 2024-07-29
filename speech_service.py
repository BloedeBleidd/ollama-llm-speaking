import time
import threading
import numpy as np
import sounddevice as sd
from queue import Queue
from rich.console import Console
from transformers import AutoProcessor, BarkModel
from subprocess import Popen, PIPE
import whisper
import torch

console = Console()
stt = whisper.load_model("base")

class TextToSpeechService:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small")
        self.model.to(self.device)

    def synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1"):
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            audio_array = self.model.generate(**inputs, pad_token_id=10000)
        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate
        return sample_rate, audio_array

    def long_form_synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1"):
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.model.generation_config.sample_rate))
        for sent in sentences:
            sample_rate, audio_array = self.synthesize(sent, voice_preset)
            pieces += [audio_array, silence.copy()]
        return self.model.generation_config.sample_rate, np.concatenate(pieces)

tts = TextToSpeechService()

def record_audio(stop_event, data_queue):
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))
    with sd.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)

def transcribe(audio_np: np.ndarray) -> str:
    result = stt.transcribe(audio_np, fp16=False)
    text = result["text"].strip()
    return text

def get_llm_response(text: str) -> str:
    process = Popen(["docker", "exec", "-i", "ollama", "ollama", "run", "mistral-nemo:12b-instruct-2407-q5_K_M"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate(input=text.encode())
    response = stdout.decode().strip()
    if response.startswith("Assistant:"):
        response = response[len("Assistant:"):].strip()
    return response

def play_audio(sample_rate, audio_array):
    sd.play(audio_array, sample_rate)
    sd.wait()

if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    data_queue = Queue()
    stop_event = threading.Event()
    recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue))
    recording_thread.start()

    try:
        while True:
            time.sleep(5)  # Record for 5 seconds, adjust as needed
            stop_event.set()
            recording_thread.join()
            audio_data = b"".join(list(data_queue.queue))
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                with console.status("Generating response...", spinner="earth"):
                    response = get_llm_response(text)
                    sample_rate, audio_array = tts.long_form_synthesize(response)

                console.print(f"[cyan]Assistant: {response}")
                play_audio(sample_rate, audio_array)
            else:
                console.print("[red]No audio recorded. Please ensure your microphone is working.")

            # Reset for next recording
            data_queue = Queue()
            stop_event = threading.Event()
            recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue))
            recording_thread.start()

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")
        stop_event.set()
        recording_thread.join()

    console.print("[blue]Session ended.")
