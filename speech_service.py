from threading import Thread, Event
from queue import Queue
from scipy.io.wavfile import write
from rich.console import Console
from transformers import AutoProcessor, BarkModel
from tqdm import tqdm
import numpy as np
import sounddevice as sd
from whisper import load_model
import torch
import re
import ollama
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

console = Console()

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

def truncate_text(text: str) -> list:
    return re.split(r'(?<=[.!?:;\-]) +', text)

def audio_synthesis_thread(text_queue: Queue, audio_queue: Queue, tts: TextToSpeechService, stop_event: Event):
    while not stop_event.is_set():
        try:
            text_piece = text_queue.get(timeout=0.05)
            sample_rate, audio_array = tts.synthesize(text_piece, "v2/en_speaker_9")
            audio_queue.put((sample_rate, audio_array))
        except Exception:
            continue

def audio_playback_thread(audio_queue: Queue, stop_event: Event):
    while not stop_event.is_set():
        try:
            sample_rate, audio_array = audio_queue.get(timeout=0.01)
            sd.play(audio_array, sample_rate)
            sd.wait()
        except Exception:
            continue

def record_audio(sample_rate, channels, audio_queue, stop_event):
    console.print("[green]Recording...")
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        audio_queue.put(indata.copy())
    with sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback):
        stop_event.wait()
    console.print("[green]Finished recording.")

def transcribe_audio(filename: str) -> tuple:
    model = load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
    result = model.transcribe(filename)
    return (result['text'], result['language'])

def stream_llm_response(conversation: list, model_name: str, text_queue: Queue):
    messages = [{"role": "user", "content": msg} for msg in conversation]
    response = ollama.chat(model=model_name, messages=messages, stream=True)
    buffer = ""
    for partial_response in response:
        if "message" in partial_response and "content" in partial_response["message"]:
            content = partial_response["message"]["content"]
            console.print(content, end="")
            buffer += content
            sentences = truncate_text(buffer)
            for sentence in sentences[:-1]:
                text_queue.put(sentence)
            buffer = sentences[-1]
    if buffer.strip():
        text_queue.put(buffer.strip())
    console.print()

def ollama_pull(model_name: str):
    console.print(f"[blue]Downloading model '{model_name}'...")
    current_digest, bars = '', {}
    for progress in ollama.pull(model_name, stream=True):
        digest = progress.get('digest', '')
        if digest != current_digest and current_digest in bars:
            bars[current_digest].close()
        if not digest:
            console.print(progress.get('status'))
            continue
        if digest not in bars and (total := progress.get('total')):
            bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', unit='B', unit_scale=True)
        if completed := progress.get('completed'):
            bars[digest].update(completed - bars[digest].n)
        current_digest = digest
    console.print(f"[green]Model '{model_name}' downloaded!")

if __name__ == "__main__":
    tts = TextToSpeechService()
    if not torch.cuda.is_available():
        console.print("[yellow]CUDA is not available. Running on CPU.")
    else:
        console.print("[yellow]CUDA is available. Running on GPU.")
    sample_rate = 22050
    channels = 1
    filename = "output/user.wav"
    Path("output").mkdir(parents=True, exist_ok=True)
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")
    model_name = console.input("[blue]Enter the model name to download (e.g., 'mistral-nemo:12b-instruct-2407-q2_K'): ")
    ollama_pull(model_name)
    conversation = []
    text_queue = Queue()
    audio_queue = Queue()
    stop_event = Event()
    synth_thread = Thread(target=audio_synthesis_thread, args=(text_queue, audio_queue, tts, stop_event))
    play_thread = Thread(target=audio_playback_thread, args=(audio_queue, stop_event))
    synth_thread.start()
    play_thread.start()
    try:
        while True:
            console.input("[blue]Press Enter to start recording...")
            audio_queue = Queue()
            recording_thread = Thread(target=record_audio, args=(sample_rate, channels, audio_queue, stop_event))
            recording_thread.start()
            console.input("[blue]Press Enter to stop recording...")
            stop_event.set()
            recording_thread.join()
            console.print("[green]Processing audio...")
            frames = [audio_queue.get() for _ in range(audio_queue.qsize())]
            if not frames:
                console.print("[red]No audio recorded. Please ensure your microphone is working.")
                continue
            audio_data = np.concatenate(frames, axis=0)
            write(filename, sample_rate, audio_data)
            console.print(f"[green]Transcribing...")
            text, language = transcribe_audio(filename)
            console.print(f"[yellow]You said: {text}")
            conversation.append(text)
            console.print("[green]Assistant response:")
            stream_llm_response(conversation, model_name, text_queue)
    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")
        stop_event.set()
        synth_thread.join()
        play_thread.join()
    except Exception as e:
        console.print(f"[red]An error occurred: {e}")
    console.print("[blue]Session ended.")
