from time import sleep
from threading import Thread, Event
import numpy as np
from sounddevice import RawInputStream, play, wait
from queue import Queue
from rich.console import Console
from transformers import AutoProcessor, BarkModel
import whisper
import torch
import nltk
import ollama
from tqdm import tqdm
import scipy.io.wavfile as wav

console = Console()
stt = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")

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
        data_queue.put(np.array(indata, dtype=np.int16))
    with RawInputStream(samplerate=44100, dtype="int16", channels=1, callback=callback):
        while not stop_event.is_set():
            sleep(0.1)

def save_audio_to_wav(audio_data, filename="output/user.wav", samplerate=44100):
    wav.write(filename, samplerate, audio_data)
    console.print(f"[green]Audio saved as '{filename}'")

def transcribe(filename: str) -> str:
    result = stt.transcribe(filename, fp16=False)
    text = result["text"].strip()
    return text

def get_llm_response(conversation: list, model_name: str) -> str:
    messages = [{"role": "user", "content": msg} for msg in conversation]
    response = ollama.chat(model=model_name, messages=messages)
    return response['message']['content']

def play_audio(sample_rate, audio_array):
    play(audio_array, sample_rate)
    wait()

if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")
    model_name = console.input("[blue]Enter the model name to download (e.g., 'mistral-nemo:12b-instruct-2407-q2_K'): ")
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
    conversation = []  # To hold the context of the conversation

    while True:
        try:
            console.print("[blue]Press Enter to start recording, and Enter again to stop...")
            input("Press Enter to start recording...")

            data_queue = Queue()
            stop_event = Event()
            recording_thread = Thread(target=record_audio, args=(stop_event, data_queue))
            recording_thread.start()

            input("Press Enter to stop recording...")
            stop_event.set()
            recording_thread.join()

            console.print("[green]Processing audio...")

            audio_data = np.concatenate(list(data_queue.queue), axis=0)
            audio_data = np.asarray(audio_data, dtype=np.int16)

            # Save the audio to a WAV file
            save_audio_to_wav(audio_data)

            if audio_data.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe("output/user.wav")
                console.print(f"[yellow]You: {text}")

                conversation.append(text)

                with console.status("Generating response...", spinner="earth"):
                    response = get_llm_response(conversation, model_name)
                    conversation.append(response)
                    sample_rate, audio_array = tts.long_form_synthesize(response)

                console.print(f"[cyan]Assistant: {response}")
                play_audio(sample_rate, audio_array)
            else:
                console.print("[red]No audio recorded. Please ensure your microphone is working.")

        except KeyboardInterrupt:
            console.print("\n[red]Exiting...")
            break

    console.print("[blue]Session ended.")
