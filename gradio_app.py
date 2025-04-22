from orpheus_tts import OrpheusModel
import wave
import time
import gradio as gr
import numpy as np
import io
import os
import torch
import random

# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seeds at startup
set_seeds()

# Load the model globally
model = OrpheusModel(model_name="canopylabs/orpheus-3b-0.1-ft")

def text_to_speech(text, seed, progress=gr.Progress()):
    # Set the seed for reproducibility
    set_seeds(seed)

    start_time = time.monotonic()

    # Generate speech with deterministic parameters
    syn_tokens = model.generate_speech(
        prompt=text,
        voice="tara",
        temperature=0.01,  # Nearly deterministic temperature
    )

    # Create an in-memory buffer for the wave file
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        total_frames = 0
        for audio_chunk in progress.tqdm(syn_tokens, desc="Generating audio"):
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)

        duration = total_frames / wf.getframerate()

    end_time = time.monotonic()
    generation_time = end_time - start_time

    # Return the audio and status message
    return (
        (24000, np.frombuffer(buffer.getvalue(), dtype=np.int16)),
        f"Generated {duration:.2f} seconds of audio in {generation_time:.2f} seconds with seed {seed}"
    )

# Gradio interface with seed input
demo = gr.Interface(
    fn=text_to_speech,
    inputs=[
        gr.Textbox(
            label="Enter text to convert to speech",
            placeholder="Type your text here...",
            lines=5
        ),
        gr.Number(
            label="Random Seed",
            value=42,
            precision=0
        )
    ],
    outputs=[
        gr.Audio(label="Generated Speech", type="numpy"),
        gr.Textbox(label="Status")
    ],
    title="Orpheus Text-to-Speech",
    description="Convert text to natural-sounding speech using the Orpheus TTS model. Use the same seed for consistent results."
)

if __name__ == '__main__':
    demo.launch(share=True)