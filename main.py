import pyaudio
import numpy as np
import librosa
import scipy.io.wavfile as wav
import sys

p = pyaudio.PyAudio()

input = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

output = p.open(format=pyaudio.paInt16, channels=1, rate=44100, output=True, frames_per_buffer=1024)

def apply_robot_effect(audio_data):
    modulated = audio_data * np.sin(2 * np.pi * 50 * np.arange(len(audio_data)) / 44100)
    return modulated.astype(np.int16)

def apply_alien_effect(audio_data):
    audio_data_float = audio_data.astype(np.float32) / 32768.0  
    alien_audio = librosa.effects.pitch_shift(audio_data_float, sr=44100, n_steps=8)  
    return (alien_audio * 32768.0).astype(np.int16)  

def apply_chipmunk_effect(audio_data):
    audio_data_float = audio_data.astype(np.float32) / 32768.0  
    chipmunk_audio = librosa.effects.time_stretch(audio_data_float, rate=1.5)  
    return (chipmunk_audio * 32768.0).astype(np.int16)  

def apply_giant_effect(audio_data):
    audio_data_float = audio_data.astype(np.float32) / 32768.0  
    giant_audio = librosa.effects.time_stretch(audio_data_float, rate=0.7)  
    return (giant_audio * 32768.0).astype(np.int16)  

def apply_echo_effect(audio_data):
    echo = np.zeros_like(audio_data)
    echo_delay = 1024  
    echo[echo_delay:] = audio_data[:-echo_delay] * 0.5
    return (audio_data + echo).astype(np.int16)

def real_time_voice_changer(effect):
    recorded_audio = []  

    try:
        while True:
            mic_input_data = input.read(1024, exception_on_overflow=False)
            mic_signal = np.frombuffer(mic_input_data, dtype=np.int16)

            if effect == "robot":
                output_signal = apply_robot_effect(mic_signal)
            elif effect == "alien":
                output_signal = apply_alien_effect(mic_signal)
            elif effect == "chipmunk":
                output_signal = apply_chipmunk_effect(mic_signal)
            elif effect == "giant":
                output_signal = apply_giant_effect(mic_signal)
            elif effect == "echo":
                output_signal = apply_echo_effect(mic_signal)
            else:
                output_signal = mic_signal  

            output.write(output_signal.tobytes())
            recorded_audio.extend(output_signal)

    except KeyboardInterrupt:
        print("Stopping voice changer...")

    if recorded_audio:
        recorded_audio = np.array(recorded_audio, dtype=np.int16)
        output_file = f"output_{effect}.wav"
        wav.write(output_file, 44100, recorded_audio)
        print(f"Saved modified audio to {output_file}")

if __name__ == "__main__":
    print("Welcome to the Voice Changer!")
    print("Available effects: robot, alien, chipmunk, giant, echo")
    effect = input("Enter the effect you want to apply: ").strip().lower()

    if effect not in ["robot", "alien", "chipmunk", "giant", "echo"]:
        print("Invalid effect. Exiting...")
        sys.exit()

    print(f"Applying {effect} effect. Press Ctrl+C to stop.")
    real_time_voice_changer(effect)

input.stop_stream()
input.close()
output.stop_stream()
output.close()
p.terminate()