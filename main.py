import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import wave

p = pyaudio.PyAudio()

input = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

def load_output_audio(file_path):
    with wave.open(file_path, 'rb') as f:
        sample_rate = f.getframerate()
        data = f.readframes(f.getnframes())
        signal = np.frombuffer(data, dtype=np.int16)
    return signal, sample_rate

speaker_output_signal, sample_rate = load_output_audio("test.wav")
speaker_output_index = 0

output_audio = []

def lms_filter(x, d, mu=0.001, filter_order=64):
    n = len(x)
    w = np.zeros(filter_order, dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    e = np.zeros(n, dtype=np.float32)

    for i in range(filter_order, n):
        x_slice = x[i:i-filter_order:-1]
        y[i] = np.dot(w, x_slice)
        e[i] = d[i] - y[i]
        w += mu * e[i] * x_slice
        w = np.clip(w, -1.0, 1.0)  

    return e

def echo_cancel(mic_input, speaker_output, mu=0.001, filter_order=64):
    return lms_filter(speaker_output, mic_input, mu, filter_order)

def real_time_noise_cancellation():
    global speaker_output_index
    try:
        while True:
            mic_input_data = input.read(1024, exception_on_overflow=False)
            mic_signal = np.frombuffer(mic_input_data, dtype=np.int16).astype(np.float32) / 32768.0

            if speaker_output_index + 1024 > len(speaker_output_signal):
                speaker_output_index = 0  
            speaker_signal = speaker_output_signal[speaker_output_index:speaker_output_index + 1024].astype(np.float32) / 32768.0
            speaker_output_index += 1024

            cleaned_signal = echo_cancel(mic_signal, speaker_signal)

            output_audio.extend((cleaned_signal * 32768.0).astype(np.int16))  

    except KeyboardInterrupt:
        print("Exiting...")


real_time_noise_cancellation()

output_audio = np.array(output_audio, dtype=np.int16)
wav.write("output.wav", sample_rate, output_audio)

plt.plot(output_audio)
plt.title("Output Audio Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

input.stop_stream()
input.close()
p.terminate()