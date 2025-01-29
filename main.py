import sys
import numpy as np
import pyaudio
import librosa
import scipy.io.wavfile as wav
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QWidget)
from PyQt5.QtCore import QThread, pyqtSignal

class VoiceChangerThread(QThread):
    def __init__(self, effect):
        super().__init__()
        self.effect = effect
        self.is_running = False
        self.p = pyaudio.PyAudio()
        self.recorded_audio = []

    def run(self):
        self.is_running = True
        audio_input = self.p.open(format=pyaudio.paInt16, channels=1, rate=44100, 
                                  input=True, frames_per_buffer=1024)
        output = self.p.open(format=pyaudio.paInt16, channels=1, rate=44100, 
                             output=True, frames_per_buffer=1024)

        while self.is_running:
            try:
                mic_input_data = audio_input.read(1024, exception_on_overflow=False)
                mic_signal = np.frombuffer(mic_input_data, dtype=np.int16)
                
                if self.effect == "robot":
                    output_signal = self.apply_robot_effect(mic_signal)
                elif self.effect == "alien":
                    output_signal = self.apply_alien_effect(mic_signal)
                elif self.effect == "chipmunk":
                    output_signal = self.apply_chipmunk_effect(mic_signal)
                elif self.effect == "giant":
                    output_signal = self.apply_giant_effect(mic_signal)
                elif self.effect == "echo":
                    output_signal = self.apply_echo_effect(mic_signal)
                else:
                    output_signal = mic_signal
                
                output.write(output_signal.tobytes())
                self.recorded_audio.extend(output_signal)
            
            except Exception:
                break

        audio_input.stop_stream()
        audio_input.close()
        output.stop_stream()
        output.close()
        self.p.terminate()

    def stop(self):
        self.is_running = False
        self.wait()
        
        if self.recorded_audio:
            recorded_audio = np.array(self.recorded_audio, dtype=np.int16)
            output_file = f"output_{self.effect}.wav"
            wav.write(output_file, 44100, recorded_audio)
            print(f"Saved modified audio to {output_file}")

    def apply_robot_effect(self, audio_data):
        modulated = audio_data * np.sin(2 * np.pi * 50 * np.arange(len(audio_data)) / 44100)
        return modulated.astype(np.int16)

    def apply_alien_effect(self, audio_data):
        audio_data_float = audio_data.astype(np.float32) / 32768.0
        alien_audio = librosa.effects.pitch_shift(audio_data_float, sr=44100, n_steps=8)
        return (alien_audio * 32768.0).astype(np.int16)

    def apply_chipmunk_effect(self, audio_data):
        audio_data_float = audio_data.astype(np.float32) / 32768.0
        chipmunk_audio = librosa.effects.time_stretch(audio_data_float, rate=1.5)
        return (chipmunk_audio * 32768.0).astype(np.int16)

    def apply_giant_effect(self, audio_data):
        audio_data_float = audio_data.astype(np.float32) / 32768.0
        giant_audio = librosa.effects.time_stretch(audio_data_float, rate=0.7)
        return (giant_audio * 32768.0).astype(np.int16)

    def apply_echo_effect(self, audio_data):
        echo = np.zeros_like(audio_data)
        echo_delay = 1024
        echo[echo_delay:] = audio_data[:-echo_delay] * 0.5
        return (audio_data + echo).astype(np.int16)

class VoiceChangerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Changer")
        self.setGeometry(100, 100, 300, 200)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # Effect Selector
        effect_layout = QHBoxLayout()
        self.effect_label = QLabel("Select Effect:")
        self.effect_combo = QComboBox()
        self.effect_combo.addItems(["robot", "alien", "chipmunk", "giant", "echo"])
        effect_layout.addWidget(self.effect_label)
        effect_layout.addWidget(self.effect_combo)
        layout.addLayout(effect_layout)

        # Start/Stop Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        
        self.start_button.clicked.connect(self.start_voice_changer)
        self.stop_button.clicked.connect(self.stop_voice_changer)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        central_widget.setLayout(layout)
        self.voice_changer_thread = None

    def start_voice_changer(self):
        effect = self.effect_combo.currentText()
        self.voice_changer_thread = VoiceChangerThread(effect)
        self.voice_changer_thread.start()
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_voice_changer(self):
        if self.voice_changer_thread:
            self.voice_changer_thread.stop()
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

def main():
    app = QApplication(sys.argv)
    voice_changer = VoiceChangerApp()
    voice_changer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()