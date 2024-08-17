#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import openai
import elevenlabs
import pyaudio
import wave
import numpy as np
import collections
import faster_whisper
import torch.cuda
import os
import time
import pygame
from gtts import gTTS
import ollama
import chromadb
from openai import OpenAI
import locale
import json
locale.getpreferredencoding = lambda: "UTF-8"
import rclpy
import argparse
import queue
from rclpy.node import Node
from std_msgs.msg import Bool, String
import sounddevice as sd
from submodules.utilities import *

class Speech_Whisper_Node(Node):
    def __init__(self):
        super().__init__('speech_to_text_whisper_node')
        self.model = faster_whisper.WhisperModel(model_size_or_path="small", device='cpu', compute_type="float32")
        self.answer = ""
        self.history = [
            {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
            {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise. Short answer"},
        ]
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="docs")
        self.pub_result_voice = self.create_publisher(String, 'result', 10)
        self.pub_find_ball = self.create_publisher(String, 'find_ball', 10)
        self.sub_supress = self.create_subscription(Bool, 'supress', self.supress_callback, 10)
        self.user_text = ""
        self.talk_with_ai = True
        self.openai_client = OpenAI(base_url="http://192.168.2.5:1234/v1", api_key="lm-studio")
        self.locations_json = """
        [
            {"name": "kitchen", "x": 1.0, "y": 1.0, "theta": 0.0},
            {"name": "living room", "x": 1.0, "y": -1.0, "theta": 0.0},
            {"name": "bedroom", "x": -1.0, "y": -1.0, "theta": 0.0}
        ]
        """
        self.locations = json.loads(self.locations_json)
        self.mic_on = True
        self.p = pyaudio.PyAudio()
        self.stream = None

    def toggle_mic(self, state):
        if state and not self.mic_on:
            self.stream = self.p.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=16000,
                                      input=True,
                                      frames_per_buffer=1024)
            self.mic_on = True
            print("Microphone turned on")
        elif not state and self.mic_on:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.mic_on = False
            print("Microphone turned off")

    def record_audio(self, duration=10, threshold=500):
        if not self.mic_on:
            self.toggle_mic(True)

        CHUNK = 1024
        RATE = 16000

        print("Listening for speech...")

        frames = []
        is_speaking = False
        silence_frames = 0
        max_silence_frames = int(RATE / CHUNK * 2)  # 2 seconds of silence

        while True:
            if not self.mic_on:
                break
            data = self.stream.read(CHUNK)
            frames.append(data)

            # Convert audio chunks to numpy array
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Calculate volume
            volume = np.abs(audio_data).mean()

            if volume > threshold:
                is_speaking = True
                silence_frames = 0
            elif is_speaking:
                silence_frames += 1

            # Stop recording if silence is detected for 2 seconds after speech
            if is_speaking and silence_frames > max_silence_frames:
                break

            # Stop recording if total duration exceeds 15 seconds
            if len(frames) > int(RATE / CHUNK * 15):
                break

        print("Recording finished.")
        return frames, RATE

    def save_audio(self, frames, rate, filename="voice_record.wav"):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

    def transcribe_audio(self, filename):
        return "".join(seg.text for seg in self.model.transcribe(filename, language="en")[0])

    def main_loop(self):
        try:
            while True:
                self.toggle_mic(True)  # Ensure mic is on
                frames, rate = self.record_audio()
                self.toggle_mic(False)  # Turn off mic after recording
                
                if frames:  # Only process if we actually recorded something
                    self.save_audio(frames, rate)
                    
                    self.user_text = self.transcribe_audio("voice_record.wav")
                    print(f"Transcribed text: {self.user_text}")

                    if len(self.user_text) > 10:
                        for location in self.locations:
                            if location["name"] in self.user_text:
                                self.publish(self.pub_find_ball, "False")
                                self.answer = ""
                                self.publish(self.pub_result_voice, self.user_text)    
                                self.talk_with_ai = False
                                
                        if "home" in self.user_text and self.talk_with_ai:
                            self.publish(self.pub_find_ball, "True")
                        else:
                            self.publish(self.pub_find_ball, "False")
                            generator = self.openai_chat_response(self.user_text)
                            print(generator)
                            self.play_text_to_speech(generator)
                        self.talk_with_ai = True

                time.sleep(0.1)  # Short pause before next recording attempt

        except KeyboardInterrupt:
            print("\nStopping voice chat...")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.p.terminate()

    def generate_llama(self, message):
        self.answer = ""
        output = ollama.generate(
            model="tinyllama",
            prompt=f"Answer with short answer. Respond to this prompt: {message}"
        )
        return output['response']

    def chromadb_response(self, data):
        result = ''
        embedmodel = "nomic-embed-text"
        mainmodel = "tinyllama"
        chroma = chromadb.HttpClient(host="localhost", port=8000)
        collection = chroma.get_or_create_collection("buildragwithpython")

        queryembed = ollama.embeddings(model=embedmodel, prompt=data)['embedding']
        relevantdocs = collection.query(query_embeddings=[queryembed], n_results=1)["documents"][0]
        docs = "\n\n".join(relevantdocs)
        modelquery = f"{data} - Answer that question using the following text as a resource: {docs}"

        stream = ollama.generate(model=mainmodel, prompt=modelquery, stream=True)

        for chunk in stream:
            if chunk["response"]:
                data_text = "".join(chunk["response"])
                result += data_text
                
        return result

    def openai_chat_response(self, user_input):
        self.history.append({"role": "user", "content": user_input})
        
        completion = self.openai_client.chat.completions.create(
            model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
            messages=self.history,
            temperature=0.7,
            stream=True,
        )
        
        new_message = {"role": "assistant", "content": ""}
        for chunk in completion:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                new_message["content"] += content
                print(content, end="", flush=True)
        
        self.history.append(new_message)
        return new_message["content"]

    def play_text_to_speech(self, text, language='en', slow=False):
        self.toggle_mic(False)  # Turn off mic before speaking
        tts = gTTS(text=text, lang=language, slow=slow)
        
        temp_audio_file = "temp_audio.mp3"
        tts.save(temp_audio_file)
        
        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.stop()
        pygame.mixer.quit()

        time.sleep(0.5)  # Short pause after speaking
        os.remove(temp_audio_file)
        self.toggle_mic(True)  # Turn mic back on after speaking

    def publish(self, pub, text):
        msg = String()
        msg.data = text
        pub.publish(msg)
        self.get_logger().debug("Pub %s: %s" % (pub.topic_name, text))

    def supress_callback(self, msg):
        self.supress = msg.data
        self.get_logger().debug("Publish boolean : %s" % (msg))

def main(args=None):
    rclpy.init(args=args)
    n = Speech_Whisper_Node()
    n.main_loop()
    rclpy.shutdown()

if __name__ == '__main__':
    main()