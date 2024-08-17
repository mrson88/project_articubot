#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
import openai
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
import json
from threading import Thread

class NewChatVoiceNode(Node):
    def __init__(self):
        super().__init__('new_chat_voice_node')
        
        self.init_models_and_clients()
        self.setup_ros_communication()
        self.init_audio_parameters()
        self.init_conversation()
        
        Thread(target=self.main_processing_loop).start()

    def init_models_and_clients(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device="cpu"
        self.whisper_model = faster_whisper.WhisperModel("small", device=device)
        self.chroma_client = chromadb.Client()
        self.chroma_collection = self.chroma_client.create_collection(name="docs")
        self.openai_client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    def setup_ros_communication(self):
        self.pub_result_voice = self.create_publisher(String, 'result', 10)
        self.pub_find_ball = self.create_publisher(String, 'find_ball', 10)
        self.sub_suppress = self.create_subscription(Bool, 'suppress', self.suppress_callback, 10)

    def init_audio_parameters(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                                      input=True, frames_per_buffer=1024)
        self.audio_buffer = collections.deque(maxlen=int((16000 // 512) * 0.5))

    def init_conversation(self):
        self.history = [
            {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
            {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
        ]
        self.user_text = ""
        self.talk_with_ai = True
        self.suppress = False
        
        # Load locations from JSON file
        try:
            with open('locations.json', 'r') as f:
                self.locations = json.load(f)
        except FileNotFoundError:
            self.get_logger().warn("locations.json not found. Using default empty list.")
            self.locations = []

    def main_processing_loop(self):
        while rclpy.ok():
            try:
                self.process_audio_input()
                if len(self.user_text) > 10:
                    self.process_user_input()
            except KeyboardInterrupt:
                break
        self.cleanup()

    def process_audio_input(self):
        frames = []
        long_term_noise_level = current_noise_level = 0.0
        voice_activity_detected = False
        ambient_noise_level = 0.0
        
        self.get_logger().info("Start speaking...")
        
        while True:
            data = self.stream.read(512)
            pegel, long_term_noise_level, current_noise_level = self.get_levels(data, long_term_noise_level, current_noise_level)
            self.audio_buffer.append(data)

            if voice_activity_detected:
                frames.append(data)            
                if current_noise_level < ambient_noise_level + 300:
                    break
            elif current_noise_level > long_term_noise_level + 600:
                voice_activity_detected = True
                self.get_logger().info("Voice detected. Listening...")
                ambient_noise_level = long_term_noise_level
                frames.extend(list(self.audio_buffer))
        
        self.transcribe_audio(frames)

    def transcribe_audio(self, frames):
        with wave.open("voice_record.wav", 'wb') as wf:
            wf.setparams((1, self.audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
            wf.writeframes(b''.join(frames))
        self.user_text = "".join(seg.text for seg in self.whisper_model.transcribe("voice_record.wav", language="en")[0])
        self.get_logger().info(f"Transcribed: {self.user_text}")

    def process_user_input(self):
        if any(location["name"].lower() in self.user_text.lower() for location in self.locations):
            self.publish(self.pub_find_ball, "False")
            self.publish(self.pub_result_voice, self.user_text)
            self.talk_with_ai = False
        elif "home" in self.user_text.lower() and self.talk_with_ai:
            self.publish(self.pub_find_ball, "True")
        else:
            self.publish(self.pub_find_ball, "False")
            response = self.generate_ai_response()
            self.play_text_to_speech(response)
        self.talk_with_ai = True

    def generate_ai_response(self):
        # Choose which AI model to use here
        return self.openai_chat_response(self.user_text)
        # Alternatively, you could use:
        # return self.generate_llama(self.user_text)
        # or
        # return self.chromadb_response(self.user_text)

    def openai_chat_response(self, user_input):
        self.history.append({"role": "user", "content": user_input})
        try:
            completion = self.openai_client.chat.completions.create(
                model="local-model",  # Use the appropriate model name for your local setup
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
        except Exception as e:
            self.get_logger().error(f"Error in OpenAI chat response: {e}")
            return "I'm sorry, I encountered an error while processing your request."

    def generate_llama(self, message):
        try:
            response = ollama.generate(
                model="tinyllama",
                prompt=f"Answer with short answer. Respond to this prompt: {message}"
            )
            return response['response']
        except Exception as e:
            self.get_logger().error(f"Error in Llama generation: {e}")
            return "I'm sorry, I encountered an error while processing your request."

    def chromadb_response(self, data):
        try:
            embedmodel = "nomic-embed-text"
            mainmodel = "tinyllama"
            
            queryembed = ollama.embeddings(model=embedmodel, prompt=data)['embedding']
            relevantdocs = self.chroma_collection.query(query_embeddings=[queryembed], n_results=1)["documents"][0]
            docs = "\n\n".join(relevantdocs)
            modelquery = f"{data} - Answer that question using the following text as a resource: {docs}"

            stream = ollama.generate(model=mainmodel, prompt=modelquery, stream=True)
            result = ""
            for chunk in stream:
                if chunk["response"]:
                    result += chunk["response"]
            return result
        except Exception as e:
            self.get_logger().error(f"Error in ChromaDB response: {e}")
            return "I'm sorry, I encountered an error while processing your request."

    def play_text_to_speech(self, text, language='en', slow=False):
        try:
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
            time.sleep(1)  # Reduced wait time
            os.remove(temp_audio_file)
        except Exception as e:
            self.get_logger().error(f"Error in text-to-speech: {e}")

    def get_levels(self, data, long_term_noise_level, current_noise_level):
        pegel = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
        long_term_noise_level = long_term_noise_level * 0.995 + pegel * 0.005
        current_noise_level = current_noise_level * 0.920 + pegel * 0.080
        return pegel, long_term_noise_level, current_noise_level

    def publish(self, pub, text):
        msg = String()
        msg.data = text
        pub.publish(msg)
        self.get_logger().debug(f"Published to {pub.topic_name}: {text}")

    def suppress_callback(self, msg):
        self.suppress = msg.data
        self.get_logger().debug(f"Suppress set to: {msg.data}")

    def cleanup(self):
        self.get_logger().info("Cleaning up and shutting down...")
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

def main(args=None):
    rclpy.init(args=args)
    node = NewChatVoiceNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()