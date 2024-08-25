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
from ollama import Client
from openai import OpenAI
from groq import Groq
import speech_recognition as sr
import pygame
import time
import logging
import pydub
from io import BytesIO
from pydub import AudioSegment
from robot_speech_to_text.config import Config
from robot_speech_to_text.api_key_manager import get_transcription_api_key, get_response_api_key, get_tts_api_key
# from robot_speech_to_text.api_local import *
class Speech_Whisper_Node(Node):
    def __init__(self):
        super().__init__('speech_to_text_whisper_node')
        self.model = faster_whisper.WhisperModel(model_size_or_path="small", device='cpu', compute_type="float32")
        self.answer = ""
        self.history = [
            {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
            {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise. Shortest answer if you can"},
        ]
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="docs")
        self.pub_result_voice = self.create_publisher(String, 'result', 10)
        self.pub_find_ball = self.create_publisher(String, 'find_ball', 10)
        self.sub_supress = self.create_subscription(Bool, 'supress', self.supress_callback, 10)
        self.user_text = ""
        self.openai_client = OpenAI(base_url="http://192.168.2.5:11434", api_key="lm-studio")
        self.ollama_client = Client(host='http://192.168.2.5:11434')
        self.locations_json = """
        [
            {"name": "kitchen", "x": 1.0, "y": 1.0, "theta": 0.0},
            {"name": "living room", "x": 1.0, "y": -1.0, "theta": 0.0},
            {"name": "bedroom", "x": -1.0, "y": -1.0, "theta": 0.0}
        ]
        """
        self.locations = json.loads(self.locations_json)
        self.silence_threshold = 700  # Adjust this value based on your environment
        self.silence_duration = 1.0  # Duration of silence to end recording (in seconds)
        self.max_duration = 10  # Maximum recording duration in seconds
        self.unwanted_phrases = ["Thanks for watching!", "Thanks for watching.", "Thank you for watching!", "Thank you for watching."]


    def get_flight_times(self,departure: str, arrival: str) -> str:
        self.flights = {
            'NYC-LAX': {'departure': '08:00 AM', 'arrival': '11:30 AM', 'duration': '5h 30m'},
            'LAX-NYC': {'departure': '02:00 PM', 'arrival': '10:30 PM', 'duration': '5h 30m'},
            'LHR-JFK': {'departure': '10:00 AM', 'arrival': '01:00 PM', 'duration': '8h 00m'},
            'JFK-LHR': {'departure': '09:00 PM', 'arrival': '09:00 AM', 'duration': '7h 00m'},
            'CDG-DXB': {'departure': '11:00 AM', 'arrival': '08:00 PM', 'duration': '6h 00m'},
            'DXB-CDG': {'departure': '03:00 AM', 'arrival': '07:30 AM', 'duration': '7h 30m'},
    }

        key = f'{departure}-{arrival}'.upper()
        return json.dumps(self.flights.get(key, {'error': 'Flight not found'}))
    def get_antonyms(self,word: str) -> str:
        "Get the antonyms of the any given word"

        words = {
            "hot": "cold",
            "small": "big",
            "weak": "strong",
            "light": "dark",
            "lighten": "darken",
            "dark": "bright",
        }

        return json.dumps(words.get(word, "Not available in database"))
    def record_audio(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("Waiting for speech...")

        frames = []
        is_speaking = False
        silence_start = None
        recording_start = None

        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_data).mean()

                if not is_speaking:
                    if volume > self.silence_threshold:
                        print("Speech detected, starting to record...")
                        is_speaking = True
                        recording_start = time.time()
                        frames.append(data)
                else:
                    frames.append(data)
                    current_time = time.time()
                    
                    if volume <= self.silence_threshold:
                        if silence_start is None:
                            silence_start = current_time
                        elif current_time - silence_start >= self.silence_duration:
                            print("End of speech detected.")
                            break
                    else:
                        silence_start = None

                    if current_time - recording_start >= self.max_duration:
                        print(f"Maximum duration of {self.max_duration} seconds reached.")
                        break

        except Exception as e:
            print(f"Error during audio recording: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        if not frames:
            print("No speech detected.")
        else:
            print("Recording finished.")

        return frames, RATE

    def save_audio(self, frames, rate, filename="voice_record.wav"):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
    def save_audio_mp3(self, frames, rate, filename="voice_record.mp3"):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

    def record_audio_api(self,file_path, timeout=10, phrase_time_limit=None, retries=3, energy_threshold=2000, pause_threshold=1, phrase_threshold=0.1, dynamic_energy_threshold=True, calibration_duration=1):
        """
        Record audio from the microphone and save it as an MP3 file.
        
        Args:
        file_path (str): The path to save the recorded audio file.
        timeout (int): Maximum time to wait for a phrase to start (in seconds).
        phrase_time_limit (int): Maximum time for the phrase to be recorded (in seconds).
        retries (int): Number of retries if recording fails.
        energy_threshold (int): Energy threshold for considering whether a given chunk of audio is speech or not.
        pause_threshold (float): How much silence the recognizer interprets as the end of a phrase (in seconds).
        phrase_threshold (float): Minimum length of a phrase to consider for recording (in seconds).
        dynamic_energy_threshold (bool): Whether to enable dynamic energy threshold adjustment.
        calibration_duration (float): Duration of the ambient noise calibration (in seconds).
        """
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = energy_threshold
        recognizer.pause_threshold = pause_threshold
        recognizer.phrase_threshold = phrase_threshold
        recognizer.dynamic_energy_threshold = dynamic_energy_threshold
        for attempt in range(retries):
            try:
                with sr.Microphone() as source:
                    logging.info("Calibrating for ambient noise...")
                    recognizer.adjust_for_ambient_noise(source, duration=calibration_duration)
                    logging.info("Recording started")
                    # Listen for the first phrase and extract it into audio data
                    audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                    logging.info("Recording complete")
                    # Convert the recorded audio data to an MP3 file
                    wav_data = audio_data.get_wav_data()
                    audio_segment = pydub.AudioSegment.from_wav(BytesIO(wav_data))
                    mp3_data = audio_segment.export(file_path, format="mp3", bitrate="128k", parameters=["-ar", "22050", "-ac", "1"])
                    return
            except sr.WaitTimeoutError:
                logging.warning(f"Listening timed out, retrying... ({attempt + 1}/{retries})")
            except Exception as e:
                logging.error(f"Failed to record audio: {e}")
                break
        else:
            logging.error("Recording failed after all retries")
    def transcribe_audio(self, filename):
        transcription = "".join(seg.text for seg in self.model.transcribe(filename, language="en")[0])
        
        return self.clean_transcription(transcription)

    def transcribe_audio_api(self,model, api_key, audio_file_path, local_model_path=None):
        try:
            if model == 'openai':
                client = OpenAI(api_key=api_key)
                with open(audio_file_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language='en'
                    )
                return transcription.text
            elif model == 'groq':
                client = Groq(api_key=api_key)
                with open(audio_file_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=audio_file,
                        language='vi'
                    )
                return transcription.text

            
            else:
                raise ValueError("Unsupported transcription model")

        except Exception as e:
            # logging.error(Fore.RED + f"Failed to transcribe audio: {e}" + Fore.RESET)
            raise Exception("Error in transcribing audio")
        return
    def clean_transcription(self, text):
        # Remove unwanted phrases
        for phrase in self.unwanted_phrases:
            text = text.replace(phrase, "")
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        return text.strip()

    def main_loop(self):
        try:
            while True:
                frames, rate = self.record_audio()
                
                if frames:  # Only process if we actually recorded something
                    # self.save_audio(frames, rate)
                    self.save_audio_mp3(frames, rate)
                    transcription_api_key = get_transcription_api_key()
                    # self.record_audio_api("voice_record.mp3")
                    
                    # self.user_text = self.transcribe_audio("voice_record.wav")
                    self.user_text = self.transcribe_audio_api("groq", transcription_api_key, "voice_record.mp3")
                    print(f"Transcribed text: {self.user_text}")

                    if len(self.user_text) > 5:  # Reduced minimum length check
                        location_mentioned = False
                        for location in self.locations:
                            if location["name"] in self.user_text:
                                location_mentioned = True
                                self.publish(self.pub_find_ball, "False")
                                self.publish(self.pub_result_voice, self.user_text)
                                break
                        
                        if not location_mentioned:
                            if "home" in self.user_text:
                                self.publish(self.pub_find_ball, "True")

                            else:
                                self.publish(self.pub_find_ball, "False")
                                generator = self.ollama_chat_response(self.user_text)
                                print(generator)
                                self.play_text_to_speech(generator)
                        if "stop" in self.user_text:
                            self.publish(self.pub_find_ball, "False")
                time.sleep(0.1)  # Short pause before next recording attempt

        except KeyboardInterrupt:
            print("\nStopping voice chat...")

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
        try:
            self.history.append({"role": "user", "content": user_input})
            
            completion = self.openai_client.chat.completions.create(
                model="llama3.1",
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
        except:
            print("Error connect to LM server")
            return "Sorry I can't answer"

    def ollama_chat_response(self, user_input):
        try:
            self.history.append({"role": "user", "content": user_input})
            
            completion = self.ollama_client.chat(
                model="llama3.1",
                messages=self.history,
            )
            new_message = {"role": "assistant", "content": ""}
            new_message["content"] += completion["message"]["content"]
            self.history.append(new_message)
            print(completion["message"]["content"])
            return completion["message"]["content"]
        except:
            print("Error connect to Ollama server")
            return "Sorry I can't answer"       


    def ollama_chat_response_toocall(self, user_input):
        try:
            self.history.append({"role": "user", "content": user_input})
            
            completion = self.ollama_client.chat(
                model="llama3.1",
                messages=self.history,
                tools=[
                    {
                        'type': 'function',
                        'function': {
                        'name': 'get_flight_times',
                        'description': 'Get the flight times between two cities',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                            'departure': {
                                'type': 'string',
                                'description': 'The departure city (airport code)',
                            },
                            'arrival': {
                                'type': 'string',
                                'description': 'The arrival city (airport code)',
                            },
                            },
                            'required': ['departure', 'arrival'],
                        },
                        },
                    },
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_antonyms",
                                    "description": "Get the antonyms of any given words",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "word": {
                                                "type": "string",
                                                "description": "The word for which the opposite is required.",
                                            },
                                        },
                                        "required": ["word"],
                                    },
                                },
                            },


                        ],
            )

            # Process function calls made by the model
            if completion['message'].get('tool_calls'):
                available_functions = {
                    'get_flight_times': self.get_flight_times,
                    "get_antonyms": self.get_antonyms,
                    # "get_stock_price": get_stock_price,
                }
                for tool in completion['message']['tool_calls']:
                    function_to_call = available_functions[tool['function']['name']]
                    if function_to_call == self.get_flight_times:
                        function_response = function_to_call(
                            tool["function"]["arguments"]["departure"],
                            tool["function"]["arguments"]["arrival"],
                        )
                        print(f"function response: {function_response}")
                        return f"function response: {function_response}"

                    elif function_to_call == self.get_antonyms:
                        function_response = function_to_call(
                            tool["function"]["arguments"]["word"],
                        )
                        print(f"function response: {function_response}")   
                        return f"function response: {function_response}"         
            else:
                new_message = {"role": "assistant", "content": ""}
                new_message["content"] += completion["message"]["content"]
                self.history.append(new_message)
                print(completion["message"]["content"])
                return completion["message"]["content"]
        except:
            print("Error connect to Ollama server")
            return "Sorry I can't answer"       

    def play_text_to_speech(self, text, language='vi', slow=False):
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

        time.sleep(3)
        os.remove(temp_audio_file)

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