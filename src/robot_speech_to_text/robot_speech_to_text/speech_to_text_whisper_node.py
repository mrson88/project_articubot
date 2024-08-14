#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import openai, elevenlabs, pyaudio, wave, numpy, collections, faster_whisper, torch.cuda
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
import rclpy, json, argparse, queue
from rclpy.node import Node
from std_msgs.msg import Bool, String  
import sounddevice as sd
from submodules.utilities import *
class Speech_Whisper_Node(Node):
    def __init__(self):
        super().__init__('speech_to_text_whisper_node')
        self.model, self.answer, self.history = faster_whisper.WhisperModel(model_size_or_path="small", device='cuda' if torch.cuda.is_available() else 'cpu'), "", []
        # self.model, self.answer, self.history = faster_whisper.WhisperModel(model_size_or_path="small", device='cpu'), "", []
        # self.model  = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small", device="cuda")
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="docs")
        self.pub_result_voice = self.create_publisher(String, 'result', 10)
        self.pub_find_ball = self.create_publisher(String, 'find_ball', 10)
        self.sub_supress = self.create_subscription(Bool, 'supress', self.supress_callback, 10)
        self.user_text = ""
        self.talk_with_ai = True
        self.history = [
            {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
            {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise. Short answer"},
        ]
        self.openai_client = OpenAI(base_url="http://192.168.2.5:1234/v1", api_key="lm-studio")
        self.locations_json = """
        [
    {"name": "kitchen", "x": 1.0, "y": 1.0, "theta": 0.0},
    {"name": "living room", "x": 1.0, "y": -1.0, "theta": 0.0},
    {"name": "bedroom", "x": -1.0, "y": -1.0, "theta": 0.0}
        ]
        """

        self.locations = json.loads(self.locations_json)
        try:
            while True:
                
                audio = pyaudio.PyAudio()
                # stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=1024)
                stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
                audio_buffer = collections.deque(maxlen=int((16000 // 512) * 0.5))
                frames, long_term_noise_level, current_noise_level, voice_activity_detected = [], 0.0, 0.0, False

                print("\n\nStart speaking. ", end="", flush=True)
                while True:
                    data = stream.read(512)
                    pegel, long_term_noise_level, current_noise_level = self.get_levels(data, long_term_noise_level, current_noise_level)
                    audio_buffer.append(data)

                    if voice_activity_detected:
                        frames.append(data)            
                        if current_noise_level < ambient_noise_level + 300:
                            break # voice actitivy ends 

                    if not voice_activity_detected and current_noise_level > long_term_noise_level + 600:
                        voice_activity_detected = True
                        print("I'm all ears.\n")
                        ambient_noise_level = long_term_noise_level
                        frames.extend(list(audio_buffer))
                print("Found voice chat, execute voice to text")
                stream.stop_stream(), stream.close(), audio.terminate()        
                # Transcribe recording using whisper
                with wave.open("voice_record.wav", 'wb') as wf:
                    wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
                    wf.writeframes(b''.join(frames))
                self.user_text ="".join(seg.text for seg in self.model.transcribe("voice_record.wav", language="en")[0])
                print(self.user_text)
                print("No location found in the received JSON command")

                if len(self.user_text)>10 :
                    for location in self.locations:
                        if location["name"] in self.user_text:
                            self.publish(self.pub_find_ball,"False")
                            self.answer=""
                            self.publish(self.pub_result_voice, self.user_text)    
                            self.talk_with_ai = False
                            
                            
                    if "home" in self.user_text and self.talk_with_ai:
                        self.publish(self.pub_find_ball,"True")

                    else :
                        # print(f'>>>{self.user_text}\n<<< ', end="", flush=True)
                        print(f"user_talk: {self.user_text}")
                        self.publish(self.pub_find_ball,"False")

                        # self.history.append({'role': 'user', 'content': self.user_text})

                        # Generate and stream output
                        generator = self.openai_chat_response(self.user_text)
                        # generator = self.generate_llama(self.user_text)
                        # generator = self.chromadb_response(self.user_text)
                        # generator=generate_rag_test(self.user_text)
                        print(generator)
                        self.play_text_to_speech(generator)
                    self.talk_with_ai = True

                            
                            

        except KeyboardInterrupt:
            print("\nStopping hear voice chat...")
            

        finally:
            print("\nStopped voice chat...")
            stream.stop_stream()
            stream.close()
            audio.terminate()

    def generate_llama(self,message):

        self.answer = ""

        # generate a response combining the prompt and data we retrieved in step 2
        output = ollama.generate(
        model="tinyllama",
        prompt=f"Answer with short answer. Respond to this prompt: {message}"
        )

        # print(output['response'])
        return output['response']

    def chromadb_response(self, data):
        result=''
        embedmodel = "nomic-embed-text"
        mainmodel = "tinyllama"
        chroma = chromadb.HttpClient(host="localhost", port=8000)
        collection = chroma.get_or_create_collection("buildragwithpython")

        # query = " ".join(sys.argv[1:])
        queryembed = ollama.embeddings(model=embedmodel, prompt=data)['embedding']


        relevantdocs = collection.query(query_embeddings=[queryembed], n_results=1)["documents"][0]
        docs = "\n\n".join(relevantdocs)
        modelquery = f"{data} - Answer that question using the following text as a resource: {docs}"

        stream = ollama.generate(model=mainmodel, prompt=modelquery, stream=True)

        for chunk in stream:
            if chunk["response"]:
                # print(chunk['response'], end='', flush=True)
                data_text="".join(chunk["response"])
                result+=data_text
                
        # print(result)
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

    def play_text_to_speech(self,text, language='en', slow=False):
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

    def get_levels(self,data, long_term_noise_level, current_noise_level):
        pegel = numpy.abs(numpy.frombuffer(data, dtype=numpy.int16)).mean()
        long_term_noise_level = long_term_noise_level * 0.995 + pegel * (1.0 - 0.995)
        current_noise_level = current_noise_level * 0.920 + pegel * (1.0 - 0.920)
        return pegel, long_term_noise_level, current_noise_level

    def publish(self, pub, text):
        """Publish a single text message"""
        msg = String()
        msg.data = text
        pub.publish(msg)
        self.get_logger().debug("Pub %s: %s" % (pub.topic_name, text))

    def supress_callback(self, msg):
        """Set flag if wave input has to be discared."""
        self.supress = msg.data
        self.get_logger().debug("Pubish boolean : %s" % (msg))
    
    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            self.get_logger().info(status)
        self.queue.put(bytes(indata))

def main(args=None):


    rclpy.init(args=args)
    n = Speech_Whisper_Node()
    rclpy.spin(n)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
