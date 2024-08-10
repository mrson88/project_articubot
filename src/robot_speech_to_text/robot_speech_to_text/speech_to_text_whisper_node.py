# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import openai, elevenlabs, pyaudio, wave, numpy, collections, faster_whisper, torch.cuda
# import os
# import time
# import pygame
# from gtts import gTTS
# import ollama
# import chromadb
# from openai import OpenAI
# import locale
# import json
# locale.getpreferredencoding = lambda: "UTF-8"
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
# import rclpy, json, argparse, queue
# from rclpy.node import Node
# from std_msgs.msg import Bool, String
# # from vosk import Model, KaldiRecognizer   
# import sounddevice as sd
# from submodules.utilities import *
# # from transformers import pipeline
# class Speech_Whisper_Node(Node):
#     def __init__(self):
#         super().__init__('speech_to_text_whisper_node')
#         # self.model, self.answer, self.history = faster_whisper.WhisperModel(model_size_or_path="medium", device='cuda' if torch.cuda.is_available() else 'cpu'), "", []
#         self.model, self.answer, self.history = faster_whisper.WhisperModel(model_size_or_path="small", device='cpu'), "", []
#         # self.model  = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small", device="cuda")
#         self.client = chromadb.Client()
#         self.collection = self.client.create_collection(name="docs")
#         self.pub_result_voice = self.create_publisher(String, 'result', 10)
#         self.pub_find_ball = self.create_publisher(String, 'find_ball', 10)
#         self.sub_supress = self.create_subscription(Bool, 'supress', self.supress_callback, 10)
#         self.user_text = ""
#         self.talk_with_ai = True
#         self.history = [
#             {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
#             {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
#         ]
#         self.openai_client = OpenAI(base_url="http://192.168.2.5:1234/v1", api_key="lm-studio")
#         self.locations_json = """
#         [
#     {"name": "kitchen", "x": 1.0, "y": 1.0, "theta": 0.0},
#     {"name": "living room", "x": 1.0, "y": -1.0, "theta": 0.0},
#     {"name": "bedroom", "x": -1.0, "y": -1.0, "theta": 0.0}
#         ]
#         """

#         self.locations = json.loads(self.locations_json)
#         try:
#             while True:
                
#                 audio = pyaudio.PyAudio()
#                 # stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=1024)
#                 stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
#                 audio_buffer = collections.deque(maxlen=int((16000 // 512) * 0.5))
#                 frames, long_term_noise_level, current_noise_level, voice_activity_detected = [], 0.0, 0.0, False

#                 print("\n\nStart speaking. ", end="", flush=True)
#                 while True:
#                     data = stream.read(512)
#                     pegel, long_term_noise_level, current_noise_level = self.get_levels(data, long_term_noise_level, current_noise_level)
#                     audio_buffer.append(data)

#                     if voice_activity_detected:
#                         frames.append(data)            
#                         if current_noise_level < ambient_noise_level + 500:
#                             break # voice actitivy ends 

#                     if not voice_activity_detected and current_noise_level > long_term_noise_level + 1000:
#                         voice_activity_detected = True
#                         print("I'm all ears.\n")
#                         ambient_noise_level = long_term_noise_level
#                         frames.extend(list(audio_buffer))

#                 stream.stop_stream(), stream.close(), audio.terminate()        
#                 # Transcribe recording using whisper
#                 with wave.open("voice_record.wav", 'wb') as wf:
#                     wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
#                     wf.writeframes(b''.join(frames))
#                 self.user_text ="".join(seg.text for seg in self.model.transcribe("voice_record.wav", language="en")[0])
#                 # result = self.model("voice_record.wav")
#                 # self.user_text = result['text'].strip()
#                 print(self.user_text)
#                 print("ERROR:No location found in the received JSON command")

#                 if len(self.user_text)>10 :
#                     for location in self.locations:
#                         if location["name"] in self.user_text:
#                             self.publish(self.pub_find_ball,"False")
#                             self.answer=""
#                             self.publish(self.pub_result_voice, self.user_text)    
#                             self.talk_with_ai = False
                            
                            
#                     if "home" in self.user_text and self.talk_with_ai:
#                         self.publish(self.pub_find_ball,"True")

#                     else :
#                         # print(f'>>>{self.user_text}\n<<< ', end="", flush=True)
#                         print(f"user_talk: {self.user_text}")

#                         # self.history.append({'role': 'user', 'content': self.user_text})

#                         # Generate and stream output
#                         generator = self.openai_chat_response(self.user_text)
#                         # generator = self.generate_llama(self.user_text)
#                         # generator = self.chromadb_response(self.user_text)
#                         # generator=generate_rag_test(self.user_text)
#                         print(generator)

#                         # elevenlabs.stream(elevenlabs.generate(text=generator, voice="Nicole", model="eleven_monolingual_v1", stream=True))
#                         self.play_text_to_speech(generator)

#                         # self.history.append({'role': 'assistant', 'content': self.answer})
#                         # print([system_prompt] + history[-10:])
#                     self.talk_with_ai = True

                            
                            

#         except KeyboardInterrupt:
#             print("\nStopping...")
            

#         finally:
#             stream.stop_stream()
#             stream.close()
#             audio.terminate()

#     def generate_llama(self,message):

#         self.answer = ""

#         # generate a response combining the prompt and data we retrieved in step 2
#         output = ollama.generate(
#         model="tinyllama",
#         prompt=f"Answer with short answer. Respond to this prompt: {message}"
#         )

#         # print(output['response'])
#         return output['response']

#     def chromadb_response(self, data):
#         result=''

#         # embedmodel = getconfig()["embedmodel"]
#         # mainmodel = getconfig()["mainmodel"]
#         embedmodel = "nomic-embed-text"
#         mainmodel = "tinyllama"
#         chroma = chromadb.HttpClient(host="localhost", port=8000)
#         collection = chroma.get_or_create_collection("buildragwithpython")

#         # query = " ".join(sys.argv[1:])
#         queryembed = ollama.embeddings(model=embedmodel, prompt=data)['embedding']


#         relevantdocs = collection.query(query_embeddings=[queryembed], n_results=1)["documents"][0]
#         docs = "\n\n".join(relevantdocs)
#         modelquery = f"{data} - Answer that question using the following text as a resource: {docs}"

#         stream = ollama.generate(model=mainmodel, prompt=modelquery, stream=True)

#         for chunk in stream:
#             if chunk["response"]:
#                 # print(chunk['response'], end='', flush=True)
#                 data_text="".join(chunk["response"])
#                 result+=data_text
                
#         # print(result)
#         return result

#     def openai_chat_response(self, user_input):
#         self.history.append({"role": "user", "content": user_input})
        
#         completion = self.openai_client.chat.completions.create(
#             model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
#             messages=self.history,
#             temperature=0.7,
#             stream=True,
#         )
        
#         new_message = {"role": "assistant", "content": ""}
#         for chunk in completion:
#             if chunk.choices[0].delta.content:
#                 content = chunk.choices[0].delta.content
#                 new_message["content"] += content
#                 print(content, end="", flush=True)
        
#         self.history.append(new_message)
#         return new_message["content"]

#     def play_text_to_speech(self,text, language='en', slow=False):
#         tts = gTTS(text=text, lang=language, slow=slow)
        
#         temp_audio_file = "temp_audio.mp3"
#         tts.save(temp_audio_file)
        
#         pygame.mixer.init()
#         pygame.mixer.music.load(temp_audio_file)
#         pygame.mixer.music.play()

#         while pygame.mixer.music.get_busy():
#             pygame.time.Clock().tick(10)

#         pygame.mixer.music.stop()
#         pygame.mixer.quit()

#         time.sleep(3)
#         os.remove(temp_audio_file)

#     def get_levels(self,data, long_term_noise_level, current_noise_level):
#         pegel = numpy.abs(numpy.frombuffer(data, dtype=numpy.int16)).mean()
#         long_term_noise_level = long_term_noise_level * 0.995 + pegel * (1.0 - 0.995)
#         current_noise_level = current_noise_level * 0.920 + pegel * (1.0 - 0.920)
#         return pegel, long_term_noise_level, current_noise_level

#     def publish(self, pub, text):
#         """Publish a single text message"""
#         msg = String()
#         msg.data = text
#         pub.publish(msg)
#         self.get_logger().debug("Pub %s: %s" % (pub.topic_name, text))

#     def supress_callback(self, msg):
#         """Set flag if wave input has to be discared."""
#         self.supress = msg.data
#         self.get_logger().debug("Pubish boolean : %s" % (msg))
    
#     def callback(self, indata, frames, time, status):
#         """This is called (from a separate thread) for each audio block."""
#         if status:
#             self.get_logger().info(status)
#         self.queue.put(bytes(indata))

# def main(args=None):

#     # parser = argparse.ArgumentParser(
#     #     description='STT ROS Node. A speach to text recognizer using Vosk speech recognition toolkit.',
#     #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     # parser.add_argument(
#     #     '-l', '--list', action="store_true", help='list available devices')
#     # parsed, remaining = parser.parse_known_args()

#     # if parsed.list:
#     #     print(sd.query_devices())
#     #     parser.exit(0)

#     rclpy.init(args=args)
#     n = Speech_Whisper_Node()
#     rclpy.spin(n)
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
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
import json
import locale

locale.getpreferredencoding = lambda: "UTF-8"

class Speech_Whisper_Node(Node):
    def __init__(self):
        super().__init__('speech_to_text_whisper_node')
        self.setup_whisper_model()
        self.setup_publishers_and_subscribers()
        self.setup_openai_client()
        self.setup_chromadb()
        self.setup_chat_history()
        self.load_locations()
        self.wake_word = "robot"

    def setup_whisper_model(self):
        self.model = faster_whisper.WhisperModel(model_size_or_path="small", device='cpu')

    def setup_publishers_and_subscribers(self):
        self.pub_result_voice = self.create_publisher(String, 'result', 10)
        self.pub_find_ball = self.create_publisher(String, 'find_ball', 10)
        self.sub_supress = self.create_subscription(Bool, 'supress', self.supress_callback, 10)

    def setup_openai_client(self):
        self.openai_client = OpenAI(base_url="http://192.168.2.5:1234/v1", api_key="lm-studio")

    def setup_chromadb(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="docs")

    def setup_chat_history(self):
        self.history = [
            {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
            {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
        ]

    def load_locations(self):
        locations_json = """
        [
            {"name": "kitchen", "x": 1.0, "y": 1.0, "theta": 0.0},
            {"name": "living room", "x": 1.0, "y": -1.0, "theta": 0.0},
            {"name": "bedroom", "x": -1.0, "y": -1.0, "theta": 0.0}
        ]
        """
        self.locations = json.loads(locations_json)

    def run(self):
        try:
            while True:
                if self.listen_for_wake_word():
                    self.get_logger().info("Wake word detected. Listening for command...")
                    self.process_audio_input()
                    if len(self.user_text) > 10:
                        self.process_user_input()
                else:
                    time.sleep(0.1)  # Short sleep to prevent CPU overuse
        except KeyboardInterrupt:
            self.get_logger().info("Stopping...")

    def listen_for_wake_word(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        
        frames = []
        for _ in range(int(16000 / 1024 * 2)):  # Listen for 2 seconds
            data = stream.read(1024)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open("wake_word.wav", 'wb') as wf:
            wf.setparams((1, pyaudio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
            wf.writeframes(b''.join(frames))
        
        wake_word_text = "".join(seg.text for seg in self.model.transcribe("wake_word.wav", language="en")[0])
        return self.wake_word.lower() in wake_word_text.lower()

    def process_audio_input(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        audio_buffer = collections.deque(maxlen=int((16000 // 512) * 0.5))
        frames, long_term_noise_level, current_noise_level, voice_activity_detected = [], 0.0, 0.0, False
        ambient_noise_level = 0.0

        self.get_logger().info("Listening for command...")
        while True:
            data = stream.read(512)
            pegel, long_term_noise_level, current_noise_level = self.get_levels(data, long_term_noise_level, current_noise_level)
            audio_buffer.append(data)

            if voice_activity_detected:
                frames.append(data)            
                if current_noise_level < ambient_noise_level + 500:
                    break  # voice activity ends 

            if not voice_activity_detected and current_noise_level > long_term_noise_level + 1000:
                voice_activity_detected = True
                ambient_noise_level = long_term_noise_level
                frames.extend(list(audio_buffer))

        stream.stop_stream()
        stream.close()
        audio.terminate()

        self.transcribe_audio(frames, audio)

    def transcribe_audio(self, frames, audio):
        with wave.open("voice_record.wav", 'wb') as wf:
            wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
            wf.writeframes(b''.join(frames))
        
        self.user_text = "".join(seg.text for seg in self.model.transcribe("voice_record.wav", language="en")[0])
        self.get_logger().info(f"Transcribed: {self.user_text}")

    def process_user_input(self):
        if self.check_location_command():
            return

        if "home" in self.user_text.lower():
            self.publish(self.pub_find_ball, "True")
        else:
            self.get_logger().info(f"Processing: {self.user_text}")
            response = self.openai_chat_response(self.user_text)
            self.get_logger().info(f"Assistant: {response}")
            self.play_text_to_speech(response)

    def check_location_command(self):
        for location in self.locations:
            if location["name"] in self.user_text.lower():
                self.publish(self.pub_find_ball, "False")
                self.publish(self.pub_result_voice, self.user_text)
                return True
        return False

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

    @staticmethod
    def get_levels(data, long_term_noise_level, current_noise_level):
        pegel = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
        long_term_noise_level = long_term_noise_level * 0.995 + pegel * 0.005
        current_noise_level = current_noise_level * 0.920 + pegel * 0.080
        return pegel, long_term_noise_level, current_noise_level

    def publish(self, pub, text):
        msg = String()
        msg.data = text
        pub.publish(msg)
        self.get_logger().debug(f"Published to {pub.topic_name}: {text}")

    def supress_callback(self, msg):
        self.supress = msg.data
        self.get_logger().debug(f"Suppress: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = Speech_Whisper_Node()
    node.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()