#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import os
import openai
from openai import OpenAI
import json

class Improved_Speech_Whisper_Node(Node):
    def __init__(self):
        super().__init__('improved_speech_to_text_whisper_node')
        
        self.pub_result_voice = self.create_publisher(String, 'result', 10)
        self.pub_find_ball = self.create_publisher(String, 'find_ball', 10)
        self.sub_supress = self.create_subscription(Bool, 'supress', self.supress_callback, 10)
        
        self.recognizer = sr.Recognizer()
        self.openai_client = OpenAI(base_url="http://192.168.2.5:1234/v1", api_key="lm-studio")
        
        self.messages_array = [
            {'role': 'system', 'content': 'You are an intelligent assistant named Cortana. You always provide well-reasoned answers that are both correct and helpful.'},
        ]
        
        self.locations_json = """
        [
            {"name": "kitchen", "x": 1.0, "y": 1.0, "theta": 0.0},
            {"name": "living room", "x": 1.0, "y": -1.0, "theta": 0.0},
            {"name": "bedroom", "x": -1.0, "y": -1.0, "theta": 0.0}
        ]
        """
        self.locations = json.loads(self.locations_json)
        self.supress = False

    def run(self):
        while rclpy.ok():
            self.listen()

    def listen(self):
        with sr.Microphone() as source:
            print("Listening.....")
            self.recognizer.pause_threshold = 1
            audio = self.recognizer.listen(source)
        
        try:
            print('Recognizing...')
            query = self.recognizer.recognize_google(audio, language='en-in')
            print(f'User has said: {query}')
            self.messages_array.append({'role': 'user', 'content': query})
            self.process_query(query)
        except Exception as e:
            print('Say that again please...', e)

    def process_query(self, query):
        if len(query) > 10:
            for location in self.locations:
                if location["name"] in query.lower():
                    self.publish(self.pub_find_ball, "False")
                    self.publish(self.pub_result_voice, query)
                    return

            if "home" in query.lower():
                self.publish(self.pub_find_ball, "True")
            else:
                self.publish(self.pub_find_ball, "False")
                response = self.generate_response()
                self.speak(response)

    def generate_response(self):
        print('Generating response...')
        completion = self.openai_client.chat.completions.create(
            model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
            messages=self.messages_array,
            temperature=0.7,
        )
        response = completion.choices[0].message.content
        self.messages_array.append({'role': 'assistant', 'content': response})
        return response

    def speak(self, text):
        print('Speaking:', text)
        speech = gTTS(text=text, lang='en', slow=False)
        speech.save('response.mp3')
        playsound('response.mp3')
        os.remove('response.mp3')

    def publish(self, pub, text):
        msg = String()
        msg.data = text
        pub.publish(msg)
        self.get_logger().debug("Pub %s: %s" % (pub.topic_name, text))

    def supress_callback(self, msg):
        self.supress = msg.data
        self.get_logger().debug("Suppress boolean : %s" % (msg))

def main(args=None):
    rclpy.init(args=args)
    node = Improved_Speech_Whisper_Node()
    node.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()