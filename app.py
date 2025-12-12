import os
import json
import requests
from openai import OpenAI
from PyPDF2 import PdfReader
import gradio as gr
from dotenv import load_dotenv

load_dotenv(override=True)

def push(text):
    requests.post(
        url="https://api.pushover.net/1/messages.json",
        data={
            "user": os.getenc("PUSHOVER_USER"),
            "token": os.getenv("PUSHOVER_TOKEN"),
            "message": text
        }
    )

def record_unknown_user(email, name="Not Provided", notes="Not Provided"):
    push(f"User {name} with userid {email} wants to connect with context {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recorded unknown question: {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record and send Connection Requests of a person with his name, email and Notes",

    "parameters":{
        "type": "object",
        
        "properties": {
            "email": {
                "type": "string",
                "description": "Email Id of connection"
            },

            "name":{
                "type": "string",
                "description": "Name of Connection"
            },

            "notes":{
                "type": "string",
                "description": "Giving relevant context of conservation/history."
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }

}

record_unknown_question_json = {
    #Envelope Layer 1
    "name": "record_unknown_question",
    "description": "Always use this tool to record those questions which couldn't be answered since you didn't know the answer.",

    #Layer 2 
    "parameters":{
        "type": "object",
        #layer 3
        "properties":{
            #Layer 4
            "question":{
                "type": "string",
                "description": "Unknown question whose answer is not known"
            }            
        },
        "required": ["question"],
        "additionalProperties": False
    }

}


tools = [{"type": "function", "function": record_user_details_json},
         {"type": "function", "function": record_unknown_question_json}]


class Me:
    def __init__(self):
        #Stuff to be created as soon as object is created
        self.gemini = OpenAI(api_key= os.getenv("GOOGLE_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

        self.name = "Dhruv Tiwari"
        self.linkedin = ""

        reader = PdfReader("Profile.pdf")
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        with open("summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def handle_tool_calls(self, tool_calls):                # Better is individual if statement!!
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool used: {tool_name}", flush=True)

            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {"Tool Error: Tool doesn't exist"}  #Executing if!!
            results.append({"role": "tool", "content":json.dumps(result), "tool_call_id": tool_call.id})

        return results
    
    def my_system_prompt(self):
        system_prompt = f"""You are acting as {self.name}. You are answering questions related to {self.name}'s career, background, skills \n
and experience on {self.name}'s website. Your task is to represent yourself as {self.name} and try to be as faithful as \n
possible. You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer. \n
Respond as if you are responding to a future client who came across website.Do not reply with actions.\n
If you don't know an answer to a question, use your tool record_unknown_question for recording that querry.\n
If user is engaging in discussion, tell him to get in touch via email and record his email, name and some past conversations and record these details\n
through your tool record_user_details. 
"""
        system_prompt += f"\n\n ##Summary: \n{self.summary} \n\n ##LinkedIn Profile: \n {self.linkedin}"
        system_prompt += f"\n\n Based on this context, answer the querries as {self.name}"

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.my_system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
    
        while not done:
            response = self.gemini.chat.completions.create(model = "gemini-2.5-flash", messages=messages, tools=tools)
            if response.choices[0].finish_reason == "tool_calls":
                msg = response.choices[0].message
                tool_result = self.handle_tool_calls(msg.tool_calls)
                print(tool_result)
                messages.append(msg)
                messages.extend(tool_result)
            else:
                done = True

        return response.choices[0].message.content    


if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, title = "Dhruv's Profile Chatbot").launch()