from dotenv import load_dotenv
load_dotenv()  # loading all the environment variables

import streamlit as st
import os
import uuid
import pyaudio
import google.generativeai as genai
from streamlit_chat import message
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from gtts import gTTS
from langdetect import detect
from pymongo import MongoClient

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio, language="en-US")  # Set language to English (US)
        return text
    except sr.UnknownValueError:
        st.warning("Speech recognition could not understand audio")
    except sr.RequestError as e:
        st.warning(f"Could not request results from Google Speech Recognition service; {e}")

def text_to_speech(text, language):
    tts = gTTS(text=text, lang=language)
    tts.save("response.mp3")
    st.audio("response.mp3")

def main():
    try:
        mongodb_url = os.getenv("MONGODB_URL")
        client = MongoClient(mongodb_url)
        db = client["chat_app"]
        users_collection = db["users"]
        
        st.set_page_config(page_title="Teaching Assistant", layout="wide")

        # Initialize session state for chat history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        # Configure Google Generative AI
        google_api_key = os.getenv("GOOGLE_API_KEY")
        model = genai.GenerativeModel("gemini-pro")
        chat = model.start_chat(history=[])

        def get_gemini_response(question):
            response = chat.send_message(question, stream=True)
            return response

        # Sidebar navigation  
        menu = ["Chat", "About"]
        choice = st.sidebar.selectbox("Select an option", menu)

        # Chat section
        if choice == "Chat":
            st.header("Teaching Assistant")

            # User authentication 
            if 'user_id' not in st.session_state:
                st.session_state['user_id'] = str(uuid.uuid4())
            
            user_id = st.session_state['user_id']
            
            st.title("Welcome to the Teaching ChatBot")
            
            # Chat interface
            input_type = st.radio("Select input type", ("Text", "Voice"))
            
            if input_type == "Text":
                input_text = st.text_area("Enter your question", key="input")
            else:
                input_audio = audio_recorder()
                if input_audio:
                    input_text = speech_to_text()
                    st.write(f"Transcribed text: {input_text}")
                else:
                    input_text = ""

            submit = st.button("Ask")

            if submit and input_text:
                with st.spinner("Generating response..."):
                    language = detect(input_text)
                    response = get_gemini_response(input_text)
                    
                    # Display user query
                    st.session_state['chat_history'].append({"role": "user", "content": input_text})
                    message(input_text, is_user=True, key=f"user_msg_{len(st.session_state['chat_history'])}")
                    
                    # Display bot response  
                    response_text = ""
                    for chunk in response:
                        response_text += chunk.text
                    st.session_state['chat_history'].append({"role": "assistant", "content": response_text})
                    message(response_text, is_user=False, key=f"bot_msg_{len(st.session_state['chat_history'])}")
                    
                    # Update user's chat history in the database
                    users_collection.update_one({"user_id": user_id}, {"$set": {"chat_history": st.session_state['chat_history']}}, upsert=True)
                    
                    # Convert response to speech
                    text_to_speech(response_text, language)

            # Display chat history
            with st.expander("Chat History"):
                for msg in st.session_state['chat_history']:
                    message(msg['content'], is_user=(msg['role'] == 'user'), key=msg['content'])

        # About section
        else:
            st.subheader("About")
            st.write("This is a Teaching Assistant app powered by Google Generative AI and Streamlit.")
            st.write("It allows users to ask questions and get responses from the Gemini Pro model.")
            st.write("The app supports text and voice input.")
            st.write("Make sure to use it and give us a review.") 
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        # Log the error or take appropriate action
    
    finally:
        # Close the MongoDB connection
        if client:
            client.close()

if __name__ == "__main__":
    main()
