import streamlit as st
import time
# from langchain.llms import OpenAI

def calc(img):
    time.sleep(5)
    st.write("Still in dev...:")


def generate_response(img):
    with st.chat_message("ai", avatar='./avator.jpeg'):
        with st.status("Diagnosing...", expanded=True):
            calc(img)

def pick_img():
    uploaded_file = st.file_uploader(
    "Choose a photo", type=["jpg", "jpeg", "png", "gif", "bmp"]
    )
    if uploaded_file:
        st.session_state.img = uploaded_file.read()
        
    picture = st.camera_input("Or take a picture")
    if picture:
        st.session_state.img = picture

if 'img' not in st.session_state:
    st.session_state.img = None
if 'pre_img' not in st.session_state:
    st.session_state.pre_img = None

def view():
    st.title("🍅 Dr. Tomato")
    with st.sidebar:
        pick_img()

    if st.session_state.img:
        if st.session_state.pre_img != st.session_state.img:
            st.image(st.session_state.img)
            generate_response(st.session_state.img)
            st.session_state.pre_img = st.session_state.img

view()