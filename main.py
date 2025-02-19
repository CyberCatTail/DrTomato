import streamlit as st
import time
import torch
from PIL import Image
import timm
from torchvision import transforms
from openai import OpenAI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = timm.create_model(
    "resnet50d", pretrained=True, num_classes=10, drop_path_rate=0.05)

model.load_state_dict(torch.load('./model_weights.pth', map_location=device))

model = model.to(device)
model.eval()

data_config = timm.data.resolve_data_config({}, model=model, verbose=True)
data_mean = data_config["mean"]
data_std = data_config["std"]

classes = {0: 'Bacterial Spot',
           1: 'Early Blight', 
           2: 'Late Blight', 
           3: 'Leaf Mold', 
           4: 'Septoria Leaf Spot', 
           5: 'Spider Mites Two-spotted spider mite',
           6: 'Target Spot', 
           7: 'Tomato Yellow Leaf Curl Virus', 
           8: 'Tomato Mosaic Virus', 
           9: 'Healthy'}

image_size = (256, 256)

transformer = transforms.Compose([transforms.Resize(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=data_mean, std=data_std)])

@torch.no_grad()
def classify(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transformer(image)
    image_tensor.unsqueeze_(0)
    output = model(image_tensor)
    index = output.data.numpy().argmax()
    pred = classes[index]
    return pred

client = OpenAI()

def LLM(status):
    try:
        content = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"my tomato is in {status} disease, please give me suggestions"
                }
            ]
        )
        return content.choices[0].message.content
    except Exception as e:
        return "" 

def calc(img):
    status = classify(img)
    if status == 'Healthy':
        st.write("Congras, your tomato looks very healthy.")
    else:
        suggestion = LLM(status)
        st.write(f"Your tomatoes are highly likely to have **{status}** disease.")
        st.write(suggestion)
        st.write("we also suggest consulting with our plant experts. You can reach them via email: drtomato.clinic@uwo.ca.")
        st.write("---")
        st.write("**Disclaimer:** This recommendation is for reference only and should not be considered a substitute for professional diagnosis or treatment. Please consult a qualified expert before taking any action.")

def generate_response(img):
    with st.chat_message("ai", avatar='./avator.jpeg'):
        with st.status("Diagnosing...", expanded=True):
            calc(img)

def pick_img():
    uploaded_file = st.file_uploader(
    "Choose a photo", type=["jpg", "jpeg", "png", "gif", "bmp"]
    )
    if uploaded_file:
        st.session_state.img = uploaded_file
        
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