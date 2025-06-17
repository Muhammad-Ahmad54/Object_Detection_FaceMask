import streamlit as st
from PIL import Image
import tempfile
import cv2
from ultralytics import YOLO

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO('yolo11n.pt')  # make sure this file is in the same directory or provide full path

model = load_model()

st.title("Face Mask Detection App 😷")
st.write("Upload an image or video to detect face masks using a YOLO model.")

option = st.radio("Choose input type:", ('Image', 'Video'))

if option == 'Image':
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        with st.spinner("Running detection..."):
            results = model.predict(image)

        st.success("Detection Complete")
        for r in results:
            img = r.plot()  # Draw boxes
            st.image(img, caption="Detection Output", use_container_width=True)

elif option == 'Video':
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        st.success("Running video detection...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame)
            for r in results:
                annotated_frame = r.plot()

            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()
