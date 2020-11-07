import streamlit as st
from audio_feature.audio_featurizer import audio_process, spectrogram_plot
from models.load_model import model_loader
import numpy as np
from pydub import AudioSegment
import os
import subprocess


model, encoding = model_loader("Saved_model.sav", "Encodings.sav")

st.sidebar.markdown(
    """<h1 style='text-align: center;color:  #0e76a8;'><a style='text-align: center;color:  #0e76a8;' href="https://www.linkedin.com/in/harish-natarajan-82a4b418b/" target="_blank">Linkedin Profile</a></h1>""",
    unsafe_allow_html=True)
# st.sidebar.markdown("""<h1 style='text-align: center;color: black;' ><a style='text-align: center;color: black;'href="https://github.com/rohankokkula/TEATH" target="_blank">Github Source Code</a></h1>""", unsafe_allow_html=True)

st.sidebar.markdown(
    """<style>body {background-color: #2C3454; background-image: url('https://i2.wp.com/highland-music.com/wp-content/uploads/2016/04/Blue-Background-Music-Headphone-Wallpaper-Picture-HD-Free-298292334-e1459743028815.png?ssl=1'); color:white;}</style><body></body>""",
    unsafe_allow_html=True)
st.markdown(
    """<h1 style='text-align: center; color: white;font-size:60px;margin-top:-50px;'>AUDIO CLASSIFIER</h1><h1 style='text-align: center; color: white;font-size:30px;margin-top:-30px;'>Using Machine Learning</h1>""",
    unsafe_allow_html=True)

radio = st.sidebar.radio("Select format of audio file", options=['mp3', 'wav'])
#radio = st.sidebar.radio("Select format of audio file", options=['wav'])

if radio == 'wav':

    file = st.sidebar.file_uploader("Upload Audio To Classify", type=["wav"])

    if file is not None:
        st.markdown(
            """<h1 style='color:yellow;'>Audio : </h1>""",
            unsafe_allow_html=True)
        st.audio(file)

        rad = st.sidebar.radio("Choose Options", options=["Predict", "Spectrogram"])

        # rad = st.sidebar.checkbox(label="Do You want to see the spectrogram ?")
        if rad == "Predict":
            if st.button("Classify Audio"):
                uploaded_audio = audio_process(file)

                predictions = model.predict(uploaded_audio)

                targets = encoding.inverse_transform(np.array(predictions).reshape(1, -1))
                #
                # st.write(targets[0][0])
                #
                # st.success(targets[0][0])

                st.markdown(
                    f"""<h1 style='color:yellow;'>Prediction : <span style='color:white;'>{targets[0][0]}</span></h1>""",
                    unsafe_allow_html=True)

        elif rad == "Spectrogram":
            fig = spectrogram_plot(file)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.markdown(
                f"""<h1 style='color:yellow;'>Spectrogram : </h1>""",
                unsafe_allow_html=True)
            st.pyplot(fig)



elif radio == 'mp3':
    file = st.sidebar.file_uploader("Upload Audio To Classify", type="mp3")
#    
    if file is not None:
         #subprocess.call(['ffmpeg', '-i', f'{file}', '-acodec', 'pcm_u8', '-ar', '22050', 'file.wav'])
         subprocess.call(f'ffmpeg -i {file} -acodec pcm_u8 -ar 22050 file.wav', shell=True)
#        sound = AudioSegment.from_mp3(file)
#        sound.export("file.wav", format="wav")
         st.markdown(
            """<h1 style='color:yellow;'>Audio : </h1>""",
            unsafe_allow_html=True)
         a = st.audio(file, format="audio/mp3")

        #rad = st.sidebar.radio("Choose Options", options=["Predict", "Spectrogram"])
         rad = st.sidebar.radio("Choose Options", options=["Predict"])
        # rad = st.sidebar.checkbox(label="Do You want to see the spectrogram ?")
         if rad == "Predict":
             if st.button("Classify Audio"):
                uploaded_audio = audio_process("file.wav")

                predictions = model.predict(uploaded_audio)

                targets = encoding.inverse_transform(np.array(predictions).reshape(1, -1))
                #
                # st.write(targets[0][0])
                #
                # st.success(targets[0][0])

                st.markdown(
                    f"""<h1 style='color:yellow;'>Prediction : <span style='color:white;'>{targets[0][0]}</span></h1>""",
                    unsafe_allow_html=True)

#        elif rad == "Spectrogram":
#            fig = spectrogram_plot("file.wav")
#            st.set_option('deprecation.showPyplotGlobalUse', False)
#            st.markdown(
#                f"""<h1 style='color:yellow;'>Spectrogram : </h1>""",
#                unsafe_allow_html=True)
#            st.pyplot(fig)

        # sound = AudioSegment.from_mp3(file)
        # st.write("Please Upload in wav form")
        # st.markdown(
        #     """<h1 style='color:yellow;'>Audio : </h1>""",
        #     unsafe_allow_html=True)
        # st.audio(file)

#        os.remove("file.wav")
