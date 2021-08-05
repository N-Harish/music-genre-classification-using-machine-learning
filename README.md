# music genre classification

## Technologies used
* Librosa for extracting audio features
* Scikit-learn for machine learning
* Streamlit for deployment
* PyDub for converting mp3 to wav
* Docker for deployment

## Features extracted
* chroma_stft
* rmse
* spectral_centroid
* spectral_bandwidth
* rolloff
* zero_crossing_rate

## Model used
* OutputCodeClassifier with Logistic Regression for multiclass classification


## How it workes
* First the user has an option to either view demo or upload their own audio
* Then if the user selects classify, the web app calculates all the above mentioned features and lassify the audio into one of 10 genres using our model
* If user selects spectogram, it display the mel spectogram to the user


## link of the deployed app
https://music-type-predictor.herokuapp.com
