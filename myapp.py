import streamlit as st
from service import predict
import json

st.title('News Categorizer App')

input_string = st.text_input(label = "News Input")

predictions = predict.predictions(input_string)

if len(input_string)>0:
    st.subheader('Categorized News')
    st.write("Your input text:")
    st.write(json.loads(predictions)['text'])
    st.markdown("***")
    st.write("The category of this text:")
    st.write(json.loads(predictions)['category'])