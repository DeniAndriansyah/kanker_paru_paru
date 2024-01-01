import pickle
import numpy as np
import streamlit as st
from PIL import Image



## load save model
model = pickle.load(open('kanker-paru-paru.sav', 'rb'))

##import data
image = Image.open("kanker.jpg")
st.image(image, use_column_width=True)

## judul web
st.title('Prediksi Kanker Paru-Paru')
st.text('(Deni Andriansyah)')

## atribut
col1, col2, col3 = st.columns(3)

with col1:    
               Age  = st.number_input('Umur')

with col2:
                Gender   = st.number_input('Jenis Kelamin')

with col3:
                Air_Pollution  = st.number_input('Polusi Udara')

with col1:
                Alcohol_use  = st.number_input('Penggunaan Alkohol')

with col2:
               Dust_Allergy  = st.number_input('Alergi Debu')

with col3:
                OccuPational_Hazards = st.number_input('Resiko Genetik')

with col1:
                Genetic_Risk  = st.number_input('Penyakit Paru Paru Kronis')

with col2:
                chronic_Lung_Disease  = st.number_input('Induksi Angina')

with col3:
                Balanced_Diet  = st.number_input('Diet Seimbang')

with col1:
                Obesity  = st.number_input('obesitas')

with col2:
                Coughing_of_Blood   = st.number_input('Batuk Darah')

with col3:
                Fatigue = st.number_input('Kelelahan')

with col1:
                Weight_Loss  = st.number_input('Penurunan Berat Badan')

with col2:
                Shortness_of_Breath= st.number_input('Sesak Nafas')

with col3:
                Wheezing    = st.number_input('Suara Nafas Mengi')

with col1:
                Swallowing_Difficulty    = st.number_input('Kesulitan Menelan')

with col2:
                Clubbing_of_Finger_Nails    = st.number_input('Tabuh Kuku Jari')

with col3:
                Frequent_Cold   = st.number_input('Flu')

with col1:
                Dry_Cough = st.number_input('Batuk Kering')

with col2:
                Snoring    = st.number_input('Mendengkur')


# code for prediction
kanker_diagnosis =''

## membuat tombol prediksi
if st.button('Prediksi Kanker Paru - Paru'):
     prediksi = model.predict([[Age, Gender, Air_Pollution, Alcohol_use, Dust_Allergy, OccuPational_Hazards, Genetic_Risk, chronic_Lung_Disease, Balanced_Diet, Obesity, Coughing_of_Blood, Fatigue, Weight_Loss, Shortness_of_Breath, Wheezing, Swallowing_Difficulty, Clubbing_of_Finger_Nails,  Frequent_Cold, Dry_Cough, Snoring]])

if (prediksi [0] == 0):
        prediksi = 'Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Tinggi'
    elif(prediksi == 2):
        prediksi = 'Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Sedang'
    else:
        prediksi = 'Keparahan Kanker Paru-Paru Pasien Berada di Tingkat Rendah'
st.success(prediksi)

