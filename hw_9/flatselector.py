import os

import pandas as pd
import streamlit as st

from src.functions import prepare_data, train_model, predict_price

st.set_page_config(
    page_title='Flat selector',
)

model_s = 'flat_price.pkl'

total_square = st.sidebar.number_input('Площадь (кв. м):', 0, 2070 , 137)
rooms = st.sidebar.number_input('Количество комнат (студия 0 комнат):', 0, 15, 3)


inputDF = pd.DataFrame(
    {
        'Площадь': total_square,
        'Количество комнат ': rooms
    },
    index=[0],
)

if not os.path.exists(model_s):
    data = prepare_data()
    data.to_csv('data.csv')
    train_model(data)

if st.button('Предсказать цену'):
    predicted_price = predict_price(rooms, total_square)
    k=f'Предполагаемая стоимость недвижимости: {predicted_price:,.2f} руб.'
    st.write(k.replace(",", " "))

st.image("image/ea9e96d3b5c01eb59ae0099e4399129b.jpg", use_column_width=True)
