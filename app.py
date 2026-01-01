import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# настройки страницы
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# заголовок
st.title("Car Price Predictor")
st.caption("Прогноз цены автомобиля и анализ данных")

# загрузка модели
@st.cache_resource
def load_model():
    with open("models/ridge_ohe_model.pkl", "rb") as f:
        return pickle.load(f)

model_data = load_model()
model = model_data["model"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]
df_train = model_data["train_df"]

# палитра для графиков
palette = sns.color_palette("pastel")
sns.set_style("whitegrid")

# состояние текущей вкладки
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'EDA'

def select_tab(tab_name):
    st.session_state.current_tab = tab_name

# вкладки как кнопки
tab_cols = st.columns(4)
tab_titles = ["EDA", "Категориальные", "Прогноз", "Коэффициенты"]
tab_icons = ["📊", "🧩", "🔮", "⚙️"]  # оставил только иконки без смайликов в тексте
tab_colors = ["#FFC300", "#FF5733", "#33C3FF", "#75FF33"]

for i, col in enumerate(tab_cols):
    if st.session_state.current_tab == tab_titles[i]:
        col.button(f"{tab_titles[i]}", key=i,
                   on_click=select_tab, args=(tab_titles[i],),
                   use_container_width=True)
    else:
        col.button(f"{tab_titles[i]}", key=f"other_{i}",
                   on_click=select_tab, args=(tab_titles[i],),
                   use_container_width=True)

st.markdown("---")

# eda
if st.session_state.current_tab == "EDA":
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("Описание числовых признаков:")
    st.dataframe(df_train.describe().T)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Корреляционная матрица")
        plt.figure(figsize=(7,5))
        sns.heatmap(df_train.select_dtypes(include='number').corr(),
                    annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(plt.gcf())
    with col2:
        st.subheader("Распределение цены")
        plt.figure(figsize=(7,4))
        sns.histplot(df_train['selling_price'], kde=True, color=palette[0])
        st.pyplot(plt.gcf())

# категориальные признаки
elif st.session_state.current_tab == "Категориальные":
    st.header("Категориальные признаки")
    st.markdown("Распределение цены по категориям:")

    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    cat_titles = {
        'fuel': "Тип топлива",
        'seller_type': "Тип продавца",
        'transmission': "Коробка передач",
        'owner': "Количество владельцев",
        'seats': "Количество мест"
    }

    cols_layout = st.columns(2)
    for i, col_name in enumerate(cat_cols):
        plt.figure(figsize=(6,3))
        sns.boxplot(x=col_name, y='selling_price', data=df_train, palette=palette)
        plt.title(cat_titles[col_name])
        if i % 2 == 0:
            with cols_layout[0]:
                st.pyplot(plt.gcf())
        else:
            with cols_layout[1]:
                st.pyplot(plt.gcf())

# прогноз
elif st.session_state.current_tab == "Прогноз":
    st.header("Прогноз цены автомобиля")
    st.markdown("Выберите способ ввода данных:")

    pred_tab = st.radio("", ["CSV файл", "Ручной ввод"], horizontal=True)

    def prepare_input(df):
        df_enc = pd.get_dummies(df, drop_first=True)
        df_enc = df_enc.reindex(columns=feature_names, fill_value=0)
        X_scaled = scaler.transform(df_enc)
        return X_scaled

    if pred_tab == "CSV файл":
        st.markdown("Загрузите CSV с признаками. Пример:")
        st.markdown("""
| year | km_driven | mileage | engine | max_power | seats | fuel | seller_type | transmission | owner |
|------|-----------|---------|--------|-----------|-------|------|------------|-------------|-------|
| 2015 | 50000     | 18.0    | 1197   | 77        | 5     | Petrol | Individual | Manual | First Owner |
""")
        uploaded_file = st.file_uploader("Выберите CSV", type=["csv"])
        if uploaded_file:
            input_df = pd.read_csv(uploaded_file)
            st.dataframe(input_df.head())
            try:
                X_scaled = prepare_input(input_df)
                predictions = model.predict(X_scaled)
                st.success("Прогноз цены:")
                for i, pred in enumerate(predictions):
                    st.metric(label=f"Автомобиль {i+1}", value=f"{np.expm1(pred):,.0f} ₹")
            except Exception as e:
                st.error(f"Ошибка при применении модели: {e}")
    else:
        st.markdown("Введите значения признаков вручную:")

        numeric_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
        cat_features = {
            'fuel': ['Petrol', 'Diesel', 'LPG'],
            'seller_type': ['Individual', 'Trustmark Dealer'],
            'transmission': ['Manual', 'Automatic'],
            'owner': ['First Owner','Second Owner','Third Owner','Fourth & Above Owner','Test Drive Car'],
            'seats': [4,5,6,7,8,9,10,14]
        }

        input_dict = {}
        col1, col2, col3 = st.columns(3)

        for i, feat in enumerate(numeric_features):
            col = [col1, col2, col3][i % 3]
            input_dict[feat] = col.number_input(f"{feat}", value=0.0)

        for feat, options in cat_features.items():
            input_dict[feat] = st.selectbox(f"{feat}", options)

        input_df = pd.DataFrame([input_dict])
        st.dataframe(input_df)

        try:
            X_scaled = prepare_input(input_df)
            predictions = model.predict(X_scaled)
            st.success("Прогноз:")
            st.metric("Стоимость автомобиля", f"{np.expm1(predictions[0]):,.0f} ₹")
        except Exception as e:
            st.error(f"Ошибка при применении модели: {e}")

# коэффициенты модели
elif st.session_state.current_tab == "Коэффициенты":
    st.header("Коэффициенты модели")
    st.markdown("График коэффициентов Ridge.")

    coef_series = pd.Series(model.coef_, index=feature_names).sort_values(key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(10,4))
    coef_series.plot(kind='bar', color='blue', ax=ax)
    ax.set_ylabel("Коэффициент")
    ax.set_xlabel("Признак")
    st.pyplot(fig)

    st.subheader("Описание признаков")
    feature_desc = {
        "year": "Год выпуска автомобиля",
        "km_driven": "Пробег автомобиля (км)",
        "mileage": "Расход топлива (км/л)",
        "engine": "Объем двигателя (см³)",
        "max_power": "Мощность двигателя (л.с.)",
        "fuel_Diesel": "Автомобиль с дизельным двигателем",
        "fuel_Petrol": "Автомобиль с бензиновым двигателем",
        "fuel_LPG": "Автомобиль на LPG",
        "seller_type_Individual": "Продаёт частное лицо",
        "seller_type_Trustmark Dealer": "Продаёт доверенный дилер",
        "transmission_Manual": "Механическая коробка передач",
        "owner_First Owner": "Первый владелец",
        "owner_Second Owner": "Второй владелец",
        "owner_Third Owner": "Третий владелец",
        "owner_Fourth & Above Owner": "Четвёртый и выше",
        "owner_Test Drive Car": "Тест-драйв автомобиль",
        "seats_4":"Автомобиль с 4 местами",
        "seats_5":"Автомобиль с 5 местами",
        "seats_6":"Автомобиль с 6 местами",
        "seats_7":"Автомобиль с 7 местами",
        "seats_8":"Автомобиль с 8 местами",
        "seats_9":"Автомобиль с 9 местами",
        "seats_10":"Автомобиль с 10 местами",
        "seats_14":"Автомобиль с 14 местами"}
    st.dataframe(pd.DataFrame.from_dict(feature_desc, orient='index', columns=['Описание']))
