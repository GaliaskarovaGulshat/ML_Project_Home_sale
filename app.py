import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Оценка стоимости недвижимости", layout="centered")

MODELS_DIR = "models"

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    global_mean = joblib.load(os.path.join(MODELS_DIR, "global_mean_price.pkl"))
    encoding_maps = joblib.load(os.path.join(MODELS_DIR, "target_encoding_maps.pkl"))
    return model, feature_columns, global_mean, encoding_maps


model, feature_columns, global_mean, encoding_maps = load_artifacts()

mae_value = None
if os.path.exists("model_comparison.csv"):
    try:
        comparison = pd.read_csv("model_comparison.csv", index_col=0)
        mae_value = comparison["MAE"].min()
    except Exception:
        mae_value = None


def get_categories(col_name):
    prefix = col_name + "_"
    return sorted(c[len(prefix):] for c in feature_columns if c.startswith(prefix))


подтип_варианты = get_categories("подтип")
отопление_варианты = get_categories("отопление")
валюта_варианты = get_categories("валюта")
БЕЗ_КАТЕГОРИИ = "(другое / не указано)"

город_варианты = sorted(encoding_maps.get("город", pd.Series(dtype=float)).index.astype(str).tolist())
район_варианты = sorted(encoding_maps.get("район", pd.Series(dtype=float)).index.astype(str).tolist())
улица_варианты = sorted(encoding_maps.get("улица", pd.Series(dtype=float)).index.astype(str).tolist())


templates = {
    "Продажа, просторная квартира": {
        "тип_объявления": "Продажа", "площадь": 140.0, "комнат_всего": 4,
        "спальни": 3, "гостиные": 1, "этаж": 5, "всего_этажей": 10,
        "возраст_здания": 8, "время_на_рынке": 30,
    },
    "Аренда, небольшая квартира": {
        "тип_объявления": "Аренда", "площадь": 65.0, "комнат_всего": 2,
        "спальни": 1, "гостиные": 1, "этаж": 3, "всего_этажей": 6,
        "возраст_здания": 15, "время_на_рынке": 10,
    },
    "Студия эконом-класса": {
        "тип_объявления": "Продажа", "площадь": 40.0, "комнат_всего": 1,
        "спальни": 1, "гостиные": 0, "этаж": 1, "всего_этажей": 4,
        "возраст_здания": 25, "время_на_рынке": 60,
    },
}


def apply_template(values):
    st.session_state.update(values)


st.title("Оценка стоимости недвижимости")
st.caption("Прогноз цены на основе модели, обученной на данных портала Zingat (Турция)")

st.write("**Быстрое заполнение примером:**")
cols = st.columns(len(templates))
for col, (label, values) in zip(cols, templates.items()):
    if col.button(label, use_container_width=True):
        apply_template(values)
        st.rerun()

st.divider()

col1, col2 = st.columns(2)

with col1:
    тип_объявления = st.selectbox("Тип объявления", ["Продажа", "Аренда"], key="тип_объявления")
    подтип = st.selectbox(
        "Подтип объекта", подтип_варианты + [БЕЗ_КАТЕГОРИИ], key="подтип"
    ) if подтип_варианты else БЕЗ_КАТЕГОРИИ
    отопление = st.selectbox(
        "Отопление", отопление_варианты + [БЕЗ_КАТЕГОРИИ], key="отопление"
    ) if отопление_варианты else БЕЗ_КАТЕГОРИИ
    валюта = st.selectbox(
        "Валюта объявления", валюта_варианты + [БЕЗ_КАТЕГОРИИ], key="валюта"
    ) if валюта_варианты else БЕЗ_КАТЕГОРИИ
    город = st.selectbox("Город", город_варианты, key="город") if город_варианты else ""
    район = st.selectbox("Район", район_варианты, key="район") if район_варианты else ""
    улица = st.selectbox("Улица", улица_варианты, key="улица") if улица_варианты else ""

with col2:
    площадь = st.number_input("Площадь, м²", min_value=10.0, max_value=1000.0, value=80.0, step=5.0, key="площадь")
    комнат_всего = st.number_input("Комнат всего", min_value=1, max_value=15, value=2, key="комнат_всего")
    спальни = st.number_input("Спален", min_value=0, max_value=10, value=1, key="спальни")
    гостиные = st.number_input("Гостиных", min_value=0, max_value=5, value=1, key="гостиные")
    этаж = st.number_input("Этаж", min_value=-3, max_value=60, value=2, key="этаж")
    всего_этажей = st.number_input("Этажей в доме", min_value=1, max_value=70, value=5, key="всего_этажей")
    возраст_здания = st.number_input("Возраст здания, лет", min_value=0, max_value=100, value=10, key="возраст_здания")
    время_на_рынке = st.number_input("Дней на рынке", min_value=0, max_value=1000, value=30, key="время_на_рынке")

st.divider()


def build_feature_vector():
    row = pd.Series(0.0, index=feature_columns)

    числовые = {
        "возраст_здания": возраст_здания, "этаж": этаж, "площадь": площадь,
        "время_на_рынке": время_на_рынке, "всего_этажей": всего_этажей,
        "комнат_всего": комнат_всего, "спальни": спальни, "гостиные": гостиные,
    }
    for название, значение in числовые.items():
        if название in row.index:
            row[название] = значение

    if "площадь_на_комнату" in row.index:
        row["площадь_на_комнату"] = площадь / комнат_всего if комнат_всего else площадь

    for col_name, выбор in [("подтип", подтип), ("отопление", отопление),
                             ("валюта", валюта), ("тип_объявления", тип_объявления)]:
        dummy_col = f"{col_name}_{выбор}"
        if dummy_col in row.index:
            row[dummy_col] = 1

    for col_name, выбор in [("город", город), ("район", район), ("улица", улица)]:
        te_col = f"{col_name}_te"
        if te_col in row.index:
            карта = encoding_maps.get(col_name, {})
            row[te_col] = карта.get(выбор, global_mean)

    return row.to_frame().T[feature_columns]


if st.button("Рассчитать цену", type="primary", use_container_width=True):
    X_input = build_feature_vector()
    прогноз_лог = model.predict(X_input)[0]
    прогноз_цена = np.expm1(прогноз_лог)

    st.success(f"### Прогнозируемая цена: **{прогноз_цена:,.0f} $**")
    if mae_value is not None:
        st.write(
            f"Ориентировочный диапазон: **{max(0, прогноз_цена - mae_value):,.0f} $ — "
            f"{прогноз_цена + mae_value:,.0f} $** (± средняя ошибка модели)"
        )

with st.expander("О приложении"):
    st.markdown(
        """
        **Что делает приложение.** По характеристикам объекта недвижимости в Турции модель
        предсказывает его цену в долларах США.

        **Модель.** Лучшая из трёх обученных на Этапе 3 моделей (дерево решений, случайный лес,
        градиентный бустинг) — выбрана автоматически по метрике R² на тестовой выборке.

        **Ограничения.** Модель обучена на исторических данных портала Zingat и не учитывает
        текущую рыночную ситуацию, состояние ремонта, вид из окна и другие признаки, которых
        не было в датасете. Прогноз — ориентир, а не точная оценка.
        """
    )
