import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load("iris_classifier_pipeline.joblib")

model = load_model()
target_names = ["Setosa", "Versicolor", "Virginica"]

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")
st.title("🌸 Iris Flower Classifier")
st.write("Enter flower measurements to predict the Iris species.")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.4, 0.1)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.4, 0.1)
with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.5, 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2, 0.1)

if st.button("Predict species", type="primary"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    st.success(f"Predicted species: **{target_names[prediction]}**")
    st.subheader("Prediction probabilities")
    for name, probability in zip(target_names, probabilities):
        st.write(f"{name}: {probability * 100:.1f}%")
        st.progress(float(probability))
