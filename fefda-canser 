import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open(r"C:/lung_cancer/lung_cancer.sav", "rb"))

# Encoders for dropdown fields
gender_dict = {'Male': 0, 'Female': 1}
country_dict = {
    'Malta': 0, 'Ireland': 1, 'Portugal': 2, 'France': 3, 'Sweden': 4, 'Croatia': 5, 'Greece': 6, 'Spain': 7,
    'Netherlands': 8, 'Denmark': 9, 'Slovenia': 10, 'Belgium': 11, 'Hungary': 12, 'Romania': 13, 'Poland': 14,
    'Italy': 15, 'Germany': 16, 'Estonia': 17, 'Czech Republic': 18, 'Lithuania': 19, 'Slovakia': 20,
    'Austria': 21, 'Finland': 22, 'Luxembourg': 23, 'Cyprus': 24, 'Latvia': 25, 'Bulgaria': 26
}
cancer_stage_dict = {'Stage III': 0, 'Stage IV': 1, 'Stage I': 2, 'Stage II': 3}
family_history_dict = {'No': 0, 'Yes': 1}
smoking_status_dict = {'Passive Smoker': 0, 'Never Smoked': 1, 'Former Smoker': 2, 'Current Smoker': 3}
treatment_type_dict = {'Chemotherapy': 0, 'Surgery': 1, 'Combined': 2, 'Radiation': 3}

# Prediction function
def cancer_prediction(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_array)
    return "High Risk of Lung Cancer" if prediction[0] == 0 else "Low Risk of Lung Cancer"

# Streamlit UI
def main():
    st.title("Lung Cancer Risk Prediction App")

    # Default input values
    age = st.text_input("Age", value="55")
    gender = st.selectbox("Gender", list(gender_dict.keys()), index=0)
    country = st.selectbox("Country", list(country_dict.keys()), index=3)  # France
    cancer_stage = st.selectbox("Cancer Stage", list(cancer_stage_dict.keys()), index=3)  # Stage II
    family_history = st.selectbox("Family History", list(family_history_dict.keys()), index=1)  # Yes
    smoking_status = st.selectbox("Smoking Status", list(smoking_status_dict.keys()), index=3)  # Current Smoker
    bmi = st.text_input("BMI", value="24.5")
    cholesterol_level = st.text_input("Cholesterol Level", value="190")
    hypertension = st.text_input("Hypertension (0 = No, 1 = Yes)", value="1")
    asthma = st.text_input("Asthma (0 = No, 1 = Yes)", value="0")
    cirrhosis = st.text_input("Cirrhosis (0 = No, 1 = Yes)", value="0")
    other_cancer = st.text_input("Other Cancer (0 = No, 1 = Yes)", value="1")
    treatment_type = st.selectbox("Treatment Type", list(treatment_type_dict.keys()), index=0)  # Chemotherapy

    diagnosis = ""

    if st.button("Predict Cancer Risk"):
        try:
            input_data = [
                float(age),
                gender_dict[gender],
                country_dict[country],
                cancer_stage_dict[cancer_stage],
                family_history_dict[family_history],
                smoking_status_dict[smoking_status],
                float(bmi),
                float(cholesterol_level),
                float(hypertension),
                float(asthma),
                float(cirrhosis),
                float(other_cancer),
                treatment_type_dict[treatment_type]
            ]
            diagnosis = cancer_prediction(input_data)
            st.success(diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values in the text fields.")
        except KeyError:
            st.error("Please check your text field entries.")

if __name__ == '__main__':
    main()
