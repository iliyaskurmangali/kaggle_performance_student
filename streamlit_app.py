import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model and preprocessor
best_model = joblib.load('best_model.joblib')
preprocessor = joblib.load('preprocessor.joblib')

# Define the Streamlit app
def main():
    st.title("Student GPA Prediction App")

    st.write("""
    Enter the following information to predict the GPA of a student:
    """)

    # Input fields for each feature
    study_time_weekly = st.slider("Weekly Study Time (hours)", min_value=0, max_value=20, value=10)
    absences = st.slider("Number of Absences", min_value=0, max_value=30, value=5)
    tutoring = st.selectbox("Tutoring", options=['No', 'Yes'])
    parental_support = st.selectbox("Parental Support", options=['None', 'Low', 'Moderate', 'High', 'Very High'])
    extracurricular = st.selectbox("Extracurricular Activities", options=['No', 'Yes'])

    # Button to trigger prediction
    if st.button("Predict GPA"):
        try:
            # Create a DataFrame from input data
            input_data = {
                'StudyTimeWeekly': [study_time_weekly],
                'Absences': [absences],
                'Tutoring': [1 if tutoring == 'Yes' else 0],
                'ParentalSupport': [0 if parental_support == 'None' else 1 if parental_support == 'Low' else 2 if parental_support == 'Moderate' else 3 if parental_support == 'High' else 4],
                'Extracurricular': [1 if extracurricular == 'Yes' else 0]
            }
            input_df = pd.DataFrame(input_data)

            # Ensure the DataFrame has the correct column names
            input_df.columns = ['StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular']

            # Preprocess input data using preprocessor pipeline
            input_processed = pd.DataFrame(preprocessor.transform(input_df), columns=input_df.columns)

            # Make prediction using the model
            prediction = best_model.predict(input_processed)

            # Display prediction
            st.subheader('Predicted GPA')
            st.write(f"{prediction[0]:.2f}")  # Display GPA rounded to two decimal places

        except ValueError as e:
            st.error(f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    main()
