import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_model.pkl")

st.title("Student Pass / Fail Predictor")

gender = st.selectbox("Gender", ["Male", "Female"])

sem1_pass = st.number_input("Semester 1 Subjects Passed", 0, 11)
sem1_fail = st.number_input("Semester 1 Subjects Failed", 0, 11)

sem2_pass = st.number_input("Semester 2 Subjects Passed", 0, 11)
sem2_fail = st.number_input("Semester 2 Subjects Failed", 0, 11)

attendance = st.selectbox(
    "Average Attendance Percentage",
    ["Below 50%", "50%-65%", "66%-75%", "76%-85%", "Above 85%"]
)

assignment_rate = st.slider(
    "Assignment Submission Rate (%)",
    0, 100
)

total_backlogs = st.number_input("Total Backlogs", 0, 10)

if st.button("Predict"):

    gender_map = {"Female": 0, "Male": 1}

    attendance_map = {
        "Below 50%": 0,
        "50%-65%": 1,
        "66%-75%": 2,
        "76%-85%": 3,
        "Above 85%": 4
    }

    input_df = pd.DataFrame([{
        "Gender": gender_map[gender],
        "Semester 1 Subjects Passed": sem1_pass,
        "Semester 1 Subjects Failed": sem1_fail,
        "Semester 2 Subjects Passed": sem2_pass,
        "Semester 2 Subjects Failed": sem2_fail,
        "Average Attendance Percentage": attendance_map[attendance],
        "Assignment Submission Rate (%)": assignment_rate,
        "Total Backlogs": total_backlogs
    }])

    # Make prediction AFTER input_df is created
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("Student is likely to PASS")
    else:
        st.error("Student is likely to FAIL")



