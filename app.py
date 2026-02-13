import streamlit as st
import pandas as pd
import joblib

model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Student Academic Risk Predictor")

gender = st.selectbox("Gender", ["Male", "Female"])

sem1_pass = st.number_input("Semester 1 Subjects Passed", 0, 11)
sem1_fail = st.number_input("Semester 1 Subjects Failed", 0, 11)

sem2_pass = st.number_input("Semester 2 Subjects Passed", 0, 11)
sem2_fail = st.number_input("Semester 2 Subjects Failed", 0, 11)

sem3_pass = st.number_input("Semester 3 Subjects Passed", 0, 11)
sem3_fail = st.number_input("Semester 3 Subjects Failed", 0, 11)

attendance = st.selectbox(
    "Average Attendance Percentage",
    ["Below 50%", "66%-75%", "76%-85%", "Above 85%"]
)

attendance_eligibility = st.selectbox(
    "Attendance Eligibility for Exams",
    ["Eligible", "Eligible with warning", "Not eligible at least once"]
)

assignment_rate = st.selectbox(
    "Assignment Submission Rate",
    ["Below 50%", "50%-74%", "75%-99%"]
)

timeliness = st.selectbox(
    "Timeliness of Assignment Submission",
    ["Often late", "Sometimes late", "Mostly on time"]
)

if st.button("Predict"):

    gender_map = {"Male": 1, "Female": 0}

    attendance_map = {
        "Below 50%": 0,
        "66%-75%": 1,
        "76%-85%": 2,
        "Above 85%": 3
    }

    eligibility_map = {
        "Eligible": 0,
        "Eligible with warning": 1,
        "Not eligible at least once": 2
    }

    assignment_map = {
        "Below 50%": 0,
        "50%-74%": 1,
        "75%-99%": 2
    }

    timeliness_map = {
        "Often late": 0,
        "Sometimes late": 1,
        "Mostly on time": 2
    }

    total_backlogs = sem1_fail + sem2_fail + sem3_fail

    input_df = pd.DataFrame([{
        "Gender": gender_map[gender],

        "Number of subjects passed in Semester 1 (out of 11)": sem1_pass,
        "Number of subjects failed in Semester 1": sem1_fail,
        "Semester 1 overall result": 1 if sem1_fail == 0 else 0,

        "Number of subjects passed in Semester 2 (out of 11)": sem2_pass,
        "Number of subjects failed in Semester 2": sem2_fail,
        "Semester 2 overall result": 1 if sem2_fail == 0 else 0,

        "Number of subjects passed in Semester 3 (out of 11)": sem3_pass,
        "Number of subjects failed in Semester 3": sem3_fail,
        "Semester 3 overall result": 1 if sem3_fail == 0 else 0,

        "Average attendance percentage": attendance_map[attendance],
        "Attendance eligibility for exams": eligibility_map[attendance_eligibility],
        "Assignment submission rate (average)": assignment_map[assignment_rate],
        "Timeliness of assignment submission": timeliness_map[timeliness],

        "Total number of backlogs till Semester 3": total_backlogs
    }])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("Student likely to PASS")
    else:
        st.error("Student likely to have BACKLOGS")


