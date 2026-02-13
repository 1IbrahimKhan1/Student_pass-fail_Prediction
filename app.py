
import streamlit as st
import numpy as np
import joblib

model = joblib.load("student_model.pkl")

st.title("🎓 Student Pass / Fail Prediction")

gender = st.selectbox("Gender (Encoded)", [0, 1])
sem1_pass = st.number_input("Semester 1 Subjects Passed", 0, 10)
sem1_fail = st.number_input("Semester 1 Subjects Failed", 0, 10)
sem2_pass = st.number_input("Semester 2 Subjects Passed", 0, 10)
sem2_fail = st.number_input("Semester 2 Subjects Failed", 0, 10)
attendance = st.number_input("Attendance (%)", 0, 100)
assign_rate = st.number_input("Assignment Submission Rate (%)", 0, 100)
backlogs = st.number_input("Total Backlogs", 0, 10)

if st.button("Predict Result"):
    input_data = np.array([[gender, sem1_pass, sem1_fail,
                            sem2_pass, sem2_fail,
                            attendance, assign_rate,
                            backlogs]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Student is likely to PASS")
    else:
        st.error("❌ Student is likely to FAIL")
