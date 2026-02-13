import joblib

# Assuming `model` is your trained model and `scaler` is your fitted scaler

# Save the model
def save_model(model, scaler, model_filename='trained_model.pkl', scaler_filename='scaler.pkl'):
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f'Model saved as {model_filename} and scaler saved as {scaler_filename}')

# Call this function after training to save the model and scaler
# Example:
# save_model(trained_model, fitted_scaler)