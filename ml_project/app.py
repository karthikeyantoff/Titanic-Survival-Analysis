import pandas as pd
import joblib
print("Loading saved model")
model = joblib.load('saved_titanic_datasets/titanic_model.joblib')
model_columns = joblib.load('saved_titanic_datasets/model_columns.joblib')
print("Model loaded.")
print("\nPlease enter the passenger's details:")
pclass = int(input("Passenger Class (1, 2, or 3): "))
sex = input("Sex (male or female): ").lower()
age = float(input("Age: "))
fare = float(input("Fare (e.g., 7.25): "))
family_size = int(input("Total Family Size (including self): "))

new_person_data = {
    'Pclass': [pclass],
    'Age': [age],
    'Fare': [fare],
    'FamilySize': [family_size],
    'Sex_male': [1 if sex == 'male' else 0],
}
new_df = pd.DataFrame(new_person_data)
new_df = new_df.reindex(columns=model_columns, fill_value=0)
prediction = model.predict(new_df)
print("\n--- Prediction Result ---")
if prediction[0] == 1:
    print("This person would have likely SURVIVED.")
else:
    print("This person would have likely NOT SURVIVED.")