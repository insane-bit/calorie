from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load saved objects
model = joblib.load('best_calorie_model.pkl')
scaler = joblib.load('scaler.pkl')
exercise_encoder = joblib.load('exercise_encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get input data from form
            Age = float(request.form['Age'])
            Height = float(request.form['Height'])
            Weight = float(request.form['Weight'])
            Gender = 0 if request.form['Gender'] == 'male' else 1
            Duration = float(request.form['Duration'])
            Exercise_Type = request.form['Exercise_Type']
            Time_of_Day = int(request.form['Time_of_Day'])  # 0,1,2,3

            # Encode exercise type
            exercise_encoded = exercise_encoder.transform([Exercise_Type])[0]

            # Create feature array (order same as training)
            X = np.array([[Age, Height, Weight, Gender, Duration, exercise_encoded, Time_of_Day]])

            # Scale features (only numeric ones, adjust if needed)
            # If scaler expects all features, make sure order is correct and includes all
            X_scaled = scaler.transform(X)

            # Predict calories
            calories_pred = model.predict(X_scaled)[0]

            return render_template('index.html', prediction=f"Predicted Calories Burnt: {calories_pred:.2f}")

        except Exception as e:
            return render_template('index.html', prediction=f"Error: {str(e)}")

    return render_template('index.html')

if __name__ == "__main__":
  app.run(debug=True)