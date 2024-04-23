from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

app = Flask(__name__)

# Load the dataset
data = pd.read_excel('newdataset.xlsx')

# Separate features and target variable
x = data.iloc[:, 1:6]  # Selecting columns 1 to 5 as features
y = data['Failure_Type']

# Standardize features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.7)

# Train the SVC model
model = SVC()
model.fit(xtrain, ytrain)

# Save the trained model to a file
joblib.dump(model, 'svc_model.joblib')

# Define a route for the root URL
@app.route('/')
def index():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request form
    input_data = []
    try:
        input_data.append(float(request.form['Air_temperature_[K]']))
        input_data.append(float(request.form['Rotational_speed_[rpm]']))
        input_data.append(float(request.form['Torque_[Nm]']))
        input_data.append(float(request.form['Tool_wear_[min]']))
        input_data.append(float(request.form['Process_temperature_[K]']))
        
        # Standardize the input data
        input_data = scaler.transform([input_data])
        
        # Make predictions using the trained model
        prediction = model.predict(input_data)[0]
        
        return render_template('index.html', prediction=prediction)
    
    except ValueError:
        # Handle the case where input data cannot be converted to float
        return render_template('index.html', error="Invalid input. Please enter numeric values.")

    except Exception as e:
        # Handle other exceptions
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
