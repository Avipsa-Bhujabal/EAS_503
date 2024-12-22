```python
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel

# Load the model
model_filename = 'model_rf.pkl'
pipeline = joblib.load(model_filename)

class InputData(BaseModel):
    features: list

app = FastAPI(title="Heart Disease Prediction API")

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input data to numpy array for prediction
        features = np.array(data.features).reshape(1, -1)
        
        # Make prediction using loaded model
        prediction = pipeline.predict(features)
        
        # Return prediction with more detailed response
        return {
            "prediction": int(prediction[0]),
            "prediction_label": "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```