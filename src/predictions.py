import pandas as pd
import joblib

def load_model(model_path):
    """Load the trained model from the given path."""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")

def predict(model, data):
    """Make predictions on the given data using the loaded model."""
    if data is not None:
        try:
            predictions = model.predict(data)
            return predictions
        except Exception as e:
            raise ValueError(f"Error making predictions: {e}")

def save_predictions(features, predictions, predictions_path):
    """Save the predictions along with the feature columns to a CSV file."""
    try:
        # Assuming features is a DataFrame containing the feature columns
        results_df = features.copy()
        results_df['Predictions'] = predictions
        results_df.to_csv(predictions_path, index=False)
        return True
    except Exception as e:
        raise ValueError(f"Error saving predictions: {e}")
