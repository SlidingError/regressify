# Web ML Prediction

Minimal FastAPI service that trains a linear regression model from an uploaded CSV and predicts from JSON input.

## Setup

```bash
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Running

```bash
uvicorn main:app --reload
```

- Visit `http://127.0.0.1:8000` to open the frontend (`index.html` is served automatically).
- **Train**: upload a CSV with a `target` column; you can also set the scratch model learning rate and number of iterations.
- **Predict**: send a JSON object with feature keys matching the training data.

## Notes

- CORS is enabled to allow cross‑origin requests.
- Models and feature names are stored under `app/model_store`.
- Add `scikit-learn` and `joblib` to requirements if missing.

## Improvements (not required)

- add tests (`pytest`), validation models, logging, etc.