from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware    
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from from_scratch import LinearRegressionScratch
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "app/model_store/model.joblib"
SCRATCH_MODEL_PATH = "app/model_store/scratch_model.joblib"
FEATURE_PATH = "app/model_store/features.joblib"


@app.post("/train")
async def train(
    file: UploadFile = File(...),
    learning_rate: float = Form(0.01),
    iterations: int = Form(100),
):

    df = pd.read_csv(file.file)

    if "target" not in df.columns:
        return {"error": "CSV must contain a 'target' column"}

    X = df.drop(columns=["target"])
    y = df["target"]

    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    scratch_model = LinearRegressionScratch(learning_rate=learning_rate, iterations=iterations)
    scratch_model.fit(X_train, y_train)
    os.makedirs("app/model_store", exist_ok=True)

    sklearn_pred = model.predict(X_test)
    scratch_pred = scratch_model.predict(X_test)

    sklearn_mse = mean_squared_error(y_test, sklearn_pred)
    scratch_mse = mean_squared_error(y_test, scratch_pred)

    sklearn_score = r2_score(y_test, sklearn_pred)
    scratch_score = r2_score(y_test, scratch_pred)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scratch_model, SCRATCH_MODEL_PATH)
    joblib.dump(list(X.columns), FEATURE_PATH)

    return {
        "message": "models trained",
        "features": list(X.columns),
        "rows": len(df),
        "sklearn_mse": sklearn_mse,
        "scratch_mse": scratch_mse,
        "sklearn_score": sklearn_score,
        "scratch_score": scratch_score
    }


@app.post("/predict")
async def predict(data: dict):

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCRATCH_MODEL_PATH):
        return {"error": "models not trained"}

    model = joblib.load(MODEL_PATH)
    scratch_model = joblib.load(SCRATCH_MODEL_PATH)
    features = joblib.load(FEATURE_PATH)

    try:
        X = pd.DataFrame([data])[features]
    except KeyError:
        return {"error": f"Expected features: {features}"}

    sklearn_pred = model.predict(X)[0]
    scratch_pred = scratch_model.predict(X)[0]

    return {
        "sklearn_prediction": float(sklearn_pred),
        "scratch_prediction": float(scratch_pred)
    }

@app.get("/")
def health():
    return {"status": "running"}

# serve the HTML/CSS/JS so you can just open http://localhost:8000
app.mount("/", StaticFiles(directory=".", html=True), name="static")