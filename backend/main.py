from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware    
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import os
import math

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from backend.from_scratch import LinearRegressionScratch
from backend.regression_tree import RegressionTree
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "app/model_store/model.joblib"
SCRATCH_MODEL_PATH = "app/model_store/scratch_model.joblib"
TREE_MODEL_PATH = "app/model_store/tree_model.joblib"
FEATURE_PATH = "app/model_store/features.joblib"


@app.post("/train")
async def train(
    file: UploadFile = File(...),
    learning_rate: float = Form(0.01),
    iterations: int = Form(100),
    max_depth: int = Form(0),
    min_samples: int = Form(0),
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

    # Auto-tune tree size based on training set size when users pass 0.
    n_train = len(X_train)
    if max_depth <= 0:
        max_depth = max(1, int(math.log2(n_train)) - 1)
    if min_samples <= 0:
        min_samples = max(2, int(n_train * 0.02))

    tree_model = RegressionTree(max_depth=max_depth, min_samples=min_samples)
    tree_model.fit(X_train.to_numpy(), y_train.to_numpy())

    os.makedirs("app/model_store", exist_ok=True)

    sklearn_pred = model.predict(X_test)
    scratch_pred = scratch_model.predict(X_test)
    tree_pred = tree_model.predict(X_test.to_numpy())

    sklearn_mse = mean_squared_error(y_test, sklearn_pred)
    scratch_mse = mean_squared_error(y_test, scratch_pred)

    sklearn_score = r2_score(y_test, sklearn_pred)
    scratch_score = r2_score(y_test, scratch_pred)
    tree_score = r2_score(y_test, tree_pred)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scratch_model, SCRATCH_MODEL_PATH)
    joblib.dump(tree_model, TREE_MODEL_PATH)
    joblib.dump(list(X.columns), FEATURE_PATH)

    return {
        "message": "models trained",
        "features": list(X.columns),
        "rows": len(df),
        "sklearn_mse": sklearn_mse,
        "scratch_mse": scratch_mse,
        "tree_mse": mean_squared_error(y_test, tree_pred),
        "sklearn_score": sklearn_score,
        "scratch_score": scratch_score,
        "tree_score": tree_score,
        "tree_max_depth": max_depth,
        "tree_min_samples": min_samples,
        "scratch_learning_rate": learning_rate,
        "scratch_iterations": iterations,
    }


@app.post("/predict")
async def predict(data: dict):

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCRATCH_MODEL_PATH):
        return {"error": "models not trained"}

    model = joblib.load(MODEL_PATH)
    scratch_model = joblib.load(SCRATCH_MODEL_PATH)
    tree_model = joblib.load(TREE_MODEL_PATH)
    features = joblib.load(FEATURE_PATH)

    try:
        X = pd.DataFrame([data])[features]
    except KeyError:
        return {"error": f"Expected features: {features}"}

    sklearn_pred = model.predict(X)[0]
    scratch_pred = scratch_model.predict(X)[0]
    tree_pred = tree_model.predict(X.to_numpy())[0]

    return {
        "sklearn_prediction": float(sklearn_pred),
        "scratch_prediction": float(scratch_pred),
        "tree_prediction": float(tree_pred)
    }

@app.get("/")
def health():
    return {"status": "running"}

# serve the HTML/CSS/JS so you can just open http://localhost:8000
app.mount("/", StaticFiles(directory=".", html=True), name="static")