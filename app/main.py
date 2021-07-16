from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()


@app.get("/")
def read_root():
    return {"Beer Style Prediction. Usage: /health -> Check cluster health. /beer/predictstyle?<params> -> Generate Prediction. /beer/stylesPred?<params> -> Generate predictions for a list of inputs."}

@app.get("/health")
def healthcheck():
    return 'Beer Prediction application is online and ready to go!'

def format_features(brewery_name: str, rev_aroma: float, rev_taste: float, rev_appearance: float, rev_palate: float):
    return {
        "brewery_name": [brewery_name],
        "review_aroma": [rev_aroma],
        "review_taste": [rev_taste],
        "review_appearance": [rev_appearance],
        "review_palate": [rev_palate]
    }

@app.get("/beer/predictstyle")
def predict(brewery_name: str, rev_aroma: float, rev_taste: float, rev_appearance: float, rev_palate: float):
    from feature_encode import BeerStyleCode
    from joblib import load
    
    rfc_pipeline = load("../models/pipeline_model_rfc.joblib")
    beer_style_decode = BeerStyleCode()
    
    features = format_features(brewery_name, rev_aroma, rev_taste, rev_appearance, rev_palate)
    obs = pd.DataFrame(features)
    pred = beer_style_decode.inverse_transform(rfc_pipeline.predict(obs))
    return JSONResponse(pred.to_json())

@app.get("/beer/stylespred")
def predict(brewery_name: str, rev_aroma: float, rev_taste: float, rev_appearance: float, rev_palate: float):
    from feature_encode import BeerStyleCode
    from joblib import load
    
    rfc_pipeline = load("../models/pipeline_model_rfc.joblib")
    beer_style_decode = BeerStyleCode()
    
    features = format_features(brewery_name, rev_aroma, rev_taste, rev_appearance, rev_palate)
    obs = pd.DataFrame(features)
    pred = beer_style_decode.inverse_transform(rfc_pipeline.predict(obs))
    return JSONResponse(pred)

