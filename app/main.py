from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

@app.get("/")
def read_root():
    strWelcomeMsg = """
    <html>
        <head>
            <title>Pankaj's Beer Style Prediction API</title>
        </head>
        <body>
            <h1>Welcome to Pankaj's Beer Stle Prediction API. </h1>
            <h2>Here's how you use this API:</h2>
        </body>
    </html>
    """
    return HTMLResponse(content=strWelcomeMsg, status_code=200)

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
    return JSONResponse(pred["beer_style"].to_json())

@app.get("/beer/stylespred")
def predict_beerstyles(beer_list_in_json: str):
    from feature_encode import BeerStyleCode
    from joblib import load
    import pandas as pd
    
    beer_style_list = []
    
    rfc_pipeline = load("../models/pipeline_model_rfc.joblib")
    beer_style_decode = BeerStyleCode()
    
    df_beer = pd.read_json(beer_list_in_json) # Convert incoming json to dataframe
    max_index = df_beer.shape[0] #Get the number of records in the dataframe
    
    for i in range (0, max_index):  # For each record, create a prediction and append it to the beer_style_result string
        brewery_name = df_beer.loc[i]["brewery_name"]
        review_aroma = df_beer.loc[i]["rev_aroma"]
        review_taste = df_beer.loc[i]["rev_taste"]
        review_palate = df_beer.loc[i]["rev_palate"]
        review_appearance = df_beer.loc[i]["rev_appearance"]
        features = format_features(brewery_name, review_aroma, review_taste, review_appearance, review_palate)
        obs = pd.DataFrame(features)
        pred_beer_style = beer_style_decode.inverse_transform(rfc_pipeline.predict(obs))
        print(pred_beer_style["beer_style"])
        beer_style_list.append(str(pred_beer_style["beer_style"]))
        print(beer_style_list[i])
        
    beers_df = pd.DataFrame(beer_style_list, columns=["beer_style"]) #initialize an empty dataframe
    print(beers_df.head())
    print(beers_df["beer_style"].to_json())
    
    return JSONResponse(beers_df["beer_style"].to_json())

