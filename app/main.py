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
            <h1>Welcome to Pankaj's Beer Style Prediction API. </h1>
            <br><br>
            <h2>Here's how you use this API:</h2>
            <br><br>
            <h3><b>/health</b> -> Displays health of the model and application</h3>
            <br><br>
            <h3><b>/beer/type</b> -> Requires following 4 parameters. Returns a single beer style prediction based on the inputs provided</h3>
            <h3>      brewery_name: <i>string [Mandatory] Name of the brewery that makes the beer </i> </h3>
            <h3>      rev_aroma:  <i>float [Mandatory] - A score between 1.0 and 5.0 for Review Aroma </i></h3>
            <h3>      rev_appearance: <i>float [Mandatory] - A score between 1.0 and 5.0 for Review Appearance </i> </h3>
            <h3>      rev_palate:   <i> float [Mandatory] - A score between 1.0 and 5.0 for Review Palate </i> </h3>
            <h3>      rev_taste:   <i> float [Mandatory] - A score between 1.0 and 5.0 for Review Taste </i> </h3>
            <br><br>
            <h3><b>/beers/type</b> -> Requires a JSON formatted string with a comma separated list of beer details in the following form, Parameter name is beer_list_in_jason. Returns multiple predictions, one for each beer type provided. </h3>
            <h3><i>beer_list_in_json = [{"brewery_name": BREWERY NAME 1, "rev_aroma": 1.0, "rev_appearance": 1.5, "rev_palate": 5.0, "rev_taste": 4.0} {"brewery_name": BREWERY NAME 2, "rev_aroma": 3.0, "rev_appearance": 4.5, "rev_palate": 3.0, "rev_taste": 2.0}] </i> </h3>
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

@app.get("/beer/type")
def predict(brewery_name: str, rev_aroma: float, rev_taste: float, rev_appearance: float, rev_palate: float):
    from feature_encode import BeerStyleCode
    from joblib import load
    
    rfc_pipeline = load("../models/pipeline_model_rfc.joblib")
    beer_style_decode = BeerStyleCode()
    
    features = format_features(brewery_name, rev_aroma, rev_taste, rev_appearance, rev_palate)
    obs = pd.DataFrame(features)
    pred = beer_style_decode.inverse_transform(rfc_pipeline.predict(obs))
    return JSONResponse(pred["beer_style"].to_json())

@app.get("/beers/type")
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
        #print(pred_beer_style["beer_style"])
        beer_style_list.append(str(pred_beer_style["beer_style"]))
        #print(beer_style_list[i])
        
    beers_df = pd.DataFrame(beer_style_list, columns=["beer_style"]) #initialize an empty dataframe
    #print(beers_df.head())
    #print(beers_df["beer_style"].to_json())
    
    return JSONResponse(beers_df["beer_style"].to_json())

