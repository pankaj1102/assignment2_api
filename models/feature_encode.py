from sklearn.base import BaseEstimator, TransformerMixin

class BreweryNameEncodedVal(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        return None
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        from joblib import load
        from sklearn.preprocessing import OrdinalEncoder
        
        enc_value = -1
        bname_model = load("../models/bname.joblib")
        enc_value = bname_model.transform(X)
        
        return enc_value

class BeerStyleCode(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        return None
            
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X, y=None):
        from joblib import load
        import pandas as pd
        
        bstyle_model = load("../models/beerstyle.joblib")
        enc_value = pd.DataFrame(bstyle_model.transform(X), columns=["beer_style"])
        
        return enc_value
    
    def inverse_transform(self, X, y=None):
        from joblib import load
        import pandas as pd
        
        bstyle_model = load("../models/beerstyle.joblib")
        bstyle_txt = pd.DataFrame(bstyle_model.inverse_transform(X), columns = ["beer_style"])
        
        return bstyle_txt
    
