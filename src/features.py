# src/features.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

def _clip(X):
    # Exemple simple : limite les valeurs entre -3 et 3
    return X.clip(-3, 3)

def build_numeric_preprocess():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clip", FunctionTransformer(_clip)), # Nouvelle étape
    ])