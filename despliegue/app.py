from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import traceback

# 1. Inicializamos la app
app = FastAPI(title="API de Predicción de ABV")

# 2. Cargamos el pipeline entrenado
pipeline = joblib.load("xgb_end_model.pkl")  # asegúrate de tener este archivo junto a app.py

# 3. Definición de esquema de entrada
class MostoData(BaseModel):
    reposo2_min: float
    reposo2_temp: float
    reposo3_min: float
    reposo3_temp: float
    reposo4_min: float
    reposo4_temp: float
    pH_1: float
    primer_mosto_extracto: float
    ultima_agua_extracto: float
    agua_lavado_temp: float
    temperatura_llenado: float
    paila_llena_extracto: float
    ebullicion_minutos: float
    ebullicion_temp: float
    mosto_frio_extracto: float

    class Config:
        allow_population_by_field_name = True

# 4. Endpoint de predicción
@app.post("/predict")
def predict_abv(data: MostoData):
    try:
        # Convertimos el request a DataFrame
        df = pd.DataFrame([data.dict(by_alias=True)])
        # Llamamos al pipeline para obtener predicción
        abv_np = pipeline.predict(df)[0]
        abv_py = float(abv_np)
        return {"abv_predicho_%vol": round(abv_py, 3)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
# 5. Endpoint de prueba
@app.get("/")
def read_root():
    return {"message": "API de ABV funcionando. Usa /predict con POST para predecir."}
