# Usa una imagen base oficial de Python ligero
FROM python:3.9-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia sólo lo que necesitamos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código y el pipeline
COPY app.py xgb_end_model.pkl ./

# Expone el puerto en que escuchará Uvicorn
EXPOSE 8000

# Comando por defecto al iniciar el contenedor
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
