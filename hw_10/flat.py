
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.func import prepare_data, train_model, predict_price

app = FastAPI()

class PredictionInput(BaseModel):
    total_square: float
    rooms: int

class Result(BaseModel):
    predicted_price: float

data = prepare_data()
train_model(data)


@app.get("/predict_get", response_model=Result)
def predict_get(total_square:float, rooms: int ):
    prediction= predict_price(total_square, rooms)
    return {"predicted_price": prediction}

@app.post("/predict_post", response_model=Result)
def predict_post(total_square:float, rooms: int ):
    result = predict_price(total_square, rooms)
    return {"predicted_price": result}

@app.get("/health")
def health():
    return JSONResponse(content={"message": "Fine!"}, status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

