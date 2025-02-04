import os
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from fastapi.middleware.cors import CORSMiddleware


cacheModels = {}
cachesScalars = {}

WINDOW = 60
HORIZON = 4
EOPCHS = 30
BATCH = 16
ALPHA = 0.001
DEVICE = torch.device("cpu")

app = FastAPI(title="Stock Predicting Model (FOR NSE)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change this to restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

class StockRequest(BaseModel):
    stockName: str
    windowOverride: int = WINDOW
    horizon: int = HORIZON

def fetchStockData(stockName: str, period: str = "5yr"):
    ticker = f"{stockName}.NS"
    data = yf.download(ticker, period=period)
    if data.empty:
        raise ValueError(f"No data found for {stockName}")
    data = data[["Close"]]
    data.dropna(inplace=True)
    return data

def createDataSet(series: np.array, windowSize: int, horizon: int):
    X, Y = [], []
    for i in range(len(series) - windowSize - horizon + 1):
        X.append(series[i:i+windowSize])
        Y.append(series[i+windowSize:i+windowSize+horizon])
    return np.array(X), np.array(Y)


class StockLSTM(nn.Module):
    def __init__(self, inputSize: int = 1, hiddenState: int = 50, numLayer: int = 1, outputSize: int = HORIZON):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(inputSize, hiddenState, numLayer, batch_first=True)
        self.layer1 = nn.Linear(hiddenState, outputSize)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden)
        out = out[:, -1, :]  # Take the last timestep output -> (batch, hidden)
        out = self.layer1(out)  # Linear layer -> (batch, outputSize)
        return out

def trainer(stockName: str, windowSize: int, horizon: int):
    data = fetchStockData(stockName, period="5y")
    prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledPrices = scaler.fit_transform(prices)
    xNp, yNp = createDataSet(scaledPrices, windowSize, horizon)
    if len(xNp) == 0:
        raise ValueError("Not enough data to process")
    xTensor = torch.tensor(xNp, dtype=torch.float32)
    yTensor = torch.tensor(yNp, dtype=torch.float32)
    # Do NOT unsqueeze here because xNp is already (samples, windowSize, 1)
    # xTensor = xTensor.unsqueeze(-1)
    dataset = TensorDataset(xTensor, yTensor)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH, shuffle=True)
    model = StockLSTM(inputSize=1, hiddenState=50, numLayer=1, outputSize=horizon)
    lossfn = nn.MSELoss()
    optr = torch.optim.Adam(model.parameters(), lr=ALPHA)
    model.train()
    for epoch in range(EOPCHS):
        eLosses = []
        for batchX, batchY in dataloader:
            batchX = batchX.to(DEVICE)
            batchY = batchY.to(DEVICE)
            optr.zero_grad()
            outputs = model(batchX)
            loss = lossfn(outputs, batchY.squeeze(-1))
            loss.backward()
            optr.step()
            eLosses.append(loss.item())
        print(f"Epoch {epoch+1}/{EOPCHS}, loss = {np.mean(eLosses):.6f}")
    return model, scaler

@app.post("/predict")
async def predict(request: StockRequest):
    stockName = request.stockName.upper().strip()
    windowOverride = request.windowOverride
    horizon = request.horizon

    try:
        if stockName in cacheModels:
            model = cacheModels[stockName]
            scalar = cachesScalars[stockName]
        else:
            model, scalar = trainer(stockName=stockName, windowSize=windowOverride, horizon=horizon)
            cacheModels[stockName] = model
            cachesScalars[stockName] = scalar

        model.eval()
        data = fetchStockData(stockName, "1y")
        prices = data['Close'].values.reshape(-1, 1)  # shape: (n, 1)
        scaledPrices = scalar.transform(prices)

        if len(scaledPrices) < windowOverride:
            raise HTTPException(status_code=400, detail="Not enough data for prediction")

        inputWindow = scaledPrices[-windowOverride:]  # shape: (windowOverride, 1)

        # Convert to tensor and add batch dimension.
        # Final shape: (batch_size=1, sequence_length=windowOverride, input_size=1)
        inputWindow = torch.tensor(inputWindow, dtype=torch.float32)
        inputWindow = inputWindow.unsqueeze(0)  # Now shape is (1, 60, 1)
        print("Input tensor shape:", inputWindow.shape)  # Debug: Should be (1, 60, 1)

        with torch.no_grad():
            scaledPrediction = model(inputWindow)

        scaledPrediction = scaledPrediction.cpu().numpy().reshape(-1, 1)
        prediction = scalar.inverse_transform(scaledPrediction).flatten().tolist()
        response = {
            "stock": stockName,
            "predicted_prices": {f"day:{i+1}": price for i, price in enumerate(prediction)}
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_endpoint(request: StockRequest):
    stockName = request.stockName.upper().strip()
    windowOverride = request.windowOverride
    horizon = request.horizon

    try:
        model, scaler = trainer(stockName, windowOverride, horizon)
        cacheModels[stockName] = model
        cachesScalars[stockName] = scaler
        return {"message": f"Model for {stockName} trained successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
