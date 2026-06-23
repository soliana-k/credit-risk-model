from pydantic import BaseModel
from typing import List

class CreditRiskInput(BaseModel):
    TransactionMonth: int
    TransactionCount: int
    TransactionStartTime: str
    Amount: float
    #Value: float
    TotalTransactionAmount: float
    AverageTransactionAmount: float
    TransactionVariability: float
    TransactionHour: int
    TransactionDay: int
    TransactionYear: int
    ProductCategory: str
    ChannelId: str
    PricingStrategy: int
    CurrencyCode: str
    ProviderId: str
    ProductId: str
    CustomerId: str
   

class PredictionResponse(BaseModel):
    probability_of_default: float
    risk_level: str
    recommendation: str
    model_used: str