from pydantic import BaseModel
from typing import List, Optional

class CreditRiskInput(BaseModel):
    TransactionMonth: int
    TransactionCount: int
    TransactionStartTime: Optional[str] = "2026-06-03 00:00:00"
    Amount: Optional[float] = 0.0
    Value: Optional[float] = 0.0
    TotalTransactionAmount: Optional[float] = 0.0
    AverageTransactionAmount: Optional[float] = 0.0
    TransactionVariability: Optional[float] = 0.0
    TransactionHour: Optional[int] = 12
    TransactionDay: Optional[int] = 1
    TransactionYear: Optional[int] = 2026
    ProductCategory: Optional[str] = "Unknown"
    ChannelId: Optional[str] = "Unknown"
    PricingStrategy: Optional[int] = 1
    CurrencyCode: Optional[str] = "Unknown"
    ProviderId: Optional[str] = "Unknown"
    ProductId: Optional[str] = "Unknown"
    CustomerId: str
   

class PredictionResponse(BaseModel):
    probability_of_default: float
    risk_level: str
    recommendation: str
    model_used: str