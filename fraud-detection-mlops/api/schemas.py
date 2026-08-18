from pydantic import BaseModel, Field


class Transaction(BaseModel):
    transaction_id: int = Field(..., json_schema_extra={"example": 1234567})
    customer_id: int = Field(..., json_schema_extra={"example": 54321})
    amount: float = Field(..., gt=0, json_schema_extra={"example": 249.99})
    hour_of_day: int = Field(..., ge=0, le=23, json_schema_extra={"example": 2})
    day_of_week: int = Field(..., ge=0, le=6, json_schema_extra={"example": 5})
    seconds_elapsed: int = Field(..., ge=0, json_schema_extra={"example": 1500000})
    merchant_category: str = Field(..., json_schema_extra={"example": "electronics"})
    distance_from_home_km: float = Field(..., ge=0, json_schema_extra={"example": 340.5})
    is_online: int = Field(..., ge=0, le=1, json_schema_extra={"example": 1})


class PredictionResponse(BaseModel):
    transaction_id: int
    fraud_probability: float
    is_fraud_flag: bool
    decision_threshold: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    pr_auc: float
