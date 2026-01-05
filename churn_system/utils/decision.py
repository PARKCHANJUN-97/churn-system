from dataclasses import dataclass
from typing import Dict, List

@dataclass
class DecisionResult:
    probability: float
    threshold: float
    label: int
    risk_level: str
    actions: List[str]

def risk_bucket(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    if prob >= 0.40:
        return "MID"
    return "LOW"

def recommend_actions(risk: str, x: Dict[str, float]) -> List[str]:
    actions = []
    if risk == "LOW":
        return ["유지 가능성이 높습니다. 정기 혜택 안내/만족도 조사 수준으로 관리합니다."]

    actions.append("이탈 위험 고객입니다. 7일 이내 리텐션 캠페인을 우선 적용합니다.")

    if x.get("Monthly_Trans_Ct", 999) < 2.0:
        actions.append("거래 빈도 낮음 → 거래 유도 쿠폰/캐시백, 자동이체/구독 혜택 제안")
    if x.get("Avg_Utilization_Ratio", 999) < 0.10:
        actions.append("이용률 낮음 → 사용처 추천/혜택 리마인드로 사용 촉진")
    if x.get("Total_Revolving_Bal", 0) > 1500:
        actions.append("리볼빙 잔액 높음 → 상환 플랜/수수료 안내 + 상담 연결")
    if x.get("Credit_Limit", 0) > 10000 and risk == "HIGH":
        actions.append("고가치 고객 → VIP 전담 상담/특별 혜택 우선 적용")

    actions.append("HIGH면 즉시 상담 연결, MID면 자동 캠페인 후 반응 없으면 상담 확대")
    return actions

def decide(prob: float, threshold: float, x: Dict[str, float]) -> DecisionResult:
    label = 1 if prob >= threshold else 0
    risk = risk_bucket(prob)
    actions = recommend_actions(risk, x)
    return DecisionResult(probability=prob, threshold=threshold, label=label, risk_level=risk, actions=actions)
