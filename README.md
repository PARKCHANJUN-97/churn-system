# 신용카드 고객 이탈 예측 의사결정 지원 시스템

본 프로젝트는 신용카드 고객의 이탈 여부를 예측하고,  
예측 결과를 바탕으로 운영자가 합리적인 대응 의사결정을 내릴 수 있도록 돕는  
의사결정 지원 시스템(DSS)입니다.

---

## 1. 프로젝트 목적
고객 이탈 여부를 단순히 예측하는 수준을 넘어,  
모델별 예측 결과를 비교하고 임계값(threshold)을 기준으로  
이탈 위험을 판단하여 실제 운영에 활용 가능한 대응 전략을 제시하는 것을 목표로 합니다.

---

## 2. 사용 데이터
- 데이터셋: `BankChurners.csv`
- 타깃 변수: `Attrition_Flag`
  - 이탈 고객: 1
  - 유지 고객: 0

---

## 3. 사용 모델
- Logistic Regression
- Random Forest
- XGBoost

---

## 4. 시스템 주요 기능
- 핵심 고객 변수 기반 이탈 확률 예측
- 동일 입력 기준 모델별 예측 결과 비교
- 임계값(threshold)을 활용한 이탈 위험 판정
- 위험도(HIGH / MID / LOW)에 따른 대응 전략(플레이북) 제공
- Streamlit 기반 인터페이스

---

## 5. 실행 방법 (로컬)
```bash
pip install -r requirements.txt
streamlit run churn_system/app.py


