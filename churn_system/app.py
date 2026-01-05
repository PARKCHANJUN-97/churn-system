import json, joblib
import pandas as pd
import streamlit as st

from utils.features import add_derived_features, build_X, build_single_input_row
from utils.decision import decide

MODEL_FILES = {
    "XGBoost": "models/xgboost.joblib",
    "Random Forest": "models/random_forest.joblib",
    "Logistic Regression": "models/logistic_regression.joblib",
}
METRICS_PATH = "metrics.json"
DATA_PATH = "data/BankChurners.csv"


# --------------------------
# 공통 유틸리티 함수
# 모델/데이터 처리에 공통으로 사용
# --------------------------
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def align_columns(X_onehot: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    return X_onehot.reindex(columns=feature_columns, fill_value=0)

def predict_proba(model_label: str, artifact: dict, X_aligned: pd.DataFrame) -> float:
    model = artifact["model"]
    if model_label == "Logistic Regression":
        scaler = artifact["scaler"]
        X_scaled = scaler.transform(X_aligned)
        return float(model.predict_proba(X_scaled)[:, 1][0])
    return float(model.predict_proba(X_aligned)[:, 1][0])

def run_inference_for_model(model_label: str, user_input: dict, metrics: dict) -> dict:
    """
    동일 입력값으로 model_label 1개에 대해
    이탈확률/threshold/판정/위험도를 반환 (비교표용)
    """
    artifact = joblib.load(MODEL_FILES[model_label])

    row = build_single_input_row(user_input)
    row = add_derived_features(row)
    X_onehot = build_X(row)
    X_aligned = align_columns(X_onehot, artifact["feature_columns"])

    prob = predict_proba(model_label, artifact, X_aligned)

    key_to_label_local = {
        "xgboost": "XGBoost",
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression"
    }
    model_key = [k for k, v in key_to_label_local.items() if v == model_label][0]
    threshold = float(metrics["models"][model_key]["threshold"])

    decision = decide(
        prob,
        threshold,
        x={
            "Monthly_Trans_Ct": float(row["Monthly_Trans_Ct"].iloc[0]),
            "Avg_Utilization_Ratio": float(row["Avg_Utilization_Ratio"].iloc[0]),
            "Total_Revolving_Bal": float(row["Total_Revolving_Bal"].iloc[0]),
            "Credit_Limit": float(row["Credit_Limit"].iloc[0]),
            "Total_Amt_Chng_Q4_Q1": float(row["Total_Amt_Chng_Q4_Q1"].iloc[0]),
            "Total_Ct_Chng_Q4_Q1": float(row["Total_Ct_Chng_Q4_Q1"].iloc[0]),
        }
    )

    return {
        "모델": model_label,
        "이탈확률": float(decision.probability),
        "threshold": float(decision.threshold),
        "판정": "이탈 위험(1)" if int(decision.label) == 1 else "유지 가능(0)",
        "위험도": str(decision.risk_level),
    }


@st.cache_data(show_spinner=False)
def compute_defaults_from_data(csv_path: str = DATA_PATH) -> dict:
    """
    UI에 노출하지 않는 변수(고급 입력)를 자동 보정하기 위한 기준값 계산
    - 숫자형: 중앙값
    - 범주형: 최빈값
    """
    df = pd.read_csv(csv_path)
    defaults = {
        "Gender": df["Gender"].mode().iloc[0] if "Gender" in df.columns else "M",
        "Total_Relationship_Count": float(df["Total_Relationship_Count"].median()),
        "Total_Ct_Chng_Q4_Q1": float(df["Total_Ct_Chng_Q4_Q1"].median()),
        "Total_Amt_Chng_Q4_Q1": float(df["Total_Amt_Chng_Q4_Q1"].median()),
    }
    return defaults


# --------------------------
# 사용자 인터페이스(UI) 구성
# --------------------------
st.set_page_config(page_title="이탈 예측 시스템", layout="centered")
st.title("신용카드 고객 이탈 예측 의사결정 지원 시스템")
st.caption("입력 → 예측(확률) → 임계값(threshold) 판정 → 대응 전략(플레이북) 추천")

metrics = load_metrics()
best_key = max(metrics["models"].keys(), key=lambda k: metrics["models"][k]["f1"])
key_to_label = {"xgboost": "XGBoost", "random_forest": "Random Forest", "logistic_regression": "Logistic Regression"}
default_label = key_to_label.get(best_key, "XGBoost")

model_label = st.selectbox(
    "모델 선택",
    list(MODEL_FILES.keys()),
    index=list(MODEL_FILES.keys()).index(default_label)
)

st.subheader("고객 정보 입력 (의사결정 핵심 입력 변수 7개)")
st.caption("고급 입력을 열지 않으면 나머지 변수는 학습 데이터의 기준값(중앙값/최빈값)으로 자동 보정됩니다.")

defaults = compute_defaults_from_data()

# 핵심 입력 변수 7개
# - 사용자에게 직접 입력받는 변수
c1, c2 = st.columns(2)
with c1:
    Customer_Age = st.number_input("Customer_Age (고객 연령)", 18, 100, 45)
    Credit_Limit = st.number_input("Credit_Limit (신용 한도)", value=8000.0, step=500.0)
    Total_Revolving_Bal = st.number_input("Total_Revolving_Bal (리볼빙 잔액)", value=1200.0, step=50.0)
    Months_on_book = st.number_input("Months_on_book (거래 기간/개월)", value=36, step=1)

with c2:
    Total_Trans_Ct = st.number_input("Total_Trans_Ct (총 거래 횟수)", value=45, step=1)
    Total_Trans_Amt = st.number_input("Total_Trans_Amt (총 거래 금액)", value=3500.0, step=100.0)
    Avg_Utilization_Ratio = st.number_input("Avg_Utilization_Ratio (평균 이용률)", 0.0, 1.0, 0.25, 0.01)

# 사용자 입력 데이터 구성
# - UI에서 입력받은 의사결정 핵심 변수 7개
# - 나머지 변수는 학습 데이터 기준값으로 자동 보정

user_input = {
    # 핵심 입력 변수
    "Customer_Age": float(Customer_Age),
    "Credit_Limit": float(Credit_Limit),
    "Total_Trans_Ct": int(Total_Trans_Ct),
    "Total_Trans_Amt": float(Total_Trans_Amt),
    "Avg_Utilization_Ratio": float(Avg_Utilization_Ratio),
    "Total_Revolving_Bal": float(Total_Revolving_Bal),
    "Months_on_book": int(Months_on_book),

# 고급 변수 (자동 보정)
# - 학습 데이터 기준값으로 자동 설정
    "Gender": str(defaults["Gender"]),
    "Total_Relationship_Count": float(defaults["Total_Relationship_Count"]),
    "Total_Ct_Chng_Q4_Q1": float(defaults["Total_Ct_Chng_Q4_Q1"]),
    "Total_Amt_Chng_Q4_Q1": float(defaults["Total_Amt_Chng_Q4_Q1"]),
}

# --- 고급 입력 영역 (선택) ---
with st.expander("고급 입력(선택) — 필요할 때만 조정"):
    st.caption("기본값은 학습 데이터의 기준값입니다. 필요 시 사용자가 직접 조정할 수 있습니다.")
    user_input["Gender"] = st.selectbox(
        "Gender",
        ["M", "F"],
        index=0 if user_input["Gender"] == "M" else 1,
        key="gender_select_v2"
    )
    user_input["Total_Relationship_Count"] = st.number_input(
        "Total_Relationship_Count",
        min_value=1.0, max_value=10.0,
        value=float(user_input["Total_Relationship_Count"]),
        step=1.0
    )
    user_input["Total_Ct_Chng_Q4_Q1"] = st.number_input(
        "Total_Ct_Chng_Q4_Q1",
        value=float(user_input["Total_Ct_Chng_Q4_Q1"]),
        step=0.01
    )
    user_input["Total_Amt_Chng_Q4_Q1"] = st.number_input(
        "Total_Amt_Chng_Q4_Q1",
        value=float(user_input["Total_Amt_Chng_Q4_Q1"]),
        step=0.01
    )

# --------------------------
# 실행 버튼 영역
# - 단일 모델 예측
# - 모델 간 결과 비교
# --------------------------
b1, b2 = st.columns(2)
run_single = b1.button("이탈 확률 예측하기(선택 모델)")
run_compare = b2.button("모델 3개 비교하기")


# --------------------------
# 단일 모델 예측 실행 로직
# --------------------------
if run_single:
    artifact = joblib.load(MODEL_FILES[model_label])

    row = build_single_input_row(user_input)
    row = add_derived_features(row)
    X_onehot = build_X(row)
    X_aligned = align_columns(X_onehot, artifact["feature_columns"])

    prob = predict_proba(model_label, artifact, X_aligned)

    model_key = [k for k, v in key_to_label.items() if v == model_label][0]
    threshold = float(metrics["models"][model_key]["threshold"])

    decision = decide(
        prob,
        threshold,
        x={
            "Monthly_Trans_Ct": float(row["Monthly_Trans_Ct"].iloc[0]),
            "Avg_Utilization_Ratio": float(row["Avg_Utilization_Ratio"].iloc[0]),
            "Total_Revolving_Bal": float(row["Total_Revolving_Bal"].iloc[0]),
            "Credit_Limit": float(row["Credit_Limit"].iloc[0]),
            "Total_Amt_Chng_Q4_Q1": float(row["Total_Amt_Chng_Q4_Q1"].iloc[0]),
            "Total_Ct_Chng_Q4_Q1": float(row["Total_Ct_Chng_Q4_Q1"].iloc[0]),
        }
    )

    st.markdown("---")
    st.subheader("결과 (선택 모델)")

    a, b, c = st.columns(3)
    with a:
        st.metric("이탈 확률", f"{decision.probability:.3f}")
    with b:
        st.metric("판정 기준(threshold)", f"{decision.threshold:.3f}")
    with c:
        st.metric("위험도", decision.risk_level)
    
    with st.expander("위험도 등급 정의 (HIGH/MID/LOW)"):
        st.write("""
    - **LOW**: 현재 이탈 가능성이 낮아 일반 유지 정책으로 관리합니다.  
    - **MID**: 단기 이탈 가능성이 존재하여 선제적 리텐션 조치가 필요합니다.  
    - **HIGH**: 이탈 가능성이 매우 높아 즉각적인 개입이 요구됩니다.
    """)

    if decision.label == 1:
        st.error("판정: 이탈 위험(1) — 리텐션 대응이 필요합니다.")
    else:
        st.success("판정: 유지 가능(0) — 기본 유지 정책을 적용합니다.")

    st.subheader("판정 근거(설명)")
    st.caption("모델 예측과 함께, 운영자가 이해할 수 있도록 입력 지표 기반 근거를 함께 제공합니다.")
    if hasattr(decision, "score"):
        st.write(f"규칙 기반 위험 점수: **{decision.score}점**")
    if hasattr(decision, "reasons"):
        for r in decision.reasons:
            st.write(f"- {r}")

    st.subheader("추천 대응 전략(플레이북)")
    if hasattr(decision, "actions_today"):
        tab1, tab2, tab3 = st.tabs(["오늘(즉시)", "7일 이내", "30일 이내"])
        with tab1:
            for i, a_ in enumerate(decision.actions_today, 1):
                st.write(f"{i}. {a_}")
        with tab2:
            for i, a_ in enumerate(decision.actions_7d, 1):
                st.write(f"{i}. {a_}")
        with tab3:
            for i, a_ in enumerate(decision.actions_30d, 1):
                st.write(f"{i}. {a_}")
    else:
        for i, a_ in enumerate(getattr(decision, "actions", []), 1):
            st.write(f"{i}. {a_}")

    with st.expander("운영 정보(모델/입력 요약)"):
        st.write(f"- 사용 모델: **{model_label}**")
        st.write(f"- 적용 임계값(threshold): **{threshold:.3f}**")
        st.json(user_input)


# --------------------------
# 모델 3개 결과 비교 기능
# 동일 입력값 기준으로 예측 결과 차이 확인
# --------------------------
if run_compare:
    st.markdown("---")
    st.subheader("모델별 결과 비교 (동일 입력 기준)")
    st.caption("동일한 입력값에 대해 3개 모델의 예측 확률/판정/위험도가 어떻게 달라지는지 비교합니다.")

    results = []
    for lbl in MODEL_FILES.keys():
        results.append(run_inference_for_model(lbl, user_input, metrics))

    df = pd.DataFrame(results)
    df["이탈확률"] = df["이탈확률"].map(lambda x: round(x, 3))
    df["threshold"] = df["threshold"].map(lambda x: round(x, 3))
    df = df.sort_values(by="이탈확률", ascending=False).reset_index(drop=True)

    st.dataframe(df, use_container_width=True)
    st.caption("※ 본 기능은 모델 선택에 따른 의사결정 결과 차이를 확인하기 위한 비교 기능입니다.")
