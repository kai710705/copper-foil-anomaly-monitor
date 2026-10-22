"""Streamlit 主程式：製程資料視覺化與異常預測。"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "copper_foil_cleaned.csv"
MODEL_PATH = BASE_DIR / "random_forest_anomaly_model.joblib"

st.set_page_config(page_title="銅箔製程異常監測", page_icon=":material/monitoring:", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    """載入清理後資料；資料不會在每次互動時重複讀取。"""
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig", parse_dates=["Timestamp"])


@st.cache_resource
def load_model() -> dict:
    """載入包含前處理與 Random Forest 的 joblib artifact。"""
    return joblib.load(MODEL_PATH)


def show_overview(data: pd.DataFrame) -> None:
    """第 1 頁：顯示主要製程特徵與品質標籤圖表。"""
    st.title("銅箔製程資料視覺化")
    st.caption("資料來源：copper_foil_cleaned.csv")

    with st.container(horizontal=True):
        st.metric("資料筆數", f"{len(data):,}", border=True)
        st.metric("平均厚度 (μm)", f"{data['Thick_Mean_um'].mean():.2f}", border=True)
        st.metric("異常筆數", f"{int(data['Is_Anomaly'].sum()):,}", border=True)

    chart_data = data.sort_values("Timestamp").set_index("Timestamp")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("厚度趨勢")
            st.line_chart(
                chart_data[["Thick_Left_um", "Thick_Center_um", "Thick_Right_um"]],
                alt="左側、中心與右側厚度隨時間變化",
            )
    with right:
        with st.container(border=True):
            st.subheader("製程參數趨勢")
            st.line_chart(
                chart_data[["Bath_Temp_C", "Cu_Conc_gL", "Line_Speed_mpm"]],
                alt="浴液溫度、銅濃度與線速隨時間變化",
            )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("異常與品質標籤")
            labels = data[["Is_Anomaly", "Is_Out_of_Spec", "Is_Nonuniform"]].sum()
            st.bar_chart(labels.rename("筆數"), alt="異常、超規與不均勻標籤筆數")
    with right:
        with st.container(border=True):
            st.subheader("製程參數與平均厚度")
            st.scatter_chart(
                data,
                x="Bath_Temp_C",
                y="Thick_Mean_um",
                color="Is_Anomaly",
                alt="浴液溫度與平均厚度的關係，依異常標籤著色",
            )

    with st.container(border=True):
        st.subheader("資料預覽")
        st.dataframe(data.head(20), hide_index=True, alt="清理後資料前 20 筆")


def show_prediction(data: pd.DataFrame) -> None:
    """第 2 頁：建立表單，讓使用者輸入製程資料並呼叫模型。"""
    st.title("製程異常預測")
    st.caption("輸入一筆製程資料後，按下按鈕取得 Random Forest 預測結果。")

    if not MODEL_PATH.exists():
        st.error(
            "找不到模型檔 random_forest_anomaly_model.joblib。請先執行 "
            "03_random_forest_training.ipynb。",
            icon=":material/error:",
        )
        return

    artifact = load_model()
    model = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]
    defaults = data.median(numeric_only=True)

    with st.form("prediction_form", border=True):
        st.subheader("輸入製程參數")
        row = {}
        c1, c2, c3 = st.columns(3)
        with c1:
            row["Line_Speed_mpm"] = st.number_input("線速 (mpm)", value=float(defaults.get("Line_Speed_mpm", 3.2)), format="%.3f")
            row["Current_ASD"] = st.number_input("電流 (ASD)", value=float(defaults.get("Current_ASD", 58.0)), format="%.3f")
            row["Bath_Temp_C"] = st.number_input("浴液溫度 (°C)", value=float(defaults.get("Bath_Temp_C", 50.0)), format="%.3f")
            row["Cu_Conc_gL"] = st.number_input("銅濃度 (g/L)", value=float(defaults.get("Cu_Conc_gL", 88.0)), format="%.3f")
        with c2:
            row["Additive_ppm"] = st.number_input("添加劑 (ppm)", value=float(defaults.get("Additive_ppm", 3.5)), format="%.3f")
            row["Jet_Pressure_bar"] = st.number_input("噴流壓力 (bar)", value=float(defaults.get("Jet_Pressure_bar", 2.5)), format="%.3f")
            row["Edge_Current_Bias"] = st.number_input("邊緣電流偏差", value=float(defaults.get("Edge_Current_Bias", 0.0)), format="%.3f")
            row["Shift"] = st.selectbox("班別", sorted(data["Shift"].dropna().unique().tolist()))
        with c3:
            row["Thick_Left_um"] = st.number_input("左側厚度 (μm)", value=float(defaults.get("Thick_Left_um", 18.0)), format="%.3f")
            row["Thick_Center_um"] = st.number_input("中心厚度 (μm)", value=float(defaults.get("Thick_Center_um", 18.0)), format="%.3f")
            row["Thick_Right_um"] = st.number_input("右側厚度 (μm)", value=float(defaults.get("Thick_Right_um", 18.0)), format="%.3f")
            row["Thick_Mean_um"] = st.number_input("平均厚度 (μm)", value=float(defaults.get("Thick_Mean_um", 18.0)), format="%.3f")
            row["Thick_Range_um"] = st.number_input("厚度範圍 (μm)", value=float(defaults.get("Thick_Range_um", 0.2)), format="%.3f")
        submitted = st.form_submit_button("開始異常預測", type="primary", width="stretch")

    if submitted:
        # 補上訓練時由 Timestamp 產生的特徵，並依 artifact 欄位順序建立輸入資料。
        row.update({"timestamp_hour": 12, "timestamp_dayofweek": 2, "timestamp_month": 1})
        input_df = pd.DataFrame([row]).reindex(columns=feature_columns)
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0, 1])
        if prediction == 1:
            st.error(f"預測結果：異常（異常機率 {probability:.1%}）", icon=":material/warning:")
        else:
            st.success(f"預測結果：正常（異常機率 {probability:.1%}）", icon=":material/check_circle:")
        st.progress(probability, text="異常機率")
        st.dataframe(input_df, hide_index=True, alt="送入模型的預測資料")


if not DATA_PATH.exists():
    st.error(f"找不到資料檔：{DATA_PATH}", icon=":material/error:")
    st.stop()

data = load_data()
with st.sidebar:
    st.header("功能分頁")
    page = st.radio("選擇功能", ["資料視覺化", "異常預測"], index=0)
    st.divider()
    st.caption("Random Forest 異常分類系統")

if page == "資料視覺化":
    show_overview(data)
else:
    show_prediction(data)
