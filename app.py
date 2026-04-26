#!/usr/bin/env python
# coding: utf-8

import os
import sys
import io
import base64
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


# =========================
# 页面基础配置
# =========================
st.set_page_config(
    page_title="基于GBDT的预测",
    layout="wide"
)

st.title("基于GBDT的预测")


# =========================
# absence / presence 映射
# =========================
OPTION_LIST = ["absence", "presence"]

OPTION_MAP = {
    "absence": 0,
    "presence": 1
}


# =========================
# 加载模型
# =========================
@st.cache_resource
def load_model():
    """
    正常加载 GBDT.pkl。
    如果模型反序列化时出现 No module named '_loss'，
    自动做 sklearn._loss._loss 兼容映射。
    """
    try:
        model = joblib.load("GBDT.pkl")
        return model

    except ModuleNotFoundError as e:
        if str(e) == "No module named '_loss'" or getattr(e, "name", None) == "_loss":
            import sklearn._loss._loss as cy_loss
            sys.modules["_loss"] = cy_loss
            model = joblib.load("GBDT.pkl")
            return model
        else:
            raise e


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


try:
    clf = load_model()
except Exception as e:
    st.error("模型加载失败，请检查 GBDT.pkl 是否在 app.py 同级目录，以及 sklearn 版本是否匹配。")
    st.exception(e)
    st.stop()


# =========================
# 获取特征名
# =========================
if not hasattr(clf, "feature_names_in_"):
    st.error("当前模型没有 feature_names_in_ 属性，无法自动生成输入项。")
    st.stop()

feature_names = list(clf.feature_names_in_)

if len(feature_names) != 23:
    st.warning(f"当前模型特征数量为 {len(feature_names)} 个，不是 23 个，请确认模型是否正确。")


# =========================
# 初始化输入状态
# =========================
def init_input_state():
    """
    初始化每个变量的下拉框状态。
    默认全部为 absence。
    """
    for i, name in enumerate(feature_names):
        key = f"input_{i}"

        if key not in st.session_state:
            st.session_state[key] = "absence"

        # 防止之前 number_input 留下 0.0 / 1.0 等旧状态导致 selectbox 报错
        if st.session_state[key] not in OPTION_LIST:
            st.session_state[key] = "absence"


def reset_inputs():
    """
    重置所有输入为 absence。
    """
    for i, name in enumerate(feature_names):
        st.session_state[f"input_{i}"] = "absence"


init_input_state()


# =========================
# 输入区域：两列
# 第一列 12 个，第二列 11 个
# =========================
left_features = feature_names[:12]
right_features = feature_names[12:]

col1, col2 = st.columns(2)

with col1:
    for i, name in enumerate(left_features):
        st.selectbox(
            label=name,
            options=OPTION_LIST,
            key=f"input_{i}"
        )

with col2:
    for j, name in enumerate(right_features):
        i = j + 12
        st.selectbox(
            label=name,
            options=OPTION_LIST,
            key=f"input_{i}"
        )


# =========================
# 按钮区域
# =========================
st.markdown("---")

btn_col1, btn_col2 = st.columns([1, 1])

with btn_col1:
    predict_btn = st.button(
        "模型预测",
        type="primary",
        use_container_width=True
    )

with btn_col2:
    st.button(
        "模型输入重置",
        on_click=reset_inputs,
        use_container_width=True
    )


# =========================
# 预测和 SHAP 解释
# =========================
if predict_btn:

    # 原始下拉选择：absence / presence
    input_label_dict = {
        name: st.session_state[f"input_{i}"]
        for i, name in enumerate(feature_names)
    }

    # 传入模型的数值：absence -> 0, presence -> 1
    input_dict = {
        name: OPTION_MAP[st.session_state[f"input_{i}"]]
        for i, name in enumerate(feature_names)
    }

    X_label = pd.DataFrame([input_label_dict], columns=feature_names)
    X = pd.DataFrame([input_dict], columns=feature_names)

    st.subheader("当前输入数据")

    with st.expander("查看原始输入：absence / presence", expanded=False):
        st.dataframe(X_label, use_container_width=True)

    with st.expander("查看传入模型的数值：0 / 1", expanded=True):
        st.dataframe(X, use_container_width=True)

    try:
        pred_class = clf.predict(X)[0]

        if hasattr(clf, "predict_proba"):
            pred_proba_all = clf.predict_proba(X)
            classes = list(clf.classes_)

            # 默认解释和显示 class=1 的概率；
            # 如果模型类别里没有 1，则默认取最后一个类别。
            if 1 in classes:
                target_class = 1
                target_index = classes.index(1)
            else:
                target_class = classes[-1]
                target_index = len(classes) - 1

            pred_proba = pred_proba_all[0][target_index]

            # 预测类别对应的 index，备用
            if pred_class in classes:
                pred_index = classes.index(pred_class)
            else:
                pred_index = int(np.argmax(pred_proba_all[0]))

        else:
            pred_proba = None
            target_index = 0
            pred_index = 0
            target_class = None

        st.subheader("模型预测结果")

        st.success(f"the Predict class is {pred_class}")

        if pred_proba is not None:
            st.success(f"the Predict proba is {pred_proba * 100:.2f}%")
        else:
            st.info("当前模型不支持 predict_proba，无法输出预测概率。")

    except Exception as e:
        st.error("模型预测失败。")
        st.exception(e)
        st.stop()


    # =========================
    # SHAP 力图
    # =========================
    st.subheader("SHAP 解释力图")

    try:
        explainer = load_explainer(clf)
        shap_values = explainer(X)

        values = shap_values.values
        base_values = shap_values.base_values

        # 兼容二分类 / 多分类 / 不同 SHAP 版本
        if values.ndim == 3:
            # shape: [样本数, 特征数, 类别数]
            shap_value_single = values[0, :, target_index]

            if np.ndim(base_values) == 2:
                expected_value = base_values[0, target_index]
            elif np.ndim(base_values) == 1:
                expected_value = base_values[target_index]
            else:
                expected_value = base_values

        elif values.ndim == 2:
            # shape: [样本数, 特征数]
            shap_value_single = values[0]

            if np.ndim(base_values) == 1:
                expected_value = base_values[0]
            else:
                expected_value = base_values

        else:
            st.error(f"无法识别的 SHAP values 维度：{values.shape}")
            st.stop()

        # 保证 expected_value 是标量
        expected_value = float(np.ravel(expected_value)[0])

        # 防止旧图残留
        plt.close("all")

        shap.force_plot(
            expected_value,
            np.round(shap_value_single, 3),
            np.round(X.iloc[0], 3),
            feature_names=feature_names,
            figsize=(45, 4),
            matplotlib=True,
            show=False,
            text_rotation=0,
            contribution_threshold=0.03
        )

        fig = plt.gcf()

        # 保存为 SVG，网页显示更清晰
        svg_buffer = io.StringIO()
        fig.savefig(
            svg_buffer,
            format="svg",
            bbox_inches="tight"
        )
        plt.close(fig)

        svg_data = svg_buffer.getvalue()
        b64 = base64.b64encode(svg_data.encode("utf-8")).decode("utf-8")

        components.html(
            f"""
            <div style="width:100%; overflow-x:auto; border:1px solid #e6e6e6; padding:10px;">
                <img src="data:image/svg+xml;base64,{b64}" style="width:1800px;">
            </div>
            """,
            height=360,
            scrolling=False
        )

    except Exception as e:
        st.error("SHAP 力图生成失败。")
        st.exception(e)