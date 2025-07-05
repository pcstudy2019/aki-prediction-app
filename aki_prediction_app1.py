import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from joblib import load
import matplotlib

# 设置中文字体支持
matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 页面配置
st.set_page_config(
    page_title="AKI预测计算器",
    page_icon=":hospital:",
    layout="wide"
)

# 标题和介绍
st.title("急性肾损伤(AKI)预测计算器")
st.markdown("""
本应用使用随机森林模型预测病人发生急性肾损伤的风险，并通过SHAP值解释模型决策。
输入病人特征后，点击"预测"按钮获取结果。
""")

# 加载模型
@st.cache_resource
def load_model():
    try:
        model = load('best_random_forest_model.joblib')
        return model
    except FileNotFoundError:
        st.error("未找到模型文件！请确保best_random_forest_model.joblib在正确的路径下。")
        return None

# 特征名称（严格按照训练时的顺序和大小写）
FEATURE_NAMES = ['Hyperkalemia', 'Anemia', 'Hyponatremia', 'AGE', 'Hypertension', 
                 'IHD', 'Diabetesmellitus', 'Hypokalemia']

# 特征中文显示名称（用于界面）
FEATURE_DISPLAY_NAMES = {
    'Hyperkalemia': '高钾血症',
    'Anemia': '贫血',
    'Hyponatremia': '低钠血症',
    'AGE': '年龄',
    'Hypertension': '高血压',
    'IHD': '缺血性心脏病',
    'Diabetesmellitus': '糖尿病',
    'Hypokalemia': '低钾血症'
}

# 创建输入表单
def create_input_form():
    with st.form("patient_input_form"):
        st.subheader("病人特征输入")
        
        # 创建输入字段（按训练时的顺序排列）
        input_data = {}
        
        # 分类特征（确保与训练时的编码一致：0=否，1=是）
        cols1 = st.columns(4)
        with cols1[0]:
            input_data['Hyperkalemia'] = st.selectbox(
                FEATURE_DISPLAY_NAMES['Hyperkalemia'], 
                [0, 1], 
                format_func=lambda x: "否" if x == 0 else "是"
            )
        with cols1[1]:
            input_data['Anemia'] = st.selectbox(
                FEATURE_DISPLAY_NAMES['Anemia'], 
                [0, 1], 
                format_func=lambda x: "否" if x == 0 else "是"
            )
        with cols1[2]:
            input_data['Hyponatremia'] = st.selectbox(
                FEATURE_DISPLAY_NAMES['Hyponatremia'], 
                [0, 1], 
                format_func=lambda x: "否" if x == 0 else "是"
            )
        with cols1[3]:
            input_data['Hypertension'] = st.selectbox(
                FEATURE_DISPLAY_NAMES['Hypertension'], 
                [0, 1], 
                format_func=lambda x: "否" if x == 0 else "是"
            )
        
        cols2 = st.columns(4)
        with cols2[0]:
            input_data['IHD'] = st.selectbox(
                FEATURE_DISPLAY_NAMES['IHD'], 
                [0, 1], 
                format_func=lambda x: "否" if x == 0 else "是"
            )
        with cols2[1]:
            input_data['Diabetesmellitus'] = st.selectbox(
                FEATURE_DISPLAY_NAMES['Diabetesmellitus'], 
                [0, 1], 
                format_func=lambda x: "否" if x == 0 else "是"
            )
        with cols2[2]:
            input_data['Hypokalemia'] = st.selectbox(
                FEATURE_DISPLAY_NAMES['Hypokalemia'], 
                [0, 1], 
                format_func=lambda x: "否" if x == 0 else "是"
            )
        with cols2[3]:
            input_data['AGE'] = st.slider(
                FEATURE_DISPLAY_NAMES['AGE'], 
                min_value=18, 
                max_value=100, 
                value=65
            )
        
        # 提交按钮
        submitted = st.form_submit_button("预测")
        
        if submitted:
            # 创建DataFrame时严格按照FEATURE_NAMES的顺序排列列
            patient_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
            return patient_df
    return None

# 计算SHAP值并可视化（删除force plot，保留其他两种图）
def explain_prediction(model, patient_data, X_train):
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    
    # 计算SHAP值
    shap_values = explainer.shap_values(patient_data)
    
    # 创建特征重要性条形图（保留）
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values[1], 
        patient_data,
        feature_names=FEATURE_NAMES,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    
    # 创建特征影响方向图（保留）
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values[1], 
        patient_data,
        feature_names=FEATURE_NAMES,
        show=False
    )
    plt.tight_layout()
    
    return fig2, fig3, shap_values[1][0]

# 主函数
def main():
    # 加载模型
    model = load_model()
    if model is None:
        return
    
    # 创建输入表单
    patient_data = create_input_form()
    
    if patient_data is not None:
        # 显示输入数据（使用中文名称）
        st.subheader("输入的病人数据")
        display_df = patient_data.copy()
        display_df.columns = [FEATURE_DISPLAY_NAMES.get(col, col) for col in display_df.columns]
        st.dataframe(display_df)
        
        # 预测
        prediction = model.predict(patient_data)
        probability = model.predict_proba(patient_data)[:, 1][0]
        
        # 显示预测结果
        st.subheader("预测结果")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "AKI风险概率", 
                f"{probability:.2%}",
                delta="高风险" if probability > 0.02244498 else "低风险",
                delta_color="inverse" if probability > 0.02244498 else "normal"
            )
        with col2:
            st.write(f"预测类别: {'AKI' if prediction[0] == 1 else '非AKI'}")
        
        # 为SHAP值计算准备训练数据摘要
        st.subheader("模型决策解释")
        with st.spinner("正在计算SHAP值..."):
            # 由于没有实际训练数据，我们用一个模拟数据代替
            # 实际应用中应该加载训练数据
            X_train = pd.DataFrame(
                np.random.randn(100, len(FEATURE_NAMES)),
                columns=FEATURE_NAMES
            )
            
            # 计算并显示SHAP解释（删除force plot调用）
            bar_plot, direction_plot, shap_values = explain_prediction(model, patient_data, X_train)
            
            # 显示特征重要性条形图 - 展示各特征的相对重要性（保留）
            st.subheader("特征重要性")
            st.pyplot(bar_plot)
            
            # 显示特征影响方向图 - 展示特征值与SHAP值的关系（保留）
            st.subheader("特征影响方向")
            st.pyplot(direction_plot)
            
            # 显示SHAP值表格
            st.subheader("特征贡献值 (SHAP值)")
            shap_df = pd.DataFrame({
                '特征': [FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_NAMES],
                '英文名称': FEATURE_NAMES,  # 显示原始英文名称用于调试
                'SHAP值': shap_values,
                '影响方向': ['增加风险' if x > 0 else '降低风险' for x in shap_values],
                '影响程度': ['高' if abs(x) > np.mean(np.abs(shap_values)) else '低' for x in shap_values]
            })
            shap_df = shap_df.sort_values('SHAP值', ascending=False)
            st.dataframe(shap_df)

if __name__ == "__main__":
    main()
