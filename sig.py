import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import statsmodels.stats.multicomp as mc
import numpy as np

# 上传数据
st.title("显著性分析仪表盘")
st.write("请上传CSV格式的文件，包含‘group’和‘value’列。")

uploaded_file = st.file_uploader("选择文件", type="csv")
if uploaded_file:
    # 读取数据
    data = pd.read_csv(uploaded_file)
    st.write("数据预览：", data.head())

    # 检查数据格式
    if 'group' in data.columns and 'value' in data.columns:
        
        # 计算分组的均值和SD
        summary_df = data.groupby('group')['value'].agg(['mean', 'std']).reset_index()
        
        # 单因素方差分析（ANOVA）
        groups = data.groupby('group')['value'].apply(list)
        anova_result = f_oneway(*groups)
        st.write("ANOVA结果：", anova_result)
        
        # 多重比较测试 (Tukey HSD)
        comp = mc.MultiComparison(data['value'], data['group'])
        tukey_result = comp.tukeyhsd()
        
        # 提取显著性结果，并生成字母标记
        summary_df['sig'] = tukey_result.groupsunique
        tukey_summary = tukey_result.summary()
        letters = {}  # 用于保存每组的显著性标记字母
        for i, row in enumerate(tukey_summary.data[1:]):
            group1, group2, _, _, reject = row[:5]
            if reject:
                # 如果两组有显著性差异，给不同的字母
                if group1 not in letters:
                    letters[group1] = chr(97 + len(letters))  # 分配新的字母
                if group2 not in letters:
                    letters[group2] = chr(97 + len(letters))
            else:
                # 如果无显著性差异，分配相同的字母
                letters[group2] = letters.get(group1, chr(97 + len(letters)))
        
        # 将字母标记加入summary_df
        summary_df['letter'] = summary_df['group'].map(letters)
        
        # 可视化柱状图
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 自定义颜色和标题
        sns.barplot(data=data, x='group', y='value', ci="sd", palette="muted", ax=ax)
        ax.set_title("Significant analysis", fontsize=14, weight='bold')
        ax.set_xlabel("Group", fontsize=12)
        ax.set_ylabel("Mean ± SD", fontsize=12)
        
        # 添加显著性标记字母在误差条上方
        for i, (group, row) in enumerate(summary_df.iterrows()):
            mean = row['mean']
            sd = row['std']
            letter = row['letter']
            ax.errorbar(i, mean, yerr=sd, fmt='none', c='black', capsize=5)  # 绘制误差条
            ax.text(i, mean + sd + 0.1, letter, ha='center', va='bottom', fontsize=12, color='black')
        
        st.pyplot(fig)
        
        # 显示分组统计结果表格
        summary_df = summary_df[['group', 'mean', 'std', 'letter']]
        summary_df.columns = ['Group', 'Mean', 'SD', 'significant letter']
        st.write("分组统计结果：")
        st.write(summary_df)

    else:
        st.write("上传的数据格式不正确，请确保包含‘group’和‘value’列。")
