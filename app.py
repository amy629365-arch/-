import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="犯罪预测助手", page_icon="🔮")

st.title("🔮 犯罪预测助手")

# 数据
df = pd.DataFrame({
    '年份': [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    '刑事案件数': [1095, 2373, 12643, 50167, 51980, 60556, 52611, 51458, 53406, 48744, 47992],
    '经济': [13593, 15154, 16796, 18363, 19998, 21863, 23984, 26415, 28103, 30904, 32745],
})

# 计算相关性
corr = df['经济'].corr(df['刑事案件数'])
st.metric("相关系数", f"{corr:.3f}")

# 训练模型
model = LinearRegression()
model.fit(df[['经济']], df['刑事案件数'])

# 输入框
col1, col2 = st.columns(2)
with col1:
    year = st.number_input("请输入要预测的年份", min_value=2023, max_value=2035, value=2023, step=1)
with col2:
    economy = st.number_input("请输入该年份的经济数据（亿元）", value=35364.0)

if st.button("开始预测"):
    pred = model.predict([[economy]])[0]
    
    year_int = int(year)
    
    st.success(f"预测{year_int}年刑事案件数：{int(pred):,} 起")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 历史数据
    ax.plot(df['年份'], df['刑事案件数'], 'b-o', linewidth=2, markersize=8, label='历史数据')
    
    # 预测点（红点）
    ax.scatter(year_int, pred, color='red', s=150, zorder=5, label=f'预测{year_int}年: {int(pred):,}', marker='o')
    
    # 虚线
    last_year = df['年份'].iloc[-1]
    last_crime = df['刑事案件数'].iloc[-1]
    ax.plot([last_year, year_int], [last_crime, pred], 'r--', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('刑事案件数', fontsize=12)
    ax.set_title('刑事案件变化趋势与预测', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(list(range(2012, year_int + 2, 2)))
    
    st.pyplot(fig)
    
    st.caption(f"📊 基于{year_int}年经济数据 {economy:,.0f} 亿元，预测刑事案件数为 {int(pred):,} 起")
