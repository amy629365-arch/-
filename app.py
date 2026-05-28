import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="犯罪预测助手", page_icon="🔮")

st.title("🔮 犯罪预测助手")

df = pd.DataFrame({
    '年份': [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    '刑事案件数': [1095, 2373, 12643, 50167, 51980, 60556, 52611, 51458, 53406, 48744, 47992],
    '经济': [13593, 15154, 16796, 18363, 19998, 21863, 23984, 26415, 28103, 30904, 32745],
})

corr = df['经济'].corr(df['刑事案件数'])
st.metric("相关系数", f"{corr:.3f}")

model = LinearRegression()
model.fit(df[['经济']], df['刑事案件数'])

economy = st.number_input("请输入2023年经济数据（亿元）", value=35364)

if st.button("开始预测"):
    pred = model.predict([[economy]])[0]
    st.success(f"预测2023年刑事案件数：{int(pred):,} 起")
    
    fig, ax = plt.subplots()
    ax.plot(df['年份'], df['刑事案件数'], 'b-o', label='历史数据')
    ax.scatter(2023, pred, color='r', s=100, label=f'预测: {int(pred)}')
    ax.legend()
    ax.set_xlabel("年份")
    ax.set_ylabel("刑事案件数")
    st.pyplot(fig)