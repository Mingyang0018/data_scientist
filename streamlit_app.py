import os
import streamlit as st
import pandas as pd
import ydata_profiling
from streamlit_ydata_profiling import st_profile_report
import datetime
# 设置页面布局
st.set_page_config(layout="wide")

# 设置标题
st.title('生成数据分析报告')
# 设置上传文件组件
st.subheader('上传CSV或Excel')
uploaded_file = st.file_uploader(label='Hello', label_visibility='collapsed', type=['csv', 'xls', 'xlsx'], accept_multiple_files=False)

# 根据是否选择了文件显示消息
try:
    # 获取文件后缀
    ext = os.path.splitext(uploaded_file.name)[1] if uploaded_file else ''
    if ext == '.csv':
        df = pd.read_csv(uploaded_file)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(uploaded_file)
    else:
        st.warning('请选择CSV或Excel文件!')
        df = None
    st.subheader('数据表')
    st.write(df)
    st.subheader('数据分析报告')
    report = ydata_profiling.ProfileReport(df, title='Profile Report', explorative=True)
    # 添加下载按钮
    report_html = report.to_html()
    now = datetime.datetime.now()
    now_str = now.strftime('%Y%m%d%H%M%S')
    file_name = f'datareport_{now_str}.html'
    btn = st.download_button(label='下载', data=report_html, file_name=file_name, mime='text/html')
    st_profile_report(report, navbar=True)
    st.balloons()
        
except:
    st.warning('请先上传数据文件!')
