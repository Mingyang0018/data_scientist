import os
import streamlit as st
import pandas as pd
import numpy as np
import ydata_profiling
from streamlit_ydata_profiling import st_profile_report
import datetime
from autogluon.tabular import TabularDataset, TabularPredictor

def main():
    # 设置页面布局
    st.set_page_config(layout="wide")

    # 设置标题
    st.title('数据科学家', help='这是一个数据科学家工具, 可以用于数据合并、分析和机器学习。')

    # 添加选项卡
    tab1, tab2, tab3 = st.tabs(["数据合并", "数据分析", "机器学习"])
    # 根据选项卡选择, 显示不同内容
    with tab1:
        # 创建选择文件组件
        st.subheader('选择多个CSV或Excel', divider='rainbow')
        uploaded_files = st.file_uploader("选择文件(CSV或Excel):", label_visibility='collapsed', type=['csv', 'xls', 'xlsx'], accept_multiple_files=True)
        
        df_concat = pd.DataFrame()
        columns = []
        list_file_ext = []
        # 读取文件
        for file in uploaded_files:
            # 获取文件后缀
            file_ext = os.path.splitext(file.name)[1]
            if file_ext == '.csv':
                df = pd.read_csv(file)
                list_file_ext.append(file_ext)
            elif file_ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file)
                list_file_ext.append(file_ext)
            else:
                st.warning('请选择CSV或Excel!', icon="⚠️")

            # 合并所有df
            df_concat = pd.concat([df_concat, df], ignore_index=True, join='outer', axis=0)
        
        st.subheader('合并结果', divider='rainbow', help='这是文件上下合并后的结果。')
        if df_concat.empty:
            st.write('None')
        else:
            # 获取列名
            columns = list(df_concat.columns)
            columns.insert(0, '全部')
            columns.insert(0, '无')
            # 创建下拉选择框
            column = st.selectbox('选择去重列:', columns)
            if column != '无':
                if column == '全部':
                    df_concat = df_concat.drop_duplicates(ignore_index=True)
                else:
                    # 按照去重列对df_concat进行去重
                    df_concat = df_concat.drop_duplicates(subset=[column], keep='first', ignore_index=True)
            
            now = datetime.datetime.now()
            now_str = now.strftime('%Y%m%d%H%M%S')
            file_name = f'data_concat_{now_str}.csv'
            btn = st.download_button(label='下载', 
                                    data=df_concat.to_csv(index=False).encode(), 
                                    file_name=file_name, 
                                    mime='text/csv')
            st.write(df_concat)
            st.balloons()
                    
    with tab2:
        # 设置上传文件组件
        st.subheader('选择一个CSV或Excel', divider='rainbow')
        uploaded_file = st.file_uploader(label='Hello', label_visibility='collapsed', type=['csv', 'xls', 'xlsx'], accept_multiple_files=False)

        # 初始化df
        df = pd.DataFrame()
        # 根据是否选择了文件显示消息
        if uploaded_file is not None:
            # 获取文件后缀
            ext = os.path.splitext(uploaded_file.name)[1] if uploaded_file else ''
            if ext == '.csv':
                df = pd.read_csv(uploaded_file)
            elif ext in ['.xls', '.xlsx']:
                df = pd.read_excel(uploaded_file)
            else:
                st.warning('请选择CSV或Excel!', icon="⚠️")
        
        if not df.empty:
            st.write(df)
        
        st.subheader('数据分析报告', divider='rainbow', help='这是数据统计信息的分析报告。')
        if df.empty:
            st.write('None')
        else:
            # 创建下拉选择框
            columns = list(df.columns)
            columns.insert(0, '无')
            column = st.selectbox('选择时间列:', columns)
            # 定义时间模式
            tsmode = False
            sortby = None
            title = 'Data Report'
            if column != '无':
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    df[column] = pd.to_datetime(df[column])
                    sortby = column
                    tsmode = True
                    title = 'Time-Series EDA'
                else:
                    st.warning('时间格式不匹配, 参考2024-01-05 17:38:04')
                    st.stop()
                    
            #如果数据量过大, 开启简化模式
            minimal = True if df.shape[0] > 100_000 else False
            report = ydata_profiling.ProfileReport(df, title=title, explorative=True, minimal=minimal, tsmode=tsmode, sortby=sortby)
            
            # 添加下载按钮
            now = datetime.datetime.now()
            now_str = now.strftime('%Y%m%d%H%M%S')
            file_name = f'datareport_{now_str}.html'
            btn = st.download_button(label='下载', data=report.to_html(), file_name=file_name, mime='text/html')
            st_profile_report(report, navbar=True)
            st.balloons()
            
    with tab3:
        # 模型训练
        st.subheader('1.模型训练', divider='rainbow', help='算法自动将训练数据拆分为训练集和验证集, 训练过程中, 自动根据验证集评估模型。')
        # 设置上传文件组件
        train_data_file = st.file_uploader('选择训练数据:', type=['csv', 'xls', 'xlsx'], accept_multiple_files=False)
        train_data = pd.DataFrame()
        # 根据是否选择了文件显示消息
        if train_data_file is not None:
            # 获取文件后缀
            ext = os.path.splitext(train_data_file.name)[1] if train_data_file else ''
            if ext == '.csv':
                train_data = pd.read_csv(train_data_file)
            elif ext in ['.xls', '.xlsx']:
                train_data = pd.read_excel(train_data_file)
            else:
                st.warning('请选择CSV或Excel!', icon="⚠️")
            st.write(train_data)
            
        # 设置标签
        label_col = None
        if not train_data.empty:
            label_col = st.selectbox('选择标签列:', train_data.columns)
            # 开始训练
            if label_col is not None:
                if st.button('开始训练'):
                    train_data.dropna(subset=[label_col], inplace=True)
                    if train_data.empty:
                        st.warning('有效训练数据为空!', icon="⚠️")
                    else:
                        save_path = './autoModels'
                        predictor = TabularPredictor(label=label_col, path=save_path)
                        predictor.fit(train_data, time_limit=60*60)
                        st.write(predictor.leaderboard())
                        ep = st.expander('模型注释：')
                        ep.write('''
                                model: 模型的名称。\n
                                score_val: 模型验证分数, 即在验证集上的eval_metric值。注意: 分数总是更高对应更好。这意味着log_loss和root_mean_squared_error等指标将*-1, 得到负数。\n
                                eval_metric: 用于计算分数的评估指标名称。\n
                                pred_time_val: 端到端计算验证数据预测所需的推理/预测时间。等效于模型及其所有基础模型的所有pred_time_val_marginal值之和。\n
                                fit_time: 端到端训练模型所需的拟合时间（如果模型是堆栈集成, 则包括基础模型）。等效于模型及其所有基础模型的所有fit_time_marginal值之和。\n
                                pred_time_val_marginal: 计算验证数据的预测所需的推理/预测时间（忽略基础模型的推理/预测时间）。\n
                                fit_time_marginal: 训练模型所需的拟合时间（忽略基础模型）。\n
                                stack_level: 模型的堆栈级别。堆栈级别为N的模型可以将堆栈级别小于N的任何模型集作为输入, 堆栈级别为1的模型没有模型输入。\n
                                can_infer: 模型是否能够对新数据执行推理/预测。如果为False, 则模型未保存、已删除或模型的祖先无法推理/预测。\n
                                fit_order: 模型拟合的顺序。第一个模型拟合的fit_order=1, 第N个模型拟合的fit_order=N。在袋装集合的情况下, 该顺序对应于第一子模型拟合。\n
                                ''')
                        st.balloons()
                        st.success('训练完成!', icon="✅")
        else:
            pass               
        # 模型预测
        st.subheader('2.模型预测', divider='rainbow', help='模型训练以后, 使用训练好的模型在新的数据上进行预测。')
        data_new_file = st.file_uploader('选择预测数据: ', type=['csv', 'xls', 'xlsx'], accept_multiple_files=False)
        data_new = pd.DataFrame()
        if data_new_file is not None:
            ext = os.path.splitext(data_new_file.name)[1] if data_new_file else ''
            if ext == '.csv':
                data_new = pd.read_csv(data_new_file)
            elif ext in ['.xls', '.xlsx']:
                data_new = pd.read_excel(data_new_file)
            else:
                st.warning('请选择CSV或Excel!', icon="⚠️")
            st.write(data_new)
            # 模型评估
            if not data_new.empty:
                same_feature = False
                try:
                    save_path = './autoModels'
                    predictor = TabularPredictor.load(path=save_path)
                    label = predictor.label
                    st.success('加载模型完成!', icon="✅")
                    # st.write(predictor.leaderboard(extra_info=True))
                    st.write("现有模型的预测标签(若要更换标签，需重新训练模型): ", label)
                    features = predictor.features()
                    # 判断预测数据中是否含有所有特征列
                    same_feature = np.array([True if x in data_new.columns else False for x in features]).all()
                    if not same_feature:
                        st.warning(f'预测数据中缺少部分特征列! 需要的特征列为{features}。', icon="⚠️")
                except:
                    st.warning('请先训练模型!', icon="⚠️")
                else:
                    if same_feature:
                        if st.button('开始预测'):
                            y_pred = predictor.predict(data_new)
                            label_name = label+"_pred"
                            data_new[label_name] = y_pred
                            st.write(data_new)
                            st.balloons()
                            st.success('预测完成!', icon="✅")
                            st.stop()
                
if __name__ == '__main__':
    main()
