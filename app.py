import streamlit as st
import pandas as pd
import time
import threading
import uuid
import os
import model_training

# 提升上传文件大小限制相关的提示（实际限制需在 config.toml 中配置，但可通过代码给用户提示）
# st.set_page_config 必须是第一个调用的 st 函数

# 全局并发控制 (使用 streamlit 的 cache_resource 来保证跨 session 共享)
@st.cache_resource
def get_training_lock_and_counter():
    return threading.Lock(), [0]  # [0] 是一种可变对象，能在不同线程间保持引用

global_lock, active_trainings = get_training_lock_and_counter()
MAX_CONCURRENT_TRAININGS = 3

# 自动清除缓存数据的定时器函数
def auto_clear_cache(session_state_dict, keys_to_clear):
    time.sleep(300) # 等待 5 分钟 (300秒)
    for key in keys_to_clear:
        if key in session_state_dict:
            del session_state_dict[key]

st.set_page_config(page_title="机器学习模型评估预览", layout="wide")

# 为每个用户生成唯一的 session_id
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
    st.session_state.clear()
    st.session_state['session_id'] = str(uuid.uuid4()) # 重新赋一下，因为 clear 把他清掉了
    st.session_state['initialized'] = True

st.title("🌲 Random Forest 与 XGBoost 回归模型评估面板")
st.markdown("上传您的 CSV 或 Excel 数据集，选择目标变量（因变量），即可自动训练并查看模型评估结果。")

st.divider()

# 数据上传部分
st.sidebar.header("📂 数据设置")

# 手动清除缓存按钮
if st.sidebar.button("🗑️ 清除缓存数据"):
    for key in list(st.session_state.keys()):
        if key not in ['initialized', 'session_id']:
            del st.session_state[key]
    st.sidebar.success("缓存数据已清除！")
    st.rerun()

st.sidebar.markdown("<small>提示：如果云端上传失败，请确保文件不超过限制，并尝试刷新页面重试。</small>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("上传您的数据集 (CSV 或 Excel)", type=["csv", "xlsx"])

# 只有当用户上传了文件才继续
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.sidebar.success("数据加载成功！")
        st.sidebar.write(f"数据维度: {df.shape[0]} 行 x {df.shape[1]} 列")
        
        # 选择因变量
        columns = df.columns.tolist()
        target_column = st.sidebar.selectbox("请选择目标变量 (因变量)", options=columns)
        
        # 划分训练集和测试集的比例
        test_size = st.sidebar.slider("测试集比例 (Test Size)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        st.sidebar.caption(f"当前设置: 训练集 {int((1-test_size)*100)}% / 测试集 {int(test_size*100)}%")
        
        st.sidebar.divider()
        st.sidebar.header("🔬 前置实验分析")
        run_vif = st.sidebar.checkbox("执行多重共线性分析 (VIF)", value=False)
        run_corr = st.sidebar.checkbox("执行特征相关性分析 (Correlation Matrix)", value=False)
        
        # 运行前置实验按钮
        if run_vif or run_corr:
            if st.sidebar.button("🔬 运行前置实验分析", type="secondary"):
                with st.spinner("正在运行前置实验分析，请稍候..."):
                    if run_vif:
                        st.session_state['vif_results'] = model_training.calculate_vif(df, target_column)
                    if run_corr:
                        corr_matrix, corr_fig = model_training.generate_correlation_matrix(df, target_column, st.session_state['session_id'])
                        st.session_state['corr_matrix'] = corr_matrix
                        st.session_state['corr_fig'] = corr_fig
                    st.success("前置实验分析完成！")
        
        st.sidebar.divider()
        st.sidebar.header("⚙️ 模型设置")
        
        # 模型选择改为单选
        available_models = ["Random Forest", "XGBoost", "CatBoost"]
        selected_model = st.sidebar.selectbox("请选择要训练的模型 (单选)", options=available_models)
        
        model_params = {}
        
        # 动态参数选择
        if selected_model == "Random Forest":
            st.sidebar.subheader("Random Forest 参数")
            rf_n_estimators = st.sidebar.slider("树的数量 (n_estimators) [RF]", min_value=1, max_value=500, value=100, step=1)
            rf_max_depth = st.sidebar.slider("最大深度 (max_depth) [RF]", min_value=1, max_value=50, value=10, step=1)
            rf_limit_depth = st.sidebar.checkbox("限制最大深度 [RF]", value=False)
            model_params["Random Forest"] = {
                "n_estimators": rf_n_estimators,
                "max_depth": rf_max_depth if rf_limit_depth else None
            }
            
        elif selected_model == "XGBoost":
            st.sidebar.subheader("XGBoost 参数")
            xgb_n_estimators = st.sidebar.slider("树的数量 (n_estimators) [XGB]", min_value=1, max_value=500, value=100, step=1)
            xgb_max_depth = st.sidebar.slider("最大深度 (max_depth) [XGB]", min_value=1, max_value=20, value=6, step=1)
            xgb_learning_rate = st.sidebar.number_input("学习率 (learning_rate) [XGB]", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            model_params["XGBoost"] = {
                "n_estimators": xgb_n_estimators,
                "max_depth": xgb_max_depth,
                "learning_rate": xgb_learning_rate
            }
            
        elif selected_model == "CatBoost":
            st.sidebar.subheader("CatBoost 参数")
            cb_iterations = st.sidebar.slider("迭代次数/树的数量 (iterations) [CB]", min_value=1, max_value=1000, value=100, step=10)
            cb_depth = st.sidebar.slider("树的深度 (depth) [CB]", min_value=1, max_value=16, value=6, step=1)
            cb_learning_rate = st.sidebar.number_input("学习率 (learning_rate) [CB]", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            
            st.sidebar.markdown("**特征类型指定 (可选)**")
            # 供选择的特征列（排除因变量）
            feature_columns = [col for col in columns if col != target_column]
            
            cat_features = st.sidebar.multiselect("选择类别型变量 (Categorical Features)", options=feature_columns)
            # 文本变量不能和类别变量重复，做个简单的提示或过滤
            text_features = st.sidebar.multiselect("选择文本型变量 (Text Features)", options=[col for col in feature_columns if col not in cat_features])
            
            model_params["CatBoost"] = {
                "iterations": cb_iterations,
                "depth": cb_depth,
                "learning_rate": cb_learning_rate,
                "cat_features": cat_features,
                "text_features": text_features
            }

        st.sidebar.divider()
        
        # 开始训练和终止训练按钮
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_training = st.button("▶️ 开始训练", type="primary")
        with col2:
            stop_training = st.button("⏹️ 终止训练")
            
        if stop_training:
            # 清除所有缓存数据
            for key in list(st.session_state.keys()):
                if key not in ['initialized', 'session_id']:
                    del st.session_state[key]
            st.sidebar.warning("训练已终止，缓存数据已清除！")
            st.rerun()
        
        # 预览数据
        with st.expander("预览上传的数据集"):
            st.dataframe(df.head(10))
            
        if start_training:
            if not selected_model:
                st.sidebar.warning("请选择一个模型进行训练！")
            else:
                can_train = False
                with global_lock:
                    if active_trainings[0] < MAX_CONCURRENT_TRAININGS:
                        active_trainings[0] += 1
                        can_train = True

                if can_train:
                    try:
                        with st.spinner(f"正在训练 {selected_model} 模型，请稍候...可以点击 '终止训练' 取消"):
                            # 运行训练流水线，传递 session_id 生成唯一文件
                            results = model_training.run_training_pipeline(df, target_column, selected_model, model_params, st.session_state['session_id'], test_size)
                            st.session_state['training_results'] = results
                            st.success("模型训练完成！(结果将在 5 分钟后自动清除)")
                            
                            # 启动 5 分钟后自动清除缓存的后台线程
                            keys_to_clear = ['training_results', 'vif_results', 'corr_matrix', 'corr_fig']
                            timer_thread = threading.Thread(target=auto_clear_cache, args=(st.session_state, keys_to_clear))
                            # 设置为守护线程，防止阻塞主程序退出
                            timer_thread.daemon = True
                            timer_thread.start()
                    finally:
                        with global_lock:
                            active_trainings[0] -= 1
                else:
                    st.sidebar.error("当前服务器训练任务过多 (超过并发限制)，请稍后重试！")
                
    except Exception as e:
        st.sidebar.error(f"读取数据时发生错误: {e}")
else:
    st.info("👈 请在左侧侧边栏上传您的数据集以开始。")

# 前置实验结果展示
if 'vif_results' in st.session_state or 'corr_matrix' in st.session_state:
    st.header("🔬 前置实验分析结果")
    
    if 'vif_results' in st.session_state:
        st.subheader("多重共线性分析 (VIF)")
        st.markdown("方差膨胀因子 (VIF) 用于检测特征之间的多重共线性。一般认为 VIF > 10 表示存在严重的共线性问题。")
        st.dataframe(st.session_state['vif_results'], use_container_width=True)
        
    if 'corr_matrix' in st.session_state:
        st.subheader("特征相关性矩阵 (Correlation Matrix)")
        st.markdown("展示各个数值型特征之间的皮尔逊相关系数。")
        
        st.markdown("**相关性热力图 (Heatmap)**")
        st.image(st.session_state['corr_fig'], caption="Feature Correlation Matrix", use_container_width=True)
        
        st.markdown("**相关性系数表格**")
        st.dataframe(st.session_state['corr_matrix'], use_container_width=True)
            
    st.divider()

# 如果会话中已有训练结果，则展示它们
if 'training_results' in st.session_state:
    results = st.session_state['training_results']
    
    # 定义用于展示指标的函数
    def display_metrics(metrics_dict, dataset_type):
        cols = st.columns(4)
        cols[0].metric(label=f"R² (R-squared)", value=f"{metrics_dict['R2']:.4f}")
        cols[1].metric(label=f"MSE (均方误差)", value=f"{metrics_dict['MSE']:.4f}")
        cols[2].metric(label=f"RMSE (均方根误差)", value=f"{metrics_dict['RMSE']:.4f}")
        cols[3].metric(label=f"MAE (平均绝对误差)", value=f"{metrics_dict['MAE']:.4f}")

    # 1. 随机森林部分
    if 'rf' in results:
        st.header("1. Random Forest (随机森林)")
        st.subheader("📊 评估指标")

        st.markdown("**训练集 (Training Set)**")
        display_metrics(results['rf']['metrics']['train'], "训练集")

        st.markdown("**测试集 (Test Set)**")
        display_metrics(results['rf']['metrics']['test'], "测试集")

        st.subheader("🏆 特征重要性")
        
        st.markdown("**特征重要性图表**")
        st.image(results['rf']['fig_path'], caption="Random Forest 特征重要性柱状图", use_container_width=True)
        
        st.markdown("**特征重要性表格**")
        rf_df = pd.DataFrame(results['rf']['importance'])
        rf_df.index = rf_df.index + 1
        st.dataframe(rf_df.rename(columns={'features': '特征名', 'importances': '重要性得分'}), use_container_width=True)

        st.divider()

    # 2. XGBoost部分
    if 'xgb' in results:
        st.header("2. XGBoost")
        st.subheader("📊 评估指标")

        st.markdown("**训练集 (Training Set)**")
        display_metrics(results['xgb']['metrics']['train'], "训练集")

        st.markdown("**测试集 (Test Set)**")
        display_metrics(results['xgb']['metrics']['test'], "测试集")

        st.subheader("🏆 特征重要性")
        
        st.markdown("**特征重要性图表**")
        st.image(results['xgb']['fig_path'], caption="XGBoost 特征重要性柱状图", use_container_width=True)
        
        st.markdown("**特征重要性表格**")
        xgb_df = pd.DataFrame(results['xgb']['importance'])
        xgb_df.index = xgb_df.index + 1
        st.dataframe(xgb_df.rename(columns={'features': '特征名', 'importances': '重要性得分'}), use_container_width=True)

        st.divider()

    # 3. CatBoost部分
    if 'cb' in results:
        st.header("3. CatBoost")
        st.subheader("📊 评估指标")

        st.markdown("**训练集 (Training Set)**")
        display_metrics(results['cb']['metrics']['train'], "训练集")

        st.markdown("**测试集 (Test Set)**")
        display_metrics(results['cb']['metrics']['test'], "测试集")

        st.subheader("🏆 特征重要性")
        
        st.markdown("**特征重要性图表**")
        st.image(results['cb']['fig_path'], caption="CatBoost 特征重要性柱状图", use_container_width=True)
        
        st.markdown("**特征重要性表格**")
        cb_df = pd.DataFrame(results['cb']['importance'])
        cb_df.index = cb_df.index + 1
        st.dataframe(cb_df.rename(columns={'features': '特征名', 'importances': '重要性得分'}), use_container_width=True)
