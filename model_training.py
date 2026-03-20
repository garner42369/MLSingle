import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df, target_column):
    """
    计算特征的方差膨胀因子 (VIF)
    """
    X = df.drop(columns=[target_column])
    
    # 过滤掉非数值列，VIF只能计算数值型特征
    X_numeric = X.select_dtypes(include=[np.number])
    
    vif_data = pd.DataFrame()
    vif_data["特征"] = X_numeric.columns
    
    # 计算VIF值
    # 如果数据量大，可能会比较慢
    vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) 
                       for i in range(len(X_numeric.columns))]
    
    # 按VIF降序排列
    vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)
    return vif_data

def generate_correlation_matrix(df, target_column, session_id=""):
    """
    生成相关性矩阵的热力图并返回路径及数据
    """
    # 获取数值列
    numeric_df = df.select_dtypes(include=[np.number])
    
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    
    fig_name = f"correlation_matrix_{session_id}.png" if session_id else "correlation_matrix.png"
    fig_path = os.path.join(os.getcwd(), fig_name)
    plt.savefig(fig_path)
    plt.close()
    
    return corr_matrix, fig_path

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, feature_names, session_id=""):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算 R2 分数和各种误差
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n================ {model_name} ================")
    print(f"【训练集评估】")
    print(f"R2 Score: {r2_train:.4f}")
    print(f"MSE:      {mse_train:.4f}")
    print(f"RMSE:     {rmse_train:.4f}")
    print(f"MAE:      {mae_train:.4f}")
    print(f"\n【测试集评估】")
    print(f"R2 Score: {r2_test:.4f}")
    print(f"MSE:      {mse_test:.4f}")
    print(f"RMSE:     {rmse_test:.4f}")
    print(f"MAE:      {mae_test:.4f}")
    
    # 获取特征重要性
    importances = model.feature_importances_
    # 对特征重要性进行降序排序
    indices = np.argsort(importances)[::-1]
    
    print(f"\n{model_name} 特征重要性排序:")
    for f in range(X_train.shape[1]):
        feature_name = feature_names[indices[f]]
        importance_val = importances[indices[f]]
        print(f"{f + 1}. {feature_name}: {importance_val:.4f}")
        
    # 绘制特征重要性柱状图
    plt.figure(figsize=(10, 6))
    plt.title(f"{model_name} Feature Importances")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    
    # 保存图片
    fig_name = f"{model_name.replace(' ', '_').lower()}_feature_importance_{session_id}.png" if session_id else f"{model_name.replace(' ', '_').lower()}_feature_importance.png"
    fig_path = os.path.join(os.getcwd(), fig_name)
    
    # 确保保存路径有效
    plt.savefig(fig_path)
    plt.close()
    
    # 返回评估结果和特征重要性，方便其他地方调用
    metrics = {
        'train': {'R2': r2_train, 'MSE': mse_train, 'RMSE': rmse_train, 'MAE': mae_train},
        'test': {'R2': r2_test, 'MSE': mse_test, 'RMSE': rmse_test, 'MAE': mae_test}
    }
    feature_importances = {'features': [feature_names[i] for i in indices], 'importances': importances[indices]}
    
    return metrics, feature_importances, fig_path

def run_training_pipeline(df, target_column, selected_models, model_params, session_id="", test_size=0.2):
    """
    运行训练流水线：划分数据集、根据选择训练模型、返回评估指标和图表路径
    """
    # 准备特征 (X) 和目标变量 (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    feature_names = X.columns.tolist()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    results = {}

    if 'Random Forest' in selected_models:
        params = model_params.get('Random Forest', {})
        n_estimators = params.get('n_estimators', 100)
        max_depth = params.get('max_depth', None)
        rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf_metrics, rf_importance, rf_fig = train_and_evaluate(rf_model, "Random Forest", X_train, X_test, y_train, y_test, feature_names, session_id)
        results['rf'] = {'metrics': rf_metrics, 'importance': rf_importance, 'fig_path': rf_fig}

    if 'XGBoost' in selected_models:
        params = model_params.get('XGBoost', {})
        n_estimators = params.get('n_estimators', 100)
        max_depth = params.get('max_depth', 6)
        learning_rate = params.get('learning_rate', 0.1)
        xgb_model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
        xgb_metrics, xgb_importance, xgb_fig = train_and_evaluate(xgb_model, "XGBoost", X_train, X_test, y_train, y_test, feature_names, session_id)
        results['xgb'] = {'metrics': xgb_metrics, 'importance': xgb_importance, 'fig_path': xgb_fig}

    return results
