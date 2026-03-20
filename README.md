# ML Model Evaluation Panel

这是一个基于 Streamlit 构建的机器学习模型评估面板应用。它允许用户上传自定义数据集 (CSV 或 Excel)，并自动训练 Random Forest 和 XGBoost 回归模型，提供数据前置分析（共线性 VIF 与相关性矩阵）和模型评估结果。

## 特性

- **数据支持**: 动态上传 `CSV` 或 `XLSX` 数据集。
- **前置分析**:
  - 特征多重共线性分析 (VIF)
  - 特征相关性矩阵及热力图 (Correlation Matrix)
- **模型训练**: 
  - 支持选择 Random Forest 和/或 XGBoost。
  - 支持动态调整测试集划分比例。
  - 支持调节模型关键超参数 (如 `n_estimators`, `max_depth`, `learning_rate`)。
- **结果可视化**: 
  - 计算 R², MSE, RMSE, MAE 等回归评估指标。
  - 自动生成特征重要性排序图表和表格。
- **缓存管理**: 提供手动清除、中断清除和定时 5 分钟自动销毁缓存机制。

## 本地运行指南

1. **克隆代码到本地**。
2. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```
3. **启动 Streamlit 服务**:
   ```bash
   streamlit run app.py
   ```
4. 浏览器会自动打开 `http://localhost:8501`。

## 部署到 Streamlit Community Cloud

此项目已完美配置为可直接部署到 [Streamlit Community Cloud](https://streamlit.io/cloud)。部署步骤如下：

1. **上传至 GitHub**:
   将本文件夹内的所有代码 (包括 `app.py`, `model_training.py`, `requirements.txt` 等) 提交到你的一个公开或私有 GitHub 仓库中。
   
2. **登录 Streamlit Cloud**:
   前往 [share.streamlit.io](https://share.streamlit.io) 并使用你的 GitHub 账号登录。

3. **创建新应用 (New app)**:
   - 点击右上角的 "New app" 按钮。
   - **Repository**: 选择你刚刚上传代码的 GitHub 仓库。
   - **Branch**: 默认一般是 `main` 或 `master`。
   - **Main file path**: 填写 `app.py`。
   - **App URL**: (可选) 自定义你的专属二级域名。

4. **点击 Deploy!**
   Streamlit 服务器会自动读取仓库中的 `requirements.txt` 来安装所有必要的环境（包括 `xgboost`, `scikit-learn` 等），安装完成后，你的应用即可在云端公开访问！

---
**注意事项**: 
应用在运行时会生成一些临时图表文件 (如 `correlation_matrix.png`, `random_forest_feature_importance.png` 等)。部署在 Streamlit Cloud 时，当前工作目录具有写权限，因此图表可以正常保存和渲染。若部署至 Docker 或其他云环境，请确保应用对运行目录具备写入权限。