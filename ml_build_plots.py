import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import cross_val_predict, StratifiedKFold, RandomizedSearchCV, cross_val_score, validation_curve, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc, roc_auc_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, label_binarize
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.kernel_ridge import KernelRidge
import statsmodels.api as sm
import os
from scipy.stats import randint, uniform, spearmanr, pearsonr
from scipy import stats
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def setup_logging(output_dir):
    """设置日志记录"""
    log_file = os.path.join(output_dir, f'ml_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件创建: {log_file}")
    return logger

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'ml_analysis_plots_{current_time}'
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在则创建

logger = setup_logging(output_dir)
logger.info("="*50)
logger.info("机器学习分析开始（增强图表版本）")
logger.info(f"输出目录: {output_dir}")
logger.info("="*50)

def plot_regression_residuals(y_true, y_pred, model_name, target_column, output_dir, logger):
    """绘制回归模型的残差分析图"""
    residuals = y_true - y_pred
    standardized_residuals = residuals / np.std(residuals)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.scatter(y_pred, residuals, alpha=0.6, color='steelblue')
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('预测值')
    ax1.set_ylabel('残差 (实际值 - 预测值)')
    ax1.set_title(f'残差 vs 预测值 ({model_name})')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(y_pred, standardized_residuals, alpha=0.6, color='green')
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.axhline(y=2, color='red', linestyle=':', alpha=0.7)
    ax2.axhline(y=-2, color='red', linestyle=':', alpha=0.7)
    ax2.set_xlabel('预测值')
    ax2.set_ylabel('标准化残差')
    ax2.set_title(f'标准化残差 vs 预测值 ({model_name})')
    ax2.grid(True, alpha=0.3)
    
    ax3.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--')
    ax3.set_xlabel('残差')
    ax3.set_ylabel('频数')
    ax3.set_title(f'残差分布 ({model_name})')
    ax3.grid(True, alpha=0.3)
    
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title(f'残差Q-Q图 ({model_name})')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'回归残差分析: {target_column} - {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_column}_{model_name}_residual_analysis.pdf'))
    plt.close()
    logger.info(f"残差分析图已保存: {target_column}_{model_name}_residual_analysis.pdf")

def plot_prediction_vs_actual(y_true, y_pred, model_name, target_column, output_dir, logger):
    """绘制预测值 vs 实际值图"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, color='steelblue')
    
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线 (y=x)')
    
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), 'g-', linewidth=2, alpha=0.8, label=f'拟合线 (y={z[0]:.3f}x+{z[1]:.3f})')
    
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'预测值 vs 实际值: {target_column} - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    textstr = f'R$^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nPearson r = {pearson_r:.4f}\nSpearman rho = {spearman_r:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_column}_{model_name}_prediction_vs_actual.pdf'))
    plt.close()
    logger.info(f"预测vs实际值图已保存: {target_column}_{model_name}_prediction_vs_actual.pdf")

def plot_learning_curves(model, X, y, model_name, target_column, output_dir, logger):
    """绘制学习曲线"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2', random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='训练分数')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='验证分数')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
    
    plt.xlabel('训练样本数')
    plt.ylabel('R$^2$ 分数')
    plt.title(f'学习曲线: {target_column} - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_column}_{model_name}_learning_curve.pdf'))
    plt.close()
    logger.info(f"学习曲线已保存: {target_column}_{model_name}_learning_curve.pdf")

def plot_feature_importance(model, feature_names, model_name, target_column, output_dir, logger):
    """绘制特征重要性图"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        if len(importances) != len(feature_names):
            logger.warning(f"特征重要性数量({len(importances)})与特征名称数量({len(feature_names)})不匹配，跳过特征重要性图")
            return
            
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'特征重要性: {target_column} - {model_name}')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{target_column}_{model_name}_feature_importance.pdf'))
        plt.close()
        logger.info(f"特征重要性图已保存: {target_column}_{model_name}_feature_importance.pdf")
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef.flatten()
        coef = np.abs(coef)
        
        if len(coef) != len(feature_names):
            logger.warning(f"系数数量({len(coef)})与特征名称数量({len(feature_names)})不匹配，跳过特征系数图")
            return
            
        indices = np.argsort(coef)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'特征系数(绝对值): {target_column} - {model_name}')
        plt.bar(range(len(coef)), coef[indices])
        plt.xticks(range(len(coef)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('特征')
        plt.ylabel('系数绝对值')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{target_column}_{model_name}_feature_coefficients.pdf'))
        plt.close()
        logger.info(f"特征系数图已保存: {target_column}_{model_name}_feature_coefficients.pdf")
    else:
        logger.info(f"模型 {model_name} 不支持特征重要性或系数提取，跳过特征重要性图")

def plot_cross_validation_scores(cv_scores, model_names, target_column, output_dir, logger):
    """绘制交叉验证分数分布图"""
    plt.figure(figsize=(15, 8))
    
    all_scores = []
    all_models = []
    for model_name in model_names:
        if model_name in cv_scores:
            for score in cv_scores[model_name]:
                all_scores.append(score)
                all_models.append(model_name)
    
    if all_scores:
        df = pd.DataFrame({'模型': all_models, 'R$^2$分数': all_scores})
        
        model_avg_scores = df.groupby('模型')['R$^2$分数'].mean().sort_values(ascending=False)
        sorted_models = model_avg_scores.index.tolist()
        
        sns.boxplot(data=df, x='模型', y='R$^2$分数', order=sorted_models, palette='viridis')
        plt.title(f'交叉验证R$^2$分数分布: {target_column}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{target_column}_cv_scores_distribution.pdf'))
        plt.close()
        logger.info(f"交叉验证分数分布图已保存: {target_column}_cv_scores_distribution.pdf")

def plot_model_performance_radar(regression_results, target_column, output_dir, logger, max_models=None):
    """绘制模型性能雷达图"""
    models = list(regression_results.keys())
    if len(models) < 3:
        logger.warning(f"模型数量不足，跳过雷达图绘制: {target_column}")
        return
    
    sorted_models = sorted(models, key=lambda x: regression_results[x]['R2'], reverse=True)
    
    if max_models is None:
        max_models = min(len(sorted_models), 8)  # 最多显示8个模型
    models = sorted_models[:max_models]
    
    logger.info(f"雷达图将显示 {len(models)} 个模型: {models}")
    
    metrics = ['R2', 'MSE', 'RMSE', 'MAE']
    values = []
    
    for model in models:
        model_values = []
        model_values.append(max(0, regression_results[model]['R2']))
        mse_norm = 1 / (1 + regression_results[model]['MSE'])
        rmse_norm = 1 / (1 + regression_results[model]['RMSE'])
        mae_norm = 1 / (1 + regression_results[model]['MAE'])
        model_values.extend([mse_norm, rmse_norm, mae_norm])
        values.append(model_values)
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for i, (model, model_values) in enumerate(zip(models, values)):
        model_values += model_values[:1]  # 闭合
        ax.plot(angles, model_values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, model_values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title(f'模型性能雷达图: {target_column}', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if len(models) <= 8:
        filename = f'{target_column}_model_performance_radar.pdf'
    else:
        filename = f'{target_column}_model_performance_radar_all_models.pdf'
    
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    logger.info(f"模型性能雷达图已保存: {filename}")

def plot_classification_confusion_matrix(y_true, y_pred, model_name, target_column, output_dir, logger, class_names=None):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'混淆矩阵: {target_column} - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_column}_{model_name}_confusion_matrix.pdf'))
    plt.close()
    logger.info(f"混淆矩阵已保存: {target_column}_{model_name}_confusion_matrix.pdf")

def plot_classification_metrics_comparison(classification_results, target_column, output_dir, logger):
    """绘制分类模型性能指标对比图"""
    models = list(classification_results.keys())
    auc_scores = [classification_results[model]['AUC'] for model in models]
    
    if 'MCC' in classification_results[models[0]]:
        mcc_scores = [classification_results[model]['MCC'] for model in models]
    else:
        mcc_scores = [0.5] * len(models)  # 默认值
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    bars1 = ax1.bar(range(len(models)), auc_scores, color='steelblue', alpha=0.8)
    ax1.set_xlabel('模型')
    ax1.set_ylabel('AUC分数')
    ax1.set_title(f'分类模型AUC分数对比: {target_column}')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    for bar, score in zip(bars1, auc_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    bars2 = ax2.bar(range(len(models)), mcc_scores, color='lightcoral', alpha=0.8)
    ax2.set_xlabel('模型')
    ax2.set_ylabel('MCC分数')
    ax2.set_title(f'分类模型MCC分数对比: {target_column}')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    for bar, score in zip(bars2, mcc_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_column}_classification_metrics_comparison.pdf'))
    plt.close()
    logger.info(f"分类模型性能对比图已保存: {target_column}_classification_metrics_comparison.pdf")

def plot_precision_recall_curves(y_true, classification_results, target_column, output_dir, logger):
    """绘制精确率-召回率曲线"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    plt.figure(figsize=(10, 8))
    
    for model_name, result in classification_results.items():
        y_proba = result['proba']
        
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            if y_proba.shape[1] == 2:  # 二分类
                y_score = y_proba[:, 1]
            else:  # 多分类，计算宏平均
                try:
                    avg_precision = average_precision_score(
                        label_binarize(y_true, classes=np.unique(y_true)), 
                        y_proba, 
                        average='macro'
                    )
                    plt.plot([0, 1], [avg_precision, avg_precision], '--', 
                            label=f'{model_name} (Macro-Avg AP = {avg_precision:.3f})')
                except:
                    logger.warning(f"精确率-召回率曲线计算失败: {model_name}")
                continue
        else:
            y_score = y_proba.flatten()
        
        if len(np.unique(y_true)) == 2:
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                avg_precision = average_precision_score(y_true, y_score)
                plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})')
            except:
                logger.warning(f"精确率-召回率曲线计算失败: {model_name}")
    
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title(f'精确率-召回率曲线: {target_column}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_column}_precision_recall_curves.pdf'))
    plt.close()
    logger.info(f"精确率-召回率曲线已保存: {target_column}_precision_recall_curves.pdf")

def plot_classification_feature_importance(best_models, feature_names, target_column, output_dir, logger):
    """绘制分类模型特征重要性对比图"""
    valid_models = [(name, data) for name, data in best_models if 'best_model' in data]
    
    if not valid_models:
        logger.warning(f"没有可用的分类模型用于特征重要性分析: {target_column}")
        return
    
    fig, axes = plt.subplots(len(valid_models), 1, figsize=(12, 4*len(valid_models)))
    if len(valid_models) == 1:
        axes = [axes]
    
    for idx, (model_name, model_data) in enumerate(valid_models):
        model = model_data['best_model']
        ax = axes[idx]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if len(importances) == len(feature_names):
                indices = np.argsort(importances)[::-1][:10]  # 前10个最重要的特征
                
                ax.bar(range(len(indices)), importances[indices])
                ax.set_xticks(range(len(indices)))
                ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                ax.set_title(f'特征重要性: {model_name}')
                ax.set_ylabel('重要性')
                ax.grid(True, alpha=0.3)
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)  # 多分类时取平均
            else:
                coef = np.abs(coef)
                
            if len(coef) == len(feature_names):
                indices = np.argsort(coef)[::-1][:10]  # 前10个最重要的特征
                
                ax.bar(range(len(indices)), coef[indices])
                ax.set_xticks(range(len(indices)))
                ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                ax.set_title(f'特征系数(绝对值): {model_name}')
                ax.set_ylabel('系数绝对值')
                ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{model_name}\n不支持特征重要性分析', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'特征重要性: {model_name}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_column}_classification_feature_importance_comparison.pdf'))
    plt.close()
    logger.info(f"分类模型特征重要性对比图已保存: {target_column}_classification_feature_importance_comparison.pdf")

def plot_classification_cv_scores_distribution(cv_scores_cls, target_column, output_dir, logger):
    """绘制分类模型交叉验证分数分布图"""
    plt.figure(figsize=(15, 8))
    
    all_scores = []
    all_models = []
    for model_name in cv_scores_cls:
        if model_name in cv_scores_cls:
            for score in cv_scores_cls[model_name]:
                all_scores.append(score)
                all_models.append(model_name)
    
    if all_scores:
        df = pd.DataFrame({'模型': all_models, 'AUC分数': all_scores})
        
        sns.boxplot(data=df, x='模型', y='AUC分数', palette='Set2')
        plt.title(f'分类模型交叉验证AUC分数分布: {target_column}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{target_column}_classification_cv_scores_distribution.pdf'))
        plt.close()
        logger.info(f"分类模型交叉验证分数分布图已保存: {target_column}_classification_cv_scores_distribution.pdf")

def plot_classification_performance_radar(classification_results, target_column, output_dir, logger, max_models=None):
    """绘制分类模型性能雷达图"""
    models = list(classification_results.keys())
    if len(models) < 3:
        logger.warning(f"分类模型数量不足，跳过雷达图绘制: {target_column}")
        return
    
    sorted_models = sorted(models, key=lambda x: classification_results[x]['AUC'], reverse=True)
    
    if max_models is None:
        max_models = min(len(sorted_models), 8)
    models = sorted_models[:max_models]
    
    logger.info(f"分类雷达图将显示 {len(models)} 个模型: {models}")
    
    metrics = ['AUC']
    values = []
    
    for model in models:
        model_values = []
        model_values.append(classification_results[model]['AUC'])
        values.append(model_values)
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for i, (model, model_values) in enumerate(zip(models, values)):
        model_values += model_values[:1]  # 闭合
        ax.plot(angles, model_values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, model_values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title(f'分类模型性能雷达图: {target_column}', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if len(models) <= 8:
        filename = f'{target_column}_classification_performance_radar.pdf'
    else:
        filename = f'{target_column}_classification_performance_radar_all_models.pdf'
    
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    logger.info(f"分类模型性能雷达图已保存: {filename}")

file_path = 'Flavor_OAV_Sensory_Training_v3_noisy.xlsx'
logger.info(f"开始读取数据文件: {file_path}")

try:
    data = pd.read_excel(file_path)  # 读取Excel文件
    logger.info("数据文件读取成功")
except Exception as e:
    logger.error(f"读取Excel文件失败: {e}")
    try:
        data = pd.read_excel(file_path, sheet_name=0)  # 读取第一个sheet
        logger.info("使用第一个sheet读取成功")
    except Exception as e2:
        logger.error(f"读取第一个sheet也失败: {e2}")
        exit(1)

logger.info(f"数据文件形状: {data.shape}")
logger.info(f"列名: {list(data.columns)}")
print(f"数据文件形状: {data.shape}")
print(f"列名: {list(data.columns)}")
print(f"前5行数据:")
print(data.head())

possible_sensory_columns = ['Wine', 'Flower', 'Fruit', 'Grain', 'Smoke', 'Roast', 'Cooked vegetables', 'Sauce', 'Rancid']
available_sensory_columns = [col for col in possible_sensory_columns if col in data.columns]

if not available_sensory_columns:
    sensory_keywords = ['flower', 'fruit', 'grain', 'smoke', 'roast', 'sauce', 'rancid', 'sensory', 'quality']
    available_sensory_columns = [col for col in data.columns if any(keyword in col.lower() for keyword in sensory_keywords)]

print(f"找到的感官属性列: {available_sensory_columns}")

if not available_sensory_columns:
    available_sensory_columns = list(data.columns[-5:])
    print(f"使用最后5列作为感官属性: {available_sensory_columns}")

all_feature_columns = data.drop(columns=available_sensory_columns).columns
print(f"特征列数量: {len(all_feature_columns)}")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Lasso Regression': Lasso(alpha=0.1, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
    'Kernel Ridge': KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1),
    'KNN Regressor': KNeighborsRegressor(n_neighbors=5),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(learning_rate=0.01, max_depth=5, random_state=42),
    'AdaBoost': AdaBoostRegressor(n_estimators=50, learning_rate=0.1, random_state=42),
    'XGBoost': XGBRegressor(learning_rate=0.01, max_depth=5, n_estimators=100, random_state=42),
    'MLP Regressor': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.001, random_state=42, early_stopping=True, n_iter_no_change=10)
}

all_regression_results = {}
all_cv_scores = {}
all_classification_results = {}

for target_column in available_sensory_columns:
    print(f"\n--- 正在处理目标: {target_column} ---")
    logger.info(f"开始处理目标: {target_column}")

    X = data.drop(columns=available_sensory_columns)
    y = data[target_column]

    y_binned = pd.qcut(y, q=3, labels=False, duplicates='drop')  # 若分箱重复则自动去除
    
    if y_binned.nunique() < 2:
        print(f"跳过 {target_column}: 分箱后类别数不足，无法进行分类/ROC分析。")
        continue

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y_binned)

    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    y_true_all_current_target = None
    y_pred_proba_all = {} # 存储各模型预测概率

    if len(np.unique(y_resampled)) > 2:
        y_true_all_current_target = label_binarize(y_resampled, classes=np.unique(y_resampled))
    else:
        y_true_all_current_target = y_resampled
    
    current_target_results = {}
    current_cv_scores = {}

    logger.info(f"开始回归模型自动调参: {target_column}")
    print(f"自动调参并训练回归模型: {target_column}...")
    regression_results = {}
    best_params_all = {}
    best_scores_all = {}

    param_distributions = {
        'Ridge Regression': {'alpha': uniform(0.01, 10)},
        'Lasso Regression': {'alpha': uniform(0.01, 1)},
        'ElasticNet': {'alpha': uniform(0.01, 1), 'l1_ratio': uniform(0, 1)},
        'SVR': {'C': uniform(0.1, 10), 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']},
        'Kernel Ridge': {'alpha': uniform(0.01, 10), 'gamma': uniform(0.01, 1), 'kernel': ['rbf', 'linear']},
        'KNN Regressor': {'n_neighbors': randint(3, 20)},
        'Decision Tree': {'max_depth': randint(3, 20), 'min_samples_split': randint(2, 10)},
        'Random Forest': {'n_estimators': randint(50, 200), 'max_depth': randint(3, 20)},
        'Extra Trees': {'n_estimators': randint(50, 200), 'max_depth': randint(3, 20)},
        'Gradient Boosting': {'learning_rate': uniform(0.001, 0.2), 'n_estimators': randint(50, 200), 'max_depth': randint(3, 10)},
        'AdaBoost': {'n_estimators': randint(30, 100), 'learning_rate': uniform(0.01, 1)},
        'XGBoost': {'learning_rate': uniform(0.001, 0.2), 'n_estimators': randint(50, 200), 'max_depth': randint(3, 10)},
        'MLP Regressor': {'hidden_layer_sizes': [(100,), (100, 50), (50, 25)], 'alpha': uniform(0.0001, 0.01), 'max_iter': [500, 800, 1200, 2000], 'early_stopping': [True], 'n_iter_no_change': [10]},
    }
    skip_models = ['Linear Regression']
    
    X_original = data.drop(columns=available_sensory_columns)
    y_original = data[target_column]
    scaler_original = StandardScaler()
    X_original_scaled = scaler_original.fit_transform(X_original)
    
    logger.info("第一阶段：n_iter=20 自动调参")
    for name, model in models.items():
        logger.info(f"处理回归模型: {name}")
        if name in skip_models:
            logger.info(f"  {name}: 使用默认参数")
            y_pred_cv = cross_val_predict(model, X_original_scaled, y_original, cv=5)
            r2_score_cv = r2_score(y_original, y_pred_cv)
            mse_score_cv = mean_squared_error(y_original, y_pred_cv)
            rmse_score_cv = np.sqrt(mse_score_cv)
            mae_score_cv = mean_absolute_error(y_original, y_pred_cv)
            
            cv_scores = cross_val_score(model, X_original_scaled, y_original, cv=5, scoring='r2')
            current_cv_scores[name] = cv_scores
            
            regression_results[name] = {
                'R2': r2_score_cv,
                'MSE': mse_score_cv,
                'RMSE': rmse_score_cv,
                'MAE': mae_score_cv,
                'params': 'default',
                'y_pred': y_pred_cv,
                'cv_scores': cv_scores
            }
            best_scores_all[name] = r2_score_cv
            best_params_all[name] = 'default'
            logger.info(f"  {name}: R^2 = {r2_score_cv:.4f}, RMSE = {rmse_score_cv:.4f}, MAE = {mae_score_cv:.4f} (default)")
            print(f"  {name}: R^2 = {r2_score_cv:.4f}, RMSE = {rmse_score_cv:.4f}, MAE = {mae_score_cv:.4f} (default)")
            continue
        param_dist = param_distributions.get(name, None)
        if param_dist is None:
            logger.warning(f"  {name}: 未定义参数分布，跳过")
            print(f"  {name}: 未定义参数分布，跳过。")
            continue
        
        logger.info(f"  {name}: 开始RandomizedSearchCV调参 (n_iter=20)")
        search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=5, scoring='r2', n_jobs=-1, random_state=42)
        search.fit(X_original_scaled, y_original)
        best_model = search.best_estimator_
        y_pred_cv = cross_val_predict(best_model, X_original_scaled, y_original, cv=5)
        r2_score_cv = r2_score(y_original, y_pred_cv)
        mse_score_cv = mean_squared_error(y_original, y_pred_cv)
        rmse_score_cv = np.sqrt(mse_score_cv)
        mae_score_cv = mean_absolute_error(y_original, y_pred_cv)
        
        cv_scores = cross_val_score(best_model, X_original_scaled, y_original, cv=5, scoring='r2')
        current_cv_scores[name] = cv_scores
        
        regression_results[name] = {
            'R2': r2_score_cv,
            'MSE': mse_score_cv,
            'RMSE': rmse_score_cv,
            'MAE': mae_score_cv,
            'params': search.best_params_,
            'y_pred': y_pred_cv,
            'cv_scores': cv_scores,
            'best_model': best_model
        }
        best_scores_all[name] = r2_score_cv
        best_params_all[name] = search.best_params_
        logger.info(f"  {name}: R^2 = {r2_score_cv:.4f}, RMSE = {rmse_score_cv:.4f}, MAE = {mae_score_cv:.4f}, 最佳参数 = {search.best_params_}")
        print(f"  {name}: R^2 = {r2_score_cv:.4f}, RMSE = {rmse_score_cv:.4f}, MAE = {mae_score_cv:.4f}, params={search.best_params_}")

    top3 = sorted(best_scores_all.items(), key=lambda x: x[1], reverse=True)[:3]
    logger.info(f"第二阶段：top3模型精细调参 (n_iter=30)")
    logger.info(f"Top3模型: {[name for name, _ in top3]}")
    
    for name, _ in top3:
        if name in skip_models:
            logger.info(f"  {name}: 跳过精细调参（使用默认参数）")
            continue
        model = models[name]
        param_dist = param_distributions.get(name, None)
        if param_dist is None:
            logger.warning(f"  {name}: 参数分布未定义，跳过精细调参")
            continue
        logger.info(f"  {name}: 开始精细调参 (n_iter=30)")
        print(f"  {name}: 进入top3，n_iter=30精细调参...")
        search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=30, cv=5, scoring='r2', n_jobs=-1, random_state=42)
        search.fit(X_original_scaled, y_original)
        best_model = search.best_estimator_
        y_pred_cv = cross_val_predict(best_model, X_original_scaled, y_original, cv=5)
        r2_score_cv = r2_score(y_original, y_pred_cv)
        mse_score_cv = mean_squared_error(y_original, y_pred_cv)
        rmse_score_cv = np.sqrt(mse_score_cv)
        mae_score_cv = mean_absolute_error(y_original, y_pred_cv)
        
        cv_scores = cross_val_score(best_model, X_original_scaled, y_original, cv=5, scoring='r2')
        current_cv_scores[name] = cv_scores
        
        regression_results[name] = {
            'R2': r2_score_cv,
            'MSE': mse_score_cv,
            'RMSE': rmse_score_cv,
            'MAE': mae_score_cv,
            'params': search.best_params_,
            'y_pred': y_pred_cv,
            'cv_scores': cv_scores,
            'best_model': best_model
        }
        best_scores_all[name] = r2_score_cv
        best_params_all[name] = search.best_params_
        logger.info(f"  {name}: (精细调参) R^2 = {r2_score_cv:.4f}, RMSE = {rmse_score_cv:.4f}, MAE = {mae_score_cv:.4f}, 最佳参数 = {search.best_params_}")
        print(f"  {name}: (top3精调) R^2 = {r2_score_cv:.4f}, RMSE = {rmse_score_cv:.4f}, MAE = {mae_score_cv:.4f}, params={search.best_params_}")

    logger.info(f"回归模型调参完成，最终结果:")
    for name, score in sorted(best_scores_all.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {name}: R^2 = {score:.4f}, 参数 = {best_params_all[name]}")
    
    all_regression_results[target_column] = regression_results
    all_cv_scores[target_column] = current_cv_scores

    logger.info(f"保存Top5回归模型: {target_column}")
    top5_regression_models = sorted(regression_results.items(), key=lambda x: x[1]['R2'], reverse=True)[:5]
    top5_models_info = [f'{name} (R^2={data["R2"]:.4f})' for name, data in top5_regression_models]
    logger.info(f"Top5回归模型: {top5_models_info}")
    
    models_save_dir = os.path.join(output_dir, 'saved_models', target_column)
    os.makedirs(models_save_dir, exist_ok=True)
    
    import pickle
    saved_models_info = {}
    
    for model_name, model_data in top5_regression_models:
        try:
            model_info = {
                'model_name': model_name,
                'target_column': target_column,
                'R2': model_data['R2'],
                'RMSE': model_data['RMSE'],
                'MAE': model_data['MAE'],
                'params': model_data['params'],
                'feature_columns': X_original.columns.tolist(),
                'scaler': scaler_original  # 保存标准化器
            }
            
            if 'best_model' in model_data:
                model_info['model'] = model_data['best_model']
                model_file_path = os.path.join(models_save_dir, f'{model_name.replace(" ", "_")}_model.pkl')
                
                with open(model_file_path, 'wb') as f:
                    pickle.dump(model_info, f)
                
                saved_models_info[model_name] = {
                    'file_path': model_file_path,
                    'R2': model_data['R2'],
                    'RMSE': model_data['RMSE'],
                    'MAE': model_data['MAE']
                }
                logger.info(f"已保存模型: {model_name} -> {model_file_path}")
            else:
                logger.warning(f"模型 {model_name} 没有保存的best_model，跳过保存")
                
        except Exception as e:
            logger.error(f"保存模型 {model_name} 失败: {e}")
            continue
    
    summary_file = os.path.join(models_save_dir, 'models_summary.json')
    import json
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(saved_models_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"已保存 {len(saved_models_info)} 个模型到: {models_save_dir}")
    logger.info(f"模型信息汇总保存到: {summary_file}")

    logger.info(f"开始生成详细的回归分析图: {target_column}")
    
    best_models = sorted(regression_results.items(), key=lambda x: x[1]['R2'], reverse=True)[:3]
    
    for model_name, model_results in best_models:
        y_pred = model_results['y_pred']
        
        plot_regression_residuals(y_original, y_pred, model_name, target_column, output_dir, logger)
        
        plot_prediction_vs_actual(y_original, y_pred, model_name, target_column, output_dir, logger)
        
        if 'best_model' in model_results:
            best_model = model_results['best_model']
            plot_learning_curves(best_model, X_original_scaled, y_original, model_name, target_column, output_dir, logger)
            
            plot_feature_importance(best_model, X_original.columns.tolist(), model_name, target_column, output_dir, logger)
    
    plot_cross_validation_scores(current_cv_scores, list(regression_results.keys()), target_column, output_dir, logger)
    
    plot_model_performance_radar(regression_results, target_column, output_dir, logger, max_models=len(regression_results))

    logger.info(f"开始分类模型自动调参: {target_column}")
    classification_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest Classifier': RandomForestClassifier(random_state=42),
        'Extra Trees Classifier': ExtraTreesClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'MLP': MLPClassifier(random_state=42, early_stopping=True, n_iter_no_change=10),
        'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42)
    }
    param_distributions_cls = {
        'Logistic Regression': {'C': uniform(0.01, 10), 'penalty': ['l2']},
        'SVM': {'C': uniform(0.1, 10), 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']},
        'KNN': {'n_neighbors': randint(3, 20)},
        'Decision Tree': {'max_depth': randint(3, 20), 'min_samples_split': randint(2, 10)},
        'Random Forest Classifier': {'n_estimators': randint(50, 200), 'max_depth': randint(3, 20)},
        'Extra Trees Classifier': {'n_estimators': randint(50, 200), 'max_depth': randint(3, 20)},
        'MLP': {'hidden_layer_sizes': [(100,), (100, 50), (50, 25)], 'alpha': uniform(0.0001, 0.01), 'max_iter': [500, 800, 1200, 2000]},
        'Gradient Boosting Classifier': {'learning_rate': uniform(0.001, 0.2), 'n_estimators': randint(50, 200), 'max_depth': randint(3, 10)},
        'AdaBoost': {'n_estimators': randint(30, 100), 'learning_rate': uniform(0.01, 1)},
        'XGBoost': {'learning_rate': uniform(0.001, 0.2), 'n_estimators': randint(50, 200), 'max_depth': randint(3, 10)}
    }
    skip_cls = ['Naive Bayes']

    classification_results = {}
    best_params_cls = {}
    best_scores_cls = {}

    logger.info("第一阶段：n_iter=20 分类模型自动调参")
    for name, model in classification_models.items():
        logger.info(f"处理分类模型: {name}")
        print(f"训练 {name} 用于 {target_column}...")
        if name in skip_cls:
            logger.info(f"  {name}: 使用默认参数")
            y_pred_proba = cross_val_predict(model, X_resampled_scaled, y_resampled, cv=cv, method='predict_proba')
            if y_pred_proba.shape[1] > 2:
                y_resampled_bin = label_binarize(y_resampled, classes=np.unique(y_resampled))
                auc_score = roc_auc_score(y_resampled_bin, y_pred_proba, average='macro', multi_class='ovr')
            else:
                auc_score = roc_auc_score(y_resampled, y_pred_proba[:, 1])
            classification_results[name] = {
                'AUC': auc_score,
                'params': 'default',
                'proba': y_pred_proba
            }
            best_scores_cls[name] = auc_score
            best_params_cls[name] = 'default'
            logger.info(f"  {name}: AUC = {auc_score:.4f} (default)")
            continue
        param_dist = param_distributions_cls.get(name, None)
        if param_dist is None:
            logger.warning(f"  {name}: 未定义参数分布，跳过")
            print(f"  {name}: 未定义参数分布，跳过。")
            continue
        
        logger.info(f"  {name}: 开始RandomizedSearchCV调参 (n_iter=20)")
        search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=cv, scoring='roc_auc_ovr' if len(np.unique(y_resampled)) > 2 else 'roc_auc', n_jobs=-1, random_state=42)
        search.fit(X_resampled_scaled, y_resampled)
        best_model = search.best_estimator_
        y_pred_proba = cross_val_predict(best_model, X_resampled_scaled, y_resampled, cv=cv, method='predict_proba')
        if y_pred_proba.shape[1] > 2:
            y_resampled_bin = label_binarize(y_resampled, classes=np.unique(y_resampled))
            auc_score = roc_auc_score(y_resampled_bin, y_pred_proba, average='macro', multi_class='ovr')
        else:
            auc_score = roc_auc_score(y_resampled, y_pred_proba[:, 1])
        classification_results[name] = {
            'AUC': auc_score,
            'params': search.best_params_,
            'proba': y_pred_proba,
            'best_model': best_model
        }
        best_scores_cls[name] = auc_score
        best_params_cls[name] = search.best_params_
        logger.info(f"  {name}: AUC = {auc_score:.4f}, 最佳参数 = {search.best_params_}")
        print(f"  {name}: AUC = {auc_score:.4f}, params={search.best_params_}")

    top3_cls = sorted(best_scores_cls.items(), key=lambda x: x[1], reverse=True)[:3]
    logger.info(f"第二阶段：top3分类模型精细调参 (n_iter=30)")
    logger.info(f"Top3分类模型: {[name for name, _ in top3_cls]}")
    
    for name, _ in top3_cls:
        if name in skip_cls:
            logger.info(f"  {name}: 跳过精细调参（使用默认参数）")
            continue
        model = classification_models[name]
        param_dist = param_distributions_cls.get(name, None)
        if param_dist is None:
            logger.warning(f"  {name}: 参数分布未定义，跳过精细调参")
            continue
        logger.info(f"  {name}: 开始精细调参 (n_iter=30)")
        print(f"  {name}: 进入top3，n_iter=30精细调参...")
        search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=30, cv=cv, scoring='roc_auc_ovr' if len(np.unique(y_resampled)) > 2 else 'roc_auc', n_jobs=-1, random_state=42)
        search.fit(X_resampled_scaled, y_resampled)
        best_model = search.best_estimator_
        y_pred_proba = cross_val_predict(best_model, X_resampled_scaled, y_resampled, cv=cv, method='predict_proba')
        if y_pred_proba.shape[1] > 2:
            y_resampled_bin = label_binarize(y_resampled, classes=np.unique(y_resampled))
            auc_score = roc_auc_score(y_resampled_bin, y_pred_proba, average='macro', multi_class='ovr')
        else:
            auc_score = roc_auc_score(y_resampled, y_pred_proba[:, 1])
        classification_results[name] = {
            'AUC': auc_score,
            'params': search.best_params_,
            'proba': y_pred_proba,
            'best_model': best_model
        }
        best_scores_cls[name] = auc_score
        best_params_cls[name] = search.best_params_
        logger.info(f"  {name}: (精细调参) AUC = {auc_score:.4f}, 最佳参数 = {search.best_params_}")
        print(f"  {name}: (top3精调) AUC = {auc_score:.4f}, params={search.best_params_}")

    logger.info(f"分类模型调参完成，最终结果:")
    for name, score in sorted(best_scores_cls.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {name}: AUC = {score:.4f}, 参数 = {best_params_cls[name]}")

    y_pred_proba_all = {name: classification_results[name]['proba'] for name in classification_results}

    logger.info(f"开始生成详细的分类分析图: {target_column}")
    
    classification_cv_scores = {}
    
    for model_name, model_results in classification_results.items():
        if 'best_model' in model_results:
            best_model = model_results['best_model']
            
            cv_scores = cross_val_score(best_model, X_resampled_scaled, y_resampled, cv=cv, 
                                      scoring='roc_auc_ovr' if len(np.unique(y_resampled)) > 2 else 'roc_auc')
            classification_cv_scores[model_name] = cv_scores
    
    best_classification_models = sorted(classification_results.items(), key=lambda x: x[1]['AUC'], reverse=True)[:3]
    
    for model_name, model_results in best_classification_models:
        if 'best_model' in model_results:
            best_model = model_results['best_model']
            
            y_pred_class = cross_val_predict(best_model, X_resampled_scaled, y_resampled, cv=cv, method='predict')
            
            class_names = [f'类别{i}' for i in np.unique(y_resampled)]
            plot_classification_confusion_matrix(y_resampled, y_pred_class, model_name, target_column, output_dir, logger, class_names)
    
    plot_classification_metrics_comparison(classification_results, target_column, output_dir, logger)
    
    plot_precision_recall_curves(y_resampled, classification_results, target_column, output_dir, logger)
    
    plot_classification_feature_importance(best_classification_models, X_original.columns.tolist(), target_column, output_dir, logger)
    
    plot_classification_cv_scores_distribution(classification_cv_scores, target_column, output_dir, logger)
    
    plot_classification_performance_radar(classification_results, target_column, output_dir, logger, max_models=len(classification_results))

    all_classification_results[target_column] = classification_results

    logger.info(f"开始Top3回归模型SHAP分析: {target_column}")
    
    top3_regression_models = sorted(regression_results.items(), key=lambda x: x[1]['R2'], reverse=True)[:3]
    logger.info(f"Top3回归模型: {[name for name, _ in top3_regression_models]}")
    print(f"开始为Top3回归模型生成SHAP分析: {[name for name, _ in top3_regression_models]}")
    
    for model_name, model_data in top3_regression_models:
        logger.info(f"开始SHAP分析: {model_name} (R^2 = {model_data['R2']:.4f})")
        print(f"  正在处理 {model_name} 的SHAP分析...")
        
        shap_model_dir = os.path.join(output_dir, f'SHAP_Analysis_{target_column}_{model_name.replace(" ", "_")}')
        os.makedirs(shap_model_dir, exist_ok=True)
        
        try:
            if 'best_model' in model_data:
                best_model_for_shap = model_data['best_model']
            else:
                logger.warning(f"未找到 {model_name} 的保存模型，使用默认参数重新训练")
                if model_name == 'Linear Regression':
                    best_model_for_shap = LinearRegression()
                elif model_name == 'Ridge Regression':
                    best_model_for_shap = Ridge(alpha=model_data.get('params', {}).get('alpha', 1.0), random_state=42)
                elif model_name == 'Lasso Regression':
                    best_model_for_shap = Lasso(alpha=model_data.get('params', {}).get('alpha', 0.1), random_state=42)
                elif model_name == 'ElasticNet':
                    params = model_data.get('params', {})
                    best_model_for_shap = ElasticNet(
                        alpha=params.get('alpha', 0.1), 
                        l1_ratio=params.get('l1_ratio', 0.5), 
                        random_state=42
                    )
                elif model_name == 'SVR':
                    params = model_data.get('params', {})
                    best_model_for_shap = SVR(
                        C=params.get('C', 1.0),
                        gamma=params.get('gamma', 'scale'),
                        kernel=params.get('kernel', 'rbf')
                    )
                else:
                    best_model_for_shap = GradientBoostingRegressor(learning_rate=0.01, max_depth=5, random_state=42)
                
                best_model_for_shap.fit(X_original_scaled, y_original)

            logger.info(f"计算 {model_name} 的SHAP值...")
            explainer = shap.Explainer(best_model_for_shap, X_original_scaled)
            shap_values = explainer(X_original_scaled)

            if hasattr(shap_values, 'feature_names') and shap_values.feature_names is None:
                shap_values.feature_names = X_original.columns.tolist()

            logger.info(f"生成 {model_name} SHAP条形图...")
            plt.figure(figsize=(10, 7))
            shap.summary_plot(shap_values, feature_names=X_original.columns, plot_type="bar", show=False)
            plt.title(f'SHAP特征重要性（条形图）: {target_column} - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(shap_model_dir, f'{target_column}_{model_name}_shap_bar_plot.pdf'))
            plt.close()

            logger.info(f"生成 {model_name} SHAP小提琴图...")
            plt.figure(figsize=(10, 7))
            shap.summary_plot(shap_values, plot_type="layered_violin", show=False)
            plt.title(f'SHAP特征重要性（小提琴图）: {target_column} - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(shap_model_dir, f'{target_column}_{model_name}_shap_beeswarm_plot.pdf'))
            plt.close()

            logger.info(f"生成 {model_name} SHAP依赖图...")
            mean_abs_shap_values = np.abs(shap_values.values).mean(0)
            top_feature_indices = np.argsort(mean_abs_shap_values)[::-1][:4]
            top_features = [X_original.columns[i] for i in top_feature_indices]
            logger.info(f"{model_name} 前4重要特征: {top_features}")

            for feature in top_features:
                logger.info(f"生成 {model_name} SHAP依赖图: {feature}")
                print(f"    生成SHAP依赖图: {feature} ({model_name})")
                feature_index = X_original.columns.get_loc(feature)
                shap_vals_feature = shap_values[:, feature_index].values
                feature_values = X_original[feature].values  # 用原始特征值作横轴

                plt.figure(figsize=(10, 6))
                plt.scatter(feature_values, shap_vals_feature, color='steelblue', alpha=0.6, label='SHAP值')
                if len(np.unique(feature_values)) > 1:
                    try:
                        lowess = sm.nonparametric.lowess(shap_vals_feature, feature_values, frac=0.3)
                        x_lowess, y_lowess = zip(*lowess)
                        plt.plot(x_lowess, y_lowess, color='red', linewidth=2, label='LOWESS曲线')
                        if len(y_lowess) > 1:
                            intersection_indices = np.where(np.diff(np.sign(np.array(y_lowess))))[0]
                            for idx in intersection_indices:
                                if idx + 1 < len(x_lowess):
                                    plt.axvline(x=x_lowess[idx], color='blue', linestyle='--', alpha=0.7)
                    except Exception as e:
                        logger.warning(f"LOWESS拟合失败: {feature} in {model_name}: {e}")
                        print(f"    LOWESS拟合失败: {feature} in {model_name}: {e}")
                else:
                    logger.warning(f"特征 {feature} 唯一值过少，无法LOWESS拟合")
                    print(f"    特征 {feature} 唯一值过少，无法LOWESS拟合。")
                plt.xlabel(feature)
                plt.ylabel('SHAP值')
                plt.title(f'SHAP依赖图: {feature} ({target_column} - {model_name})')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(shap_model_dir, f'{target_column}_{model_name}_{feature}_shap_dependence_plot.pdf'))
                plt.close()
            
            logger.info(f"{model_name} SHAP分析完成，结果保存在: {shap_model_dir}")
            
        except Exception as e:
            logger.error(f"{model_name} SHAP分析失败: {e}")
            print(f"  {model_name} SHAP分析失败: {e}")
            continue

    logger.info(f"生成ROC曲线: {target_column}")
    print(f"生成ROC曲线: {target_column}...")
    plt.figure(figsize=(12, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='随机 (AUC = 0.50)')

    model_colors = {
        'Logistic Regression': '#1f77b4',
        'SVM': '#ff7f0e', 
        'KNN': '#2ca02c',
        'Decision Tree': '#d62728',
        'Random Forest Classifier': '#9467bd',
        'Extra Trees Classifier': '#8c564b',
        'Naive Bayes': '#e377c2',
        'MLP': '#7f7f7f',
        'Gradient Boosting Classifier': '#bcbd22',
        'AdaBoost': '#17becf',
        'XGBoost': '#ff9896'
    }
    
    line_styles = ['-', '--', '-.', ':']

    if y_true_all_current_target.ndim > 1 and y_true_all_current_target.shape[1] > 1:
        n_classes = y_true_all_current_target.shape[1]
        logger.info(f"多分类ROC曲线，类别数: {n_classes}")
        for name, y_proba in y_pred_proba_all.items():
            color = model_colors.get(name, '#000000')  # 默认黑色
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_all_current_target[:, i], y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                linestyle = line_styles[i % len(line_styles)]
                plt.plot(fpr[i], tpr[i], color=color, linestyle=linestyle, linewidth=2,
                        label=f'{name} (类别{i}, AUC = {roc_auc[i]:.2f})')
                logger.info(f"  {name} 类别{i}: AUC = {roc_auc[i]:.4f}")
    else:  # 二分类情况
        logger.info("二分类ROC曲线")
        for idx, (name, y_proba) in enumerate(y_pred_proba_all.items()):
            color = model_colors.get(name, '#000000')  # 默认黑色
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                y_proba_binary = y_proba[:, 1]  # 使用正类的概率
            else:
                y_proba_binary = y_proba.flatten() if y_proba.ndim > 1 else y_proba
            
            fpr, tpr, _ = roc_curve(y_true_all_current_target, y_proba_binary)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC = {roc_auc:.2f})')
            logger.info(f"  {name}: AUC = {roc_auc:.4f}")

    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title(f'ROC曲线: {target_column}')
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_column}_roc_curves.pdf'))
    plt.close()

def plot_all_targets_regression_summary(all_regression_results, output_dir, logger):
    """绘制所有目标的回归模型性能汇总图"""
    logger.info("生成所有目标的回归模型性能汇总图...")
    
    targets = list(all_regression_results.keys())
    all_models = set()
    for target_results in all_regression_results.values():
        all_models.update(target_results.keys())
    all_models = list(all_models)
    
    metrics = ['R2', 'RMSE', 'MAE']
    
    for metric in metrics:
        plt.figure(figsize=(14, 8))
        
        heatmap_data = []
        for model in all_models:
            row = []
            for target in targets:
                if model in all_regression_results[target]:
                    row.append(all_regression_results[target][model][metric])
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        heatmap_data = np.array(heatmap_data)
        
        sns.heatmap(heatmap_data, 
                    xticklabels=targets, 
                    yticklabels=all_models, 
                    annot=True, 
                    fmt='.3f', 
                    cmap='RdYlBu_r' if metric in ['RMSE', 'MAE'] else 'RdYlBu',
                    cbar_kws={'label': f'{metric}分数'})
        
        plt.title(f'回归模型{metric}分数热力图')
        plt.xlabel('目标变量')
        plt.ylabel('回归模型')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'all_targets_{metric}_heatmap.pdf'))
        plt.close()
        logger.info(f"回归模型{metric}热力图已保存")

def plot_model_ranking_across_targets(all_regression_results, output_dir, logger):
    """绘制模型在所有目标上的排名图"""
    logger.info("生成模型在所有目标上的排名图...")
    
    targets = list(all_regression_results.keys())
    all_models = set()
    for target_results in all_regression_results.values():
        all_models.update(target_results.keys())
    all_models = list(all_models)
    
    model_avg_rank = {}
    model_avg_r2 = {}
    
    for model in all_models:
        ranks = []
        r2_scores = []
        for target in targets:
            if model in all_regression_results[target]:
                target_r2_scores = [(m, all_regression_results[target][m]['R2']) 
                                   for m in all_regression_results[target]]
                target_r2_scores.sort(key=lambda x: x[1], reverse=True)
                
                for rank, (m, _) in enumerate(target_r2_scores, 1):
                    if m == model:
                        ranks.append(rank)
                        break
                
                r2_scores.append(all_regression_results[target][model]['R2'])
        
        if ranks:
            model_avg_rank[model] = np.mean(ranks)
            model_avg_r2[model] = np.mean(r2_scores)
    
    sorted_models = sorted(model_avg_rank.items(), key=lambda x: x[1])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    model_names = [item[0] for item in sorted_models]
    avg_ranks = [item[1] for item in sorted_models]
    
    bars1 = ax1.barh(range(len(model_names)), avg_ranks, color='skyblue', alpha=0.8)
    ax1.set_yticks(range(len(model_names)))
    ax1.set_yticklabels(model_names)
    ax1.set_xlabel('平均排名')
    ax1.set_title('回归模型平均排名（越小越好）')
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, rank) in enumerate(zip(bars1, avg_ranks)):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{rank:.2f}', va='center', fontweight='bold')
    
    sorted_by_r2 = sorted(model_avg_r2.items(), key=lambda x: x[1], reverse=True)
    model_names_r2 = [item[0] for item in sorted_by_r2]
    avg_r2s = [item[1] for item in sorted_by_r2]
    
    bars2 = ax2.barh(range(len(model_names_r2)), avg_r2s, color='lightgreen', alpha=0.8)
    ax2.set_yticks(range(len(model_names_r2)))
    ax2.set_yticklabels(model_names_r2)
    ax2.set_xlabel('平均R$^2$分数')
    ax2.set_title('回归模型平均R$^2$分数（越大越好）')
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, r2) in enumerate(zip(bars2, avg_r2s)):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{r2:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_ranking_across_targets.pdf'))
    plt.close()
    logger.info("模型排名图已保存")

def plot_classification_vs_regression_performance(all_regression_results, all_classification_results, output_dir, logger):
    """绘制分类算法vs回归算法性能对比"""
    logger.info("开始生成分类vs回归算法性能对比图...")
    
    targets = list(all_regression_results.keys())
    
    comparison_data = []
    
    for target in targets:
        if target in all_classification_results:
            reg_results = all_regression_results[target]
            cls_results = all_classification_results[target]
            
            common_algorithms = []
            for reg_name in reg_results.keys():
                cls_name = None
                if 'Linear Regression' in reg_name:
                    cls_name = 'Logistic Regression'
                elif 'SVM' in reg_name or 'SVR' in reg_name:
                    cls_name = 'SVM'
                elif 'Random Forest' in reg_name:
                    cls_name = 'Random Forest Classifier'
                elif 'Decision Tree' in reg_name:
                    cls_name = 'Decision Tree'
                elif 'Gradient Boosting' in reg_name:
                    cls_name = 'Gradient Boosting Classifier'
                elif 'AdaBoost' in reg_name:
                    cls_name = 'AdaBoost'
                elif 'XGBoost' in reg_name:
                    cls_name = 'XGBoost'
                elif 'MLP' in reg_name:
                    cls_name = 'MLP'
                elif 'KNN' in reg_name:
                    cls_name = 'KNN'
                elif 'Extra Trees' in reg_name:
                    cls_name = 'Extra Trees Classifier'
                
                if cls_name and cls_name in cls_results:
                    common_algorithms.append({
                        'target': target,
                        'algorithm': reg_name.replace(' Regressor', '').replace(' Regression', ''),
                        'regression_r2': reg_results[reg_name]['R2'],
                        'regression_rmse': reg_results[reg_name]['RMSE'],
                        'classification_auc': cls_results[cls_name]['AUC']
                    })
    
    if not comparison_data and common_algorithms:
        comparison_data = common_algorithms
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        for target in targets:
            target_data = df[df['target'] == target]
            if not target_data.empty:
                ax1.scatter(target_data['regression_r2'], target_data['classification_auc'], 
                           label=target, s=60, alpha=0.7)
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='完美相关线')
        ax1.set_xlabel('回归模型 R$^2$ 分数')
        ax1.set_ylabel('分类模型 AUC 分数')
        ax1.set_title('回归 R$^2$ vs 分类 AUC 性能对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if len(df) > 0:
            avg_performance = df.groupby('algorithm').agg({
                'regression_r2': 'mean',
                'classification_auc': 'mean'
            }).reset_index()
            
            x = np.arange(len(avg_performance))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, avg_performance['regression_r2'], width, 
                           label='回归 R$^2$', alpha=0.8, color='skyblue')
            bars2 = ax2.bar(x + width/2, avg_performance['classification_auc'], width, 
                           label='分类 AUC', alpha=0.8, color='lightcoral')
            
            ax2.set_xlabel('算法类型')
            ax2.set_ylabel('平均性能分数')
            ax2.set_title('不同算法的回归vs分类平均性能')
            ax2.set_xticks(x)
            ax2.set_xticklabels(avg_performance['algorithm'], rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            for bar in bars1:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        pivot_r2 = df.pivot(index='algorithm', columns='target', values='regression_r2')
        sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='Blues', ax=ax3, cbar_kws={'label': 'R$^2$ 分数'})
        ax3.set_title('回归模型 R$^2$ 分数热力图')
        ax3.set_xlabel('目标变量')
        ax3.set_ylabel('算法')
        
        pivot_auc = df.pivot(index='algorithm', columns='target', values='classification_auc')
        sns.heatmap(pivot_auc, annot=True, fmt='.3f', cmap='Reds', ax=ax4, cbar_kws={'label': 'AUC 分数'})
        ax4.set_title('分类模型 AUC 分数热力图')
        ax4.set_xlabel('目标变量')
        ax4.set_ylabel('算法')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'classification_vs_regression_performance_comparison.pdf'))
        plt.close()
        logger.info("分类vs回归性能对比图已保存")
    else:
        logger.warning("没有找到可对比的分类和回归算法数据")

def plot_algorithm_consistency_analysis(all_regression_results, all_classification_results, output_dir, logger):
    """分析算法在不同任务上的一致性表现"""
    logger.info("开始生成算法一致性分析图...")
    
    targets = list(all_regression_results.keys())
    
    algorithm_mapping = {
        'Linear Regression': 'Logistic Regression',
        'SVR': 'SVM',
        'Random Forest': 'Random Forest Classifier',
        'Decision Tree': 'Decision Tree',
        'Gradient Boosting': 'Gradient Boosting Classifier',
        'AdaBoost': 'AdaBoost',
        'XGBoost': 'XGBoost',
        'MLP Regressor': 'MLP',
        'KNN Regressor': 'KNN',
        'Extra Trees': 'Extra Trees Classifier'
    }
    
    consistency_data = []
    
    for target in targets:
        if target in all_classification_results:
            reg_results = all_regression_results[target]
            cls_results = all_classification_results[target]
            
            reg_ranked = sorted(reg_results.items(), key=lambda x: x[1]['R2'], reverse=True)
            cls_ranked = sorted(cls_results.items(), key=lambda x: x[1]['AUC'], reverse=True)
            
            reg_ranks = {name: rank+1 for rank, (name, _) in enumerate(reg_ranked)}
            cls_ranks = {name: rank+1 for rank, (name, _) in enumerate(cls_ranked)}
            
            for reg_name, cls_name in algorithm_mapping.items():
                if reg_name in reg_ranks and cls_name in cls_ranks:
                    consistency_data.append({
                        'target': target,
                        'algorithm': reg_name.replace(' Regressor', '').replace(' Regression', ''),
                        'regression_rank': reg_ranks[reg_name],
                        'classification_rank': cls_ranks[cls_name],
                        'rank_diff': abs(reg_ranks[reg_name] - cls_ranks[cls_name])
                    })
    
    if consistency_data:
        df = pd.DataFrame(consistency_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(df['algorithm'].unique())))
        algorithm_colors = dict(zip(df['algorithm'].unique(), colors))
        
        for algorithm in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == algorithm]
            ax1.scatter(alg_data['regression_rank'], alg_data['classification_rank'], 
                       color=algorithm_colors[algorithm], label=algorithm, s=80, alpha=0.7)
        
        max_rank = max(df['regression_rank'].max(), df['classification_rank'].max())
        ax1.plot([1, max_rank], [1, max_rank], 'k--', alpha=0.5, label='完美一致性')
        
        ax1.set_xlabel('回归任务排名')
        ax1.set_ylabel('分类任务排名')
        ax1.set_title('算法在回归vs分类任务中的排名一致性')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        ax1.invert_xaxis()
        
        sns.boxplot(data=df, x='algorithm', y='rank_diff', ax=ax2, palette='viridis')
        ax2.set_xlabel('算法')
        ax2.set_ylabel('排名差异 (绝对值)')
        ax2.set_title('算法排名一致性差异分布')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'algorithm_consistency_analysis.pdf'))
        plt.close()
        logger.info("算法一致性分析图已保存")

def plot_task_suitability_analysis(all_regression_results, all_classification_results, output_dir, logger):
    """分析不同算法对不同任务的适用性"""
    logger.info("开始生成任务适用性分析图...")
    
    targets = list(all_regression_results.keys())
    
    algorithm_performance = {}
    
    for target in targets:
        if target in all_classification_results:
            reg_results = all_regression_results[target]
            cls_results = all_classification_results[target]
            
            for reg_name, reg_data in reg_results.items():
                alg_base = reg_name.replace(' Regressor', '').replace(' Regression', '')
                if alg_base not in algorithm_performance:
                    algorithm_performance[alg_base] = {
                        'regression_scores': [],
                        'classification_scores': [],
                        'targets': []
                    }
                
                algorithm_performance[alg_base]['regression_scores'].append(reg_data['R2'])
                algorithm_performance[alg_base]['targets'].append(target)
                
                cls_name = None
                if 'Linear' in reg_name:
                    cls_name = 'Logistic Regression'
                elif 'SVM' in reg_name or 'SVR' in reg_name:
                    cls_name = 'SVM'
                elif 'Random Forest' in reg_name:
                    cls_name = 'Random Forest Classifier'
                elif 'Decision Tree' in reg_name:
                    cls_name = 'Decision Tree'
                elif 'Gradient Boosting' in reg_name:
                    cls_name = 'Gradient Boosting Classifier'
                elif 'AdaBoost' in reg_name:
                    cls_name = 'AdaBoost'
                elif 'XGBoost' in reg_name:
                    cls_name = 'XGBoost'
                elif 'MLP' in reg_name:
                    cls_name = 'MLP'
                elif 'KNN' in reg_name:
                    cls_name = 'KNN'
                elif 'Extra Trees' in reg_name:
                    cls_name = 'Extra Trees Classifier'
                
                if cls_name and cls_name in cls_results:
                    algorithm_performance[alg_base]['classification_scores'].append(cls_results[cls_name]['AUC'])
                else:
                    algorithm_performance[alg_base]['classification_scores'].append(0.5)  # 默认值
    
    suitability_data = []
    for alg, data in algorithm_performance.items():
        if data['regression_scores'] and data['classification_scores']:
            reg_mean = np.mean(data['regression_scores'])
            reg_std = np.std(data['regression_scores'])
            cls_mean = np.mean(data['classification_scores'])
            cls_std = np.std(data['classification_scores'])
            
            suitability_data.append({
                'algorithm': alg,
                'regression_mean': reg_mean,
                'regression_std': reg_std,
                'classification_mean': cls_mean,
                'classification_std': cls_std,
                'regression_stability': 1 / (1 + reg_std),  # 稳定性指标
                'classification_stability': 1 / (1 + cls_std),
                'overall_performance': (reg_mean + cls_mean) / 2,
                'task_versatility': 1 - abs(reg_mean - cls_mean)  # 任务适应性
            })
    
    if suitability_data:
        df = pd.DataFrame(suitability_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        scatter = ax1.scatter(df['overall_performance'], df['regression_stability'], 
                            s=df['task_versatility']*500, alpha=0.6, c=df['classification_stability'], 
                            cmap='viridis')
        
        for i, row in df.iterrows():
            ax1.annotate(row['algorithm'], (row['overall_performance'], row['regression_stability']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax1.set_xlabel('综合性能 (R$^2$ + AUC) / 2')
        ax1.set_ylabel('回归稳定性')
        ax1.set_title('算法性能vs稳定性分析\n(气泡大小=任务适应性, 颜色=分类稳定性)')
        plt.colorbar(scatter, ax=ax1, label='分类稳定性')
        ax1.grid(True, alpha=0.3)
        
        top_algorithms = df.nlargest(6, 'overall_performance')
        
        metrics = ['regression_mean', 'classification_mean', 'regression_stability', 
                  'classification_stability', 'task_versatility']
        metric_labels = ['回归性能', '分类性能', '回归稳定性', '分类稳定性', '任务适应性']
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        ax2 = plt.subplot(2, 2, 2, projection='polar')
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_algorithms)))
        
        for idx, (_, row) in enumerate(top_algorithms.iterrows()):
            values = [row[metric] for metric in metrics]
            values += values[:1]
            ax2.plot(angles, values, 'o-', linewidth=2, label=row['algorithm'], color=colors[idx])
            ax2.fill(angles, values, alpha=0.1, color=colors[idx])
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metric_labels)
        ax2.set_ylim(0, 1)
        ax2.set_title('Top 6 算法综合能力雷达图', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        df_sorted = df.sort_values('task_versatility', ascending=True)
        bars = ax3.barh(range(len(df_sorted)), df_sorted['task_versatility'], color='lightgreen', alpha=0.8)
        ax3.set_yticks(range(len(df_sorted)))
        ax3.set_yticklabels(df_sorted['algorithm'])
        ax3.set_xlabel('任务适应性分数')
        ax3.set_title('算法任务适应性排名')
        ax3.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, score) in enumerate(zip(bars, df_sorted['task_versatility'])):
            ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontweight='bold')
        
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, df['regression_stability'], width, 
                       label='回归稳定性', alpha=0.8, color='skyblue')
        bars2 = ax4.bar(x + width/2, df['classification_stability'], width, 
                       label='分类稳定性', alpha=0.8, color='lightcoral')
        
        ax4.set_xlabel('算法')
        ax4.set_ylabel('稳定性分数')
        ax4.set_title('算法稳定性对比')
        ax4.set_xticks(x)
        ax4.set_xticklabels(df['algorithm'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'task_suitability_analysis.pdf'))
        plt.close()
        logger.info("任务适用性分析图已保存")

def generate_classification_regression_summary_report(all_regression_results, all_classification_results, output_dir, logger):
    """生成分类vs回归算法对比总结报告"""
    logger.info("生成分类vs回归算法对比总结报告...")
    
    targets = list(all_regression_results.keys())
    report_data = {
        'targets': targets,
        'regression_summary': {},
        'classification_summary': {},
        'best_algorithms': {},
        'consistency_analysis': {}
    }
    
    for target in targets:
        if target in all_classification_results:
            reg_results = all_regression_results[target]
            cls_results = all_classification_results[target]
            
            best_reg = max(reg_results.items(), key=lambda x: x[1]['R2'])
            best_cls = max(cls_results.items(), key=lambda x: x[1]['AUC'])
            
            report_data['best_algorithms'][target] = {
                'regression': {'name': best_reg[0], 'r2': best_reg[1]['R2']},
                'classification': {'name': best_cls[0], 'auc': best_cls[1]['AUC']}
            }
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    reg_winners = [data['regression']['name'] for data in report_data['best_algorithms'].values()]
    cls_winners = [data['classification']['name'] for data in report_data['best_algorithms'].values()]
    
    from collections import Counter
    reg_counter = Counter(reg_winners)
    cls_counter = Counter(cls_winners)
    
    reg_algorithms = list(reg_counter.keys())
    reg_counts = list(reg_counter.values())
    
    bars1 = ax1.bar(range(len(reg_algorithms)), reg_counts, color='skyblue', alpha=0.8)
    ax1.set_xlabel('回归算法')
    ax1.set_ylabel('最佳表现次数')
    ax1.set_title('回归算法最佳表现频次统计')
    ax1.set_xticks(range(len(reg_algorithms)))
    ax1.set_xticklabels(reg_algorithms, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    for bar, count in zip(bars1, reg_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    cls_algorithms = list(cls_counter.keys())
    cls_counts = list(cls_counter.values())
    
    bars2 = ax2.bar(range(len(cls_algorithms)), cls_counts, color='lightcoral', alpha=0.8)
    ax2.set_xlabel('分类算法')
    ax2.set_ylabel('最佳表现次数')
    ax2.set_title('分类算法最佳表现频次统计')
    ax2.set_xticks(range(len(cls_algorithms)))
    ax2.set_xticklabels(cls_algorithms, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    for bar, count in zip(bars2, cls_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    all_r2_scores = []
    all_auc_scores = []
    
    for target in targets:
        if target in all_classification_results:
            reg_scores = [result['R2'] for result in all_regression_results[target].values()]
            auc_scores = [result['AUC'] for result in all_classification_results[target].values()]
            all_r2_scores.extend(reg_scores)
            all_auc_scores.extend(auc_scores)
    
    ax3.hist(all_r2_scores, bins=20, alpha=0.7, color='skyblue', label=f'回归 R$^2$ (均值={np.mean(all_r2_scores):.3f})', density=True)
    ax3.hist(all_auc_scores, bins=20, alpha=0.7, color='lightcoral', label=f'分类 AUC (均值={np.mean(all_auc_scores):.3f})', density=True)
    ax3.set_xlabel('性能分数')
    ax3.set_ylabel('密度')
    ax3.set_title('回归R$^2$ vs 分类AUC 性能分布对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    target_names = list(report_data['best_algorithms'].keys())
    reg_best_scores = [data['regression']['r2'] for data in report_data['best_algorithms'].values()]
    cls_best_scores = [data['classification']['auc'] for data in report_data['best_algorithms'].values()]
    
    x = np.arange(len(target_names))
    width = 0.35
    
    bars3 = ax4.bar(x - width/2, reg_best_scores, width, label='回归最佳R$^2$', alpha=0.8, color='skyblue')
    bars4 = ax4.bar(x + width/2, cls_best_scores, width, label='分类最佳AUC', alpha=0.8, color='lightcoral')
    
    ax4.set_xlabel('目标变量')
    ax4.set_ylabel('最佳性能分数')
    ax4.set_title('各目标变量的最佳算法性能对比')
    ax4.set_xticks(x)
    ax4.set_xticklabels(target_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    for bar, score in zip(bars3, reg_best_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    for bar, score in zip(bars4, cls_best_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_regression_summary_report.pdf'))
    plt.close()
    logger.info("分类vs回归算法对比总结报告已保存")
    
    summary_text = f"""

- 分析目标数量: {len(targets)}
- 目标变量: {', '.join(targets)}
- 回归算法数量: {len(set().union(*[list(results.keys()) for results in all_regression_results.values()]))}
- 分类算法数量: {len(set().union(*[list(results.keys()) for results in all_classification_results.values() if results]))}

"""
    
    for alg, count in reg_counter.most_common():
        summary_text += f"- {alg}: {count}次最佳 ({count/len(targets)*100:.1f}%)\n"
    
    summary_text += "\n### 分类算法表现:\n"
    for alg, count in cls_counter.most_common():
        summary_text += f"- {alg}: {count}次最佳 ({count/len(targets)*100:.1f}%)\n"
    
    summary_text += f"""
- 回归模型平均R^2: {np.mean(all_r2_scores):.4f} +/- {np.std(all_r2_scores):.4f}
- 分类模型平均AUC: {np.mean(all_auc_scores):.4f} +/- {np.std(all_auc_scores):.4f}

"""
    
    for target, data in report_data['best_algorithms'].items():
        summary_text += f"### {target}:\n"
        summary_text += f"- 最佳回归: {data['regression']['name']} (R^2 = {data['regression']['r2']:.4f})\n"
        summary_text += f"- 最佳分类: {data['classification']['name']} (AUC = {data['classification']['auc']:.4f})\n\n"
    
    with open(os.path.join(output_dir, 'classification_regression_analysis_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    logger.info("分类vs回归分析文本总结已保存")

plot_all_targets_regression_summary(all_regression_results, output_dir, logger)
plot_model_ranking_across_targets(all_regression_results, output_dir, logger)

logger.info("开始生成分类与回归算法对比分析...")
plot_classification_vs_regression_performance(all_regression_results, all_classification_results, output_dir, logger)
plot_algorithm_consistency_analysis(all_regression_results, all_classification_results, output_dir, logger)
plot_task_suitability_analysis(all_regression_results, all_classification_results, output_dir, logger)
generate_classification_regression_summary_report(all_regression_results, all_classification_results, output_dir, logger)

logger.info("生成回归模型R^2最佳参数对比图...")
print("\n生成回归模型R^2最佳参数对比图...")

targets = list(all_regression_results.keys())
models_list = list(models.keys())
logger.info(f"目标变量数量: {len(targets)}")
logger.info(f"模型数量: {len(models_list)}")

valid_models = []
for model_name in models_list:
    if all(model_name in all_regression_results[target] for target in targets):
        valid_models.append(model_name)
if not valid_models:
    logger.warning("没有任何模型在所有目标下都有R^2分数，无法绘图。")
    print("没有任何模型在所有目标下都有R^2分数，无法绘图。")
else:
    logger.info(f"用于绘图的模型数量: {len(valid_models)}")
    logger.info(f"有效模型: {valid_models}")
    print("用于绘图的模型：", valid_models)

    logger.info("生成回归模型R^2对比柱状图...")
    plt.figure(figsize=(16, 10))
    x = np.arange(len(targets))
    width = 0.06  # 减小宽度以适应更多模型
    multiplier = 0
    for model_name in valid_models:
        r2_values = [all_regression_results[target][model_name]['R2'] for target in targets]
        offset = width * multiplier
        rects = plt.bar(x + offset, r2_values, width, label=model_name, alpha=0.8)
        multiplier += 1
        logger.info(f"  {model_name}: R^2范围 [{min(r2_values):.4f}, {max(r2_values):.4f}], 平均 {np.mean(r2_values):.4f}")
    plt.xlabel('目标变量')
    plt.ylabel('R$^2$ 分数')
    plt.title('回归模型R$^2$分数（最佳参数）对比')
    plt.xticks(x + width * (len(valid_models) - 1) / 2, targets, rotation=45, ha='right')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regression_r2_bestparams_comparison_bar.pdf'))
    plt.close()
    logger.info("回归模型R^2对比图已保存")

logger.info("="*50)
logger.info("机器学习分析完成（增强图表版本）")
logger.info("="*50)
print(f"\n所有分析完成！结果保存在 '{output_dir}' 文件夹中。")
print("包含：详细的回归模型分析图、分类模型ROC曲线、Top3模型SHAP分析、汇总对比图等。")
print("新增图表包括：")
print("- 回归模型残差分析图")
print("- 预测vs实际值散点图")
print("- 学习曲线")
print("- 特征重要性图")
print("- 交叉验证分数分布图")
print("- 模型性能雷达图")
print("- 所有目标的性能热力图")
print("- 模型排名汇总图")
print("- 分类模型详细分析图表")
print("- 分类vs回归算法对比分析")
print("- 算法一致性分析")
print("- 任务适用性分析")
print("- 综合对比总结报告")
print("- Top3回归模型SHAP分析（每个模型单独文件夹）")
