# 酱香白酒香气建模工具

该仓库包含两份核心脚本，用于分析酱香白酒化学成分与香气指标之间的关系。

## 文件说明
- `ml_build_plots.py`：读取训练数据，批量训练多种机器学习模型，执行随机搜索调参，并生成性能对比图与 `saved_models` 目录，便于后续预测。
- `model_predictor.py`：提供 `FlavorPredictor` 工具类，负责加载 `ml_build_plots.py` 生成的模型与缩放器，供其他脚本或服务调用。

## 使用建议
1. 按需准备运行环境（Python 3.8+），安装 `ml_build_plots.py` 开头声明的依赖。
2. 执行 `ml_build_plots.py` 生成模型与可视化结果。
3. 通过 `model_predictor.FlavorPredictor` 读取 `saved_models` 目录，以支持自定义预测流程。

## 版权信息
版权所有 © 2025，保留所有权利。仅供学习与研究使用，未经授权禁止商业用途。
