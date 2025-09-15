# 隐写检测与可视化（传统非学习方法｜MATLAB）

本项目实现了一个**不依赖深度学习**的隐写取证可视化工具：
- 对输入图像计算 **Laplacian/Sobel 残差 + 局部统计（方差/熵）** 的组合评分，生成**怀疑度热力图**；
- 支持 **Otsu/分位数阈值** 二值检测；
- 若提供真值 `location_map.png`，自动计算 **ROC/AUC、Precision/Recall/F1** 并导出 CSV。

## 目录
```
rdhei-detect/
├─ examples/      # 放示例 cover/stego/location_map
├─ scripts/       # 一键运行脚本（run_detection.m）
├─ results/       # 运行结果输出（热力图、ROC、CSV 指标）
├─ docs/          # 说明文档、截图
├─ .gitignore
├─ LICENSE
└─ README.md
```

## 快速开始
1. 将你的 `stego.png`（必需）、`cover.png`（可选）、`location_map.png`（可选）放入 `examples/`。
2. 打开 MATLAB，运行：
   ```matlab
   cd scripts
   run_detection
   ```
3. 结果会输出到 `results/`：`heatmap.png`, `overlay.png`, `binary_map.png`, （有真值时）`roc.png`, `metrics.csv`。

## 指标说明
- **AUC**：ROC 曲线下面积，越接近 1.0 越好；
- **precision/recall/F1**：在 Otsu 阈值下的分类指标；
- **roc_points.csv**：TPR/FPR 随阈值的变化表。

## 适配你的数据
- 如需调参，修改 `run_detection.m` 中的权重 `w`、窗口大小、阈值方式（Otsu/分位数）。
- 若你的数据较大，建议在 `run_detection.m` 前部修改路径，批量循环处理。

## 许可
MIT（见 `LICENSE`）。