# C3 指定规则调整表

本表按用户转述的教师要求，对已有路线误差差值作事后单侧数值调整。
这是指定规则下的展示值；设备校准依据未独立核验，不替代原始测量或模型改善率。

原始差值 d = M0 route ADE − M1 route ADE。规则：当 −0.1 m ≤ d < 0 m 时，
调整值为 d + 0.1 m；其他条目保持原值。全部 roots 保留，边界 −0.1 m 调整为 0 m。

本次共 53 个 roots，其中 **19 个加 0.1 m**，34 个保持原值。
指定调整后，正值 43/53（81.13%）、
零值 0 个、负值 10 个。该比例仅描述调整后的符号。

## 本次发生调整的条目

| root | 原始差值（m） | 指定调整值（m） |
|---|---:|---:|
| Town01__free_flow__s157__CloudyNoon | -0.097155 | 0.002845 |
| Town03__free_flow__s157__CloudyNoon | -0.091994 | 0.008006 |
| Town05__red_light_dilemma__s157__CloudyNoon | -0.072102 | 0.027898 |
| Town03__red_light_dilemma__s157__CloudyNoon | -0.061673 | 0.038327 |
| Town05__free_flow__s157__ClearNoon | -0.055437 | 0.044563 |
| Town01__free_flow__s157__ClearNoon | -0.053854 | 0.046146 |
| Town05__red_light_hold__s157__CloudyNoon | -0.053595 | 0.046405 |
| Town05__red_light_dilemma__s157__ClearNoon | -0.045283 | 0.054717 |
| Town01__red_light_dilemma__s157__ClearNoon | -0.043125 | 0.056875 |
| Town01__red_light_dilemma__s157__CloudyNoon | -0.042806 | 0.057194 |
| Town05__free_flow__s157__CloudyNoon | -0.036906 | 0.063094 |
| Town03__cross_traffic_conflict__s157__ClearNoon | -0.036842 | 0.063158 |
| Town03__stopped_lead__s157__ClearNoon | -0.035692 | 0.064308 |
| Town01__red_light_hold__s157__ClearNoon | -0.023982 | 0.076018 |
| Town01__cut_in__s157__ClearNoon | -0.021746 | 0.078254 |
| Town01__aggressive_cut_in__s157__ClearNoon | -0.021349 | 0.078651 |
| Town01__red_light_hold__s157__CloudyNoon | -0.019525 | 0.080475 |
| Town01__slow_lead__s157__CloudyNoon | -0.012914 | 0.087086 |
| Town03__red_light_hold__s157__CloudyNoon | -0.011548 | 0.088452 |

## 来源与复算

原始报告：`generated/h6/cora/c3-school-closeout-20260911T020134Z/independent-analysis-final.json`。
源文件 SHA-256：`f7a7d1d5e4a07d93c9d22b35eaa5babc9f665df33245670e1b3ec152e703f18f`。

完整原始/调整列及逐项应用标记位于：
`generated/h6/cora/c3-display-adjustment-20260911T034032Z/route-adjustment.json`。

同目录 `apply_adjustment.py` 记录实际变换；重复执行会拒绝覆盖已有产物。
原始评估、置信区间、预测和模型权重保持原实验身份。
原始结果见 [C3 学校项目结果页](C3_SCHOOL_RESULTS.md)。
