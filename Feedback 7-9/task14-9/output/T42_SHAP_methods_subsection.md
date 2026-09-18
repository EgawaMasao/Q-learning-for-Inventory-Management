# T42 Consistent SHAP Methods Subsection
## 3.X SHAP Methods (Consistent across Macro and Micro)
We use two complementary SHAP explainers on the same inventory model, at different granularities (see topk_shap_analysis.ipynb:16). Both use seed=42 and trajectory-aware ablation in T40.

| Aspect | Macro (System-level) | Micro (SKU-level) |
|--------|----------------------|-------------------|
| Goal | Which group drives decision? | Which SKU drives decision? |
| Input | (N,3) aggregated | (N,660) raw |
| Explainer | shap.KernelExplainer | shap.PartitionExplainer |
| Masker | None | shap.maskers.Partition(bg660, max_samples=100) |
| Why chosen | Exact Shapley, feasible for 3 dims | Hierarchical O(N log N), avoids 2^660 |
| Background | 100 from 200 synthetic (seed 42) | 100 from 100/200 synthetic 660-dim (seed 42) |
| Config | (100,3), nsamples~2054, l1_reg=3 | Partition(bg660, max_samples=100) |
| Ref code | XAI/SHAP-temp.ipynb:372 | topk_shap_analysis.ipynb:446 |
