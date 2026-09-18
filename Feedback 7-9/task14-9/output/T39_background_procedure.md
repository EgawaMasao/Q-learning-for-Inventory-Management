# T39 Background Construction Procedure
Seed: 42
Synthetic: inventory~U(0,1), sales~U(0,1), waste=0.025*inv+N(0,0.005) clip[0,0.1] (XAI/SHAP-temp.ipynb:5)
200 balances diversity and cost. 100 via shap.sample(random,100,seed=42) baseline (XAI:355).
KMeans 100 ablation: KMeans(k=100). Pairwise mean random=0.521, kmeans=0.526.
Conclusion: Both give rho>0.92 in T40, random 100 sufficient.
