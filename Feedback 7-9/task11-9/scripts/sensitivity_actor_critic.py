"""
sensitivity_actor_critic.py - Task 15: Sensitivity actor vs critic
Gộp với common_target_test.py, chạy trên cùng 10 states mẫu

Outputs:
  - a2c_pi_660 (baseline softmax)
  - a2c_logits_660 (pre-softmax)
  - a2c_critic_660 (V(s) scalar mean)

So sánh Jaccard Top-20 giữa 3 outputs.
Xem planTask11-9.md:138-166
"""
import numpy as np, pandas as pd

print("[sensitivity] Script template cho Task 15 - gộp với common_target_test.py")
print("[sensitivity] Wrapper a2c_critic_660: critic(s_pp) -> mean -> [B,1], SHAP trên scalar V(s)")
