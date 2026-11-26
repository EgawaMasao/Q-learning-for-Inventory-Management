# Task 1 - SHAP Feature Importance Stability Across 5 A2C_mod Seeds

Generated: 2026-11-25 23:48:24

## 1. Goal

Kiem chung SHAP feature importance co on dinh qua 5 lan huan luyen doc lap (seed 42, 123, 256, 512, 1024) hay khong, cho macro (3 nhom) va micro (660 dac trung), tren 3 scenario va 2 target (pi, logits).

## 2. Configuration

| scenario   |   sales_scale |   waste_rate |
|:-----------|--------------:|-------------:|
| EASY       |        0.5000 |       0.0100 |
| MEDIUM     |        1.0000 |       0.0250 |
| HARD       |        1.5000 |       0.0500 |

- N_STATES = 50; BG_SIZE = 100; BG_SEED = 42
- TARGETS = ['pi', 'logits']; MICRO_TOP_K = [10, 20, 50]; MACRO_TOP_K = [1, 2]
- Micro state: 660-dim [x_init | sales_t*scale | x_init*waste_rate]; macro state: mean of each block.
- Actor: 3 -> 32 -> 32 -> 32 -> 14, dropout=0.0; restore only actor with expect_partial().

## 3. Checkpoints Used

|   seed | checkpoint   |   step |   episode | checkpoint_dir                                                                       |
|-------:|:-------------|-------:|----------:|:-------------------------------------------------------------------------------------|
|     42 | ckpt-53      |  17490 |       530 | C:\GitHub\Q-learning-for-Inventory-Management\Feedback 23-11\checkpoints_a2c_42      |
|    123 | ckpt-60      |  19800 |       600 | C:\GitHub\Q-learning-for-Inventory-Management\Feedback 23-11\A2C_mod_123\checkpoints |
|    256 | ckpt-60      |  19800 |       600 | C:\GitHub\Q-learning-for-Inventory-Management\Feedback 23-11\A2C_mod_256\checkpoints |
|    512 | ckpt-60      |  19800 |       600 | C:\GitHub\Q-learning-for-Inventory-Management\Feedback 23-11\checkpoints_a2c_512     |
|   1024 | ckpt-60      |  19800 |       600 | C:\GitHub\Q-learning-for-Inventory-Management\Feedback 23-11\checkpoints_a2c_1024    |

## 4. Collapse Diagnostics

|   seed | scenario   |   mean_action |   frac_action_13 |   mean_entropy |   mean_max_prob |
|-------:|:-----------|--------------:|-----------------:|---------------:|----------------:|
|     42 | EASY       |        8.7940 |           0.0000 |         0.0007 |          0.9998 |
|     42 | MEDIUM     |        8.9780 |           0.0000 |         0.0008 |          0.9996 |
|     42 | HARD       |        9.1060 |           0.0000 |         0.0004 |          0.9998 |
|    123 | EASY       |       11.1580 |           0.8325 |         0.0132 |          0.9949 |
|    123 | MEDIUM     |       11.1140 |           0.8285 |         0.0107 |          0.9960 |
|    123 | HARD       |       11.1407 |           0.8307 |         0.0110 |          0.9960 |
|    256 | EASY       |       12.4241 |           0.8848 |         0.0096 |          0.9950 |
|    256 | MEDIUM     |       12.4350 |           0.8870 |         0.0095 |          0.9944 |
|    256 | HARD       |       12.3850 |           0.8770 |         0.0093 |          0.9945 |
|    512 | EASY       |        9.3717 |           0.0000 |         0.0069 |          0.9966 |
|    512 | MEDIUM     |        9.4320 |           0.0000 |         0.0025 |          0.9992 |
|    512 | HARD       |        9.5176 |           0.0000 |         0.0015 |          0.9993 |
|   1024 | EASY       |       10.4480 |           0.6810 |         0.0266 |          0.9873 |
|   1024 | MEDIUM     |       10.3382 |           0.6673 |         0.0140 |          0.9949 |
|   1024 | HARD       |       10.1600 |           0.6450 |         0.0086 |          0.9972 |

### Extra checkpoints

|   seed | checkpoint   | scenario   |   mean_action |   frac_action_13 |   mean_entropy |   mean_max_prob |
|-------:|:-------------|:-----------|--------------:|-----------------:|---------------:|----------------:|
|    256 | ckpt-76      | EASY       |       12.3764 |           0.8753 |         0.0062 |          0.9979 |
|    256 | ckpt-76      | MEDIUM     |       12.3314 |           0.8663 |         0.0070 |          0.9973 |
|    256 | ckpt-76      | HARD       |       12.2600 |           0.8520 |         0.0069 |          0.9975 |
|   1024 | ckpt-64      | EASY       |       10.6393 |           0.7049 |         0.0199 |          0.9915 |
|   1024 | ckpt-64      | MEDIUM     |       10.5396 |           0.6925 |         0.0101 |          0.9960 |
|   1024 | ckpt-64      | HARD       |       10.3636 |           0.6705 |         0.0220 |          0.9892 |
|     42 | ckpt-50      | EASY       |        6.8420 |           0.0000 |         0.0036 |          0.9982 |
|     42 | ckpt-50      | MEDIUM     |        7.0680 |           0.0000 |         0.0020 |          0.9994 |
|     42 | ckpt-50      | HARD       |        7.3390 |           0.0000 |         0.0003 |          0.9999 |

## 5. Rollout Metrics (100 steps)

|   seed | scenario   |   reward |   stockout |   waste |   overstock |   quan |   mean_action |
|-------:|:-----------|---------:|-----------:|--------:|------------:|-------:|--------------:|
|     42 | EASY       |   0.7941 |     0.0000 |  0.0083 |      0.0009 | 0.1967 |        1.5960 |
|     42 | MEDIUM     |   0.7650 |     0.0001 |  0.0204 |      0.0042 | 0.2103 |        3.1315 |
|     42 | HARD       |   0.6332 |     0.0017 |  0.0394 |      0.0066 | 0.3190 |        4.5910 |
|    123 | EASY       |  -0.8791 |     0.0546 |  0.0083 |      0.8224 | 0.9938 |       11.3150 |
|    123 | MEDIUM     |  -0.8761 |     0.0678 |  0.0199 |      0.7929 | 0.9955 |       11.2143 |
|    123 | HARD       |  -0.8593 |     0.0765 |  0.0377 |      0.7492 | 0.9959 |       10.9405 |
|    256 | EASY       |  -0.1147 |     0.0000 |  0.0096 |      0.9639 | 0.1411 |       12.9661 |
|    256 | MEDIUM     |  -0.3067 |     0.0133 |  0.0228 |      0.9106 | 0.3601 |       12.8116 |
|    256 | HARD       |  -0.6090 |     0.0243 |  0.0437 |      0.8730 | 0.6680 |       12.7318 |
|    512 | EASY       |   0.7781 |     0.0000 |  0.0095 |      0.0511 | 0.1613 |        9.0034 |
|    512 | MEDIUM     |   0.5122 |     0.0000 |  0.0214 |      0.0286 | 0.4378 |        9.0120 |
|    512 | HARD       |   0.2725 |     0.0009 |  0.0347 |      0.0148 | 0.6771 |        9.0769 |
|   1024 | EASY       |   0.5206 |     0.0161 |  0.0090 |      0.1866 | 0.2676 |        6.6080 |
|   1024 | MEDIUM     |   0.1859 |     0.0254 |  0.0210 |      0.2785 | 0.4892 |        7.5422 |
|   1024 | HARD       |  -0.2022 |     0.0320 |  0.0389 |      0.2911 | 0.8402 |        7.8200 |

## 6. Macro SHAP (mean +/- SD across seeds)

### Macro - pi / EASY

| feature   |   mean |     sd |     cv |
|:----------|-------:|-------:|-------:|
| Inventory | 0.0230 | 0.0095 | 0.4138 |
| Demand    | 0.0250 | 0.0273 | 1.0920 |
| Waste     | 0.0002 | 0.0002 | 1.0307 |
### Macro - pi / MEDIUM

| feature   |   mean |     sd |     cv |
|:----------|-------:|-------:|-------:|
| Inventory | 0.0221 | 0.0091 | 0.4125 |
| Demand    | 0.0250 | 0.0267 | 1.0671 |
| Waste     | 0.0002 | 0.0002 | 1.0412 |
### Macro - pi / HARD

| feature   |   mean |     sd |     cv |
|:----------|-------:|-------:|-------:|
| Inventory | 0.0221 | 0.0097 | 0.4394 |
| Demand    | 0.0242 | 0.0263 | 1.0887 |
| Waste     | 0.0007 | 0.0005 | 0.6444 |
### Macro - logits / EASY

| feature   |    mean |     sd |     cv |
|:----------|--------:|-------:|-------:|
| Inventory |  6.8810 | 4.6359 | 0.6737 |
| Demand    | 26.3392 | 7.3850 | 0.2804 |
| Waste     |  0.3438 | 0.1412 | 0.4106 |
### Macro - logits / MEDIUM

| feature   |    mean |     sd |     cv |
|:----------|--------:|-------:|-------:|
| Inventory |  7.0063 | 4.4497 | 0.6351 |
| Demand    | 25.0924 | 7.5736 | 0.3018 |
| Waste     |  0.0539 | 0.0190 | 0.3535 |
### Macro - logits / HARD

| feature   |    mean |     sd |     cv |
|:----------|--------:|-------:|-------:|
| Inventory |  7.1479 | 4.2606 | 0.5961 |
| Demand    | 23.9945 | 7.6492 | 0.3188 |
| Waste     |  0.5368 | 0.1999 | 0.3724 |

## 7. Micro SHAP - Top 20 Features (mean +/- SD across seeds)

### Micro top-20 - pi / EASY

| feature          | group     |   mean |     sd |     cv |
|:-----------------|:----------|-------:|-------:|-------:|
| sales_sku162     | sales     | 0.0002 | 0.0001 | 0.5591 |
| sales_sku66      | sales     | 0.0002 | 0.0001 | 0.7777 |
| sales_sku71      | sales     | 0.0002 | 0.0001 | 0.6971 |
| sales_sku14      | sales     | 0.0002 | 0.0001 | 0.8217 |
| sales_sku36      | sales     | 0.0002 | 0.0002 | 1.0076 |
| sales_sku128     | sales     | 0.0001 | 0.0001 | 0.9004 |
| sales_sku59      | sales     | 0.0001 | 0.0001 | 0.9985 |
| sales_sku4       | sales     | 0.0001 | 0.0001 | 1.0946 |
| sales_sku65      | sales     | 0.0001 | 0.0001 | 0.6815 |
| sales_sku137     | sales     | 0.0000 | 0.0000 | 0.9155 |
| inventory_sku205 | inventory | 0.0000 | 0.0000 | 0.9395 |
| waste_sku205     | waste     | 0.0000 | 0.0000 | 0.9395 |
| inventory_sku0   | inventory | 0.0000 | 0.0000 | 0.9399 |
| inventory_sku5   | inventory | 0.0000 | 0.0000 | 0.9399 |
| inventory_sku6   | inventory | 0.0000 | 0.0000 | 0.9399 |
| inventory_sku7   | inventory | 0.0000 | 0.0000 | 0.9399 |
| inventory_sku8   | inventory | 0.0000 | 0.0000 | 0.9399 |
| inventory_sku9   | inventory | 0.0000 | 0.0000 | 0.9399 |
| inventory_sku10  | inventory | 0.0000 | 0.0000 | 0.9399 |
| inventory_sku11  | inventory | 0.0000 | 0.0000 | 0.9399 |
### Micro top-20 - pi / MEDIUM

| feature         | group     |   mean |     sd |     cv |
|:----------------|:----------|-------:|-------:|-------:|
| sales_sku162    | sales     | 0.0002 | 0.0001 | 0.5942 |
| sales_sku66     | sales     | 0.0002 | 0.0001 | 0.7811 |
| sales_sku36     | sales     | 0.0002 | 0.0002 | 1.0063 |
| sales_sku71     | sales     | 0.0002 | 0.0001 | 0.6916 |
| sales_sku14     | sales     | 0.0001 | 0.0001 | 0.8088 |
| sales_sku128    | sales     | 0.0001 | 0.0001 | 0.9090 |
| sales_sku59     | sales     | 0.0001 | 0.0001 | 1.0192 |
| sales_sku4      | sales     | 0.0001 | 0.0001 | 1.0785 |
| sales_sku65     | sales     | 0.0001 | 0.0001 | 0.6725 |
| sales_sku137    | sales     | 0.0000 | 0.0000 | 0.9091 |
| inventory_sku5  | inventory | 0.0000 | 0.0000 | 0.9256 |
| inventory_sku6  | inventory | 0.0000 | 0.0000 | 0.9256 |
| inventory_sku7  | inventory | 0.0000 | 0.0000 | 0.9256 |
| inventory_sku8  | inventory | 0.0000 | 0.0000 | 0.9256 |
| inventory_sku9  | inventory | 0.0000 | 0.0000 | 0.9256 |
| inventory_sku10 | inventory | 0.0000 | 0.0000 | 0.9256 |
| inventory_sku11 | inventory | 0.0000 | 0.0000 | 0.9256 |
| inventory_sku12 | inventory | 0.0000 | 0.0000 | 0.9256 |
| inventory_sku13 | inventory | 0.0000 | 0.0000 | 0.9256 |
| inventory_sku14 | inventory | 0.0000 | 0.0000 | 0.9256 |
### Micro top-20 - pi / HARD

| feature          | group     |   mean |     sd |     cv |
|:-----------------|:----------|-------:|-------:|-------:|
| sales_sku162     | sales     | 0.0002 | 0.0001 | 0.6100 |
| sales_sku66      | sales     | 0.0002 | 0.0001 | 0.7868 |
| sales_sku36      | sales     | 0.0001 | 0.0001 | 0.9555 |
| sales_sku128     | sales     | 0.0001 | 0.0001 | 0.9122 |
| sales_sku14      | sales     | 0.0001 | 0.0001 | 0.7821 |
| sales_sku71      | sales     | 0.0001 | 0.0001 | 0.6773 |
| sales_sku59      | sales     | 0.0001 | 0.0001 | 1.0068 |
| sales_sku4       | sales     | 0.0001 | 0.0001 | 1.1056 |
| sales_sku65      | sales     | 0.0001 | 0.0001 | 0.6866 |
| sales_sku137     | sales     | 0.0000 | 0.0000 | 0.9084 |
| inventory_sku205 | inventory | 0.0000 | 0.0000 | 0.9367 |
| waste_sku205     | waste     | 0.0000 | 0.0000 | 0.9367 |
| inventory_sku0   | inventory | 0.0000 | 0.0000 | 0.9375 |
| inventory_sku5   | inventory | 0.0000 | 0.0000 | 0.9375 |
| inventory_sku6   | inventory | 0.0000 | 0.0000 | 0.9375 |
| inventory_sku7   | inventory | 0.0000 | 0.0000 | 0.9375 |
| inventory_sku8   | inventory | 0.0000 | 0.0000 | 0.9375 |
| inventory_sku9   | inventory | 0.0000 | 0.0000 | 0.9375 |
| inventory_sku10  | inventory | 0.0000 | 0.0000 | 0.9375 |
| inventory_sku11  | inventory | 0.0000 | 0.0000 | 0.9375 |
### Micro top-20 - logits / EASY

| feature         | group     |   mean |     sd |     cv |
|:----------------|:----------|-------:|-------:|-------:|
| sales_sku66     | sales     | 0.1555 | 0.0236 | 0.1517 |
| sales_sku14     | sales     | 0.1522 | 0.0401 | 0.2633 |
| sales_sku128    | sales     | 0.1517 | 0.0215 | 0.1419 |
| sales_sku71     | sales     | 0.1485 | 0.0359 | 0.2422 |
| sales_sku162    | sales     | 0.1330 | 0.0435 | 0.3271 |
| sales_sku65     | sales     | 0.1283 | 0.0434 | 0.3381 |
| sales_sku4      | sales     | 0.1205 | 0.0245 | 0.2035 |
| sales_sku59     | sales     | 0.1205 | 0.0245 | 0.2035 |
| sales_sku36     | sales     | 0.1185 | 0.0402 | 0.3394 |
| waste_sku188    | waste     | 0.0391 | 0.0089 | 0.2278 |
| inventory_sku19 | inventory | 0.0391 | 0.0089 | 0.2278 |
| inventory_sku0  | inventory | 0.0391 | 0.0089 | 0.2278 |
| inventory_sku5  | inventory | 0.0391 | 0.0089 | 0.2278 |
| inventory_sku6  | inventory | 0.0391 | 0.0089 | 0.2278 |
| inventory_sku7  | inventory | 0.0391 | 0.0089 | 0.2278 |
| inventory_sku8  | inventory | 0.0391 | 0.0089 | 0.2278 |
| inventory_sku9  | inventory | 0.0391 | 0.0089 | 0.2278 |
| inventory_sku10 | inventory | 0.0391 | 0.0089 | 0.2278 |
| inventory_sku11 | inventory | 0.0391 | 0.0089 | 0.2278 |
| inventory_sku12 | inventory | 0.0391 | 0.0089 | 0.2278 |
### Micro top-20 - logits / MEDIUM

| feature         | group     |   mean |     sd |     cv |
|:----------------|:----------|-------:|-------:|-------:|
| sales_sku128    | sales     | 0.1452 | 0.0214 | 0.1475 |
| sales_sku66     | sales     | 0.1440 | 0.0207 | 0.1440 |
| sales_sku14     | sales     | 0.1435 | 0.0353 | 0.2458 |
| sales_sku71     | sales     | 0.1343 | 0.0294 | 0.2189 |
| sales_sku162    | sales     | 0.1262 | 0.0415 | 0.3287 |
| sales_sku65     | sales     | 0.1235 | 0.0420 | 0.3399 |
| sales_sku36     | sales     | 0.1148 | 0.0382 | 0.3326 |
| sales_sku4      | sales     | 0.1144 | 0.0248 | 0.2164 |
| sales_sku59     | sales     | 0.1144 | 0.0248 | 0.2164 |
| waste_sku188    | waste     | 0.0369 | 0.0090 | 0.2450 |
| inventory_sku19 | inventory | 0.0369 | 0.0090 | 0.2450 |
| inventory_sku0  | inventory | 0.0369 | 0.0090 | 0.2450 |
| inventory_sku5  | inventory | 0.0369 | 0.0090 | 0.2450 |
| inventory_sku6  | inventory | 0.0369 | 0.0090 | 0.2450 |
| inventory_sku7  | inventory | 0.0369 | 0.0090 | 0.2450 |
| inventory_sku8  | inventory | 0.0369 | 0.0090 | 0.2450 |
| inventory_sku9  | inventory | 0.0369 | 0.0090 | 0.2450 |
| inventory_sku10 | inventory | 0.0369 | 0.0090 | 0.2450 |
| inventory_sku11 | inventory | 0.0369 | 0.0090 | 0.2450 |
| inventory_sku12 | inventory | 0.0369 | 0.0090 | 0.2450 |
### Micro top-20 - logits / HARD

| feature         | group     |   mean |     sd |     cv |
|:----------------|:----------|-------:|-------:|-------:|
| sales_sku128    | sales     | 0.1404 | 0.0217 | 0.1546 |
| sales_sku66     | sales     | 0.1356 | 0.0198 | 0.1459 |
| sales_sku14     | sales     | 0.1355 | 0.0320 | 0.2358 |
| sales_sku71     | sales     | 0.1218 | 0.0255 | 0.2097 |
| sales_sku162    | sales     | 0.1216 | 0.0402 | 0.3304 |
| sales_sku65     | sales     | 0.1185 | 0.0406 | 0.3428 |
| sales_sku36     | sales     | 0.1113 | 0.0351 | 0.3156 |
| sales_sku4      | sales     | 0.1092 | 0.0242 | 0.2213 |
| sales_sku59     | sales     | 0.1092 | 0.0242 | 0.2213 |
| waste_sku188    | waste     | 0.0347 | 0.0092 | 0.2651 |
| inventory_sku19 | inventory | 0.0347 | 0.0092 | 0.2651 |
| inventory_sku0  | inventory | 0.0347 | 0.0092 | 0.2651 |
| inventory_sku5  | inventory | 0.0347 | 0.0092 | 0.2651 |
| inventory_sku6  | inventory | 0.0347 | 0.0092 | 0.2651 |
| inventory_sku7  | inventory | 0.0347 | 0.0092 | 0.2651 |
| inventory_sku8  | inventory | 0.0347 | 0.0092 | 0.2651 |
| inventory_sku9  | inventory | 0.0347 | 0.0092 | 0.2651 |
| inventory_sku10 | inventory | 0.0347 | 0.0092 | 0.2651 |
| inventory_sku11 | inventory | 0.0347 | 0.0092 | 0.2651 |
| inventory_sku12 | inventory | 0.0347 | 0.0092 | 0.2651 |

## 8. Stability Summary

### macro - pi / EASY

| subset      |   k |   jaccard_mean |   jaccard_sd |   spearman_all_mean |   spearman_topk_mean |   n_pairs |
|:------------|----:|---------------:|-------------:|--------------------:|---------------------:|----------:|
| all         |   1 |         0.6000 |       0.5164 |              0.8000 |              -0.4000 |        10 |
| all         |   2 |         1.0000 |       0.0000 |              0.8000 |               0.2000 |        10 |
| excl_256    |   1 |         0.5000 |       0.5477 |              0.7500 |              -0.5000 |         6 |
| excl_256    |   2 |         1.0000 |       0.0000 |              0.7500 |               0.0000 |         6 |
| stable_pair |   1 |         0.0000 |     nan      |              0.5000 |              -1.0000 |         1 |
| stable_pair |   2 |         1.0000 |     nan      |              0.5000 |              -1.0000 |         1 |
### macro - pi / MEDIUM

| subset      |   k |   jaccard_mean |   jaccard_sd |   spearman_all_mean |   spearman_topk_mean |   n_pairs |
|:------------|----:|---------------:|-------------:|--------------------:|---------------------:|----------:|
| all         |   1 |         0.6000 |       0.5164 |              0.8000 |              -0.4000 |        10 |
| all         |   2 |         1.0000 |       0.0000 |              0.8000 |               0.2000 |        10 |
| excl_256    |   1 |         0.5000 |       0.5477 |              0.7500 |              -0.5000 |         6 |
| excl_256    |   2 |         1.0000 |       0.0000 |              0.7500 |               0.0000 |         6 |
| stable_pair |   1 |         0.0000 |     nan      |              0.5000 |              -1.0000 |         1 |
| stable_pair |   2 |         1.0000 |     nan      |              0.5000 |              -1.0000 |         1 |
### macro - pi / HARD

| subset      |   k |   jaccard_mean |   jaccard_sd |   spearman_all_mean |   spearman_topk_mean |   n_pairs |
|:------------|----:|---------------:|-------------:|--------------------:|---------------------:|----------:|
| all         |   1 |         0.6000 |       0.5164 |              0.8000 |              -0.4000 |        10 |
| all         |   2 |         1.0000 |       0.0000 |              0.8000 |               0.2000 |        10 |
| excl_256    |   1 |         0.5000 |       0.5477 |              0.7500 |              -0.5000 |         6 |
| excl_256    |   2 |         1.0000 |       0.0000 |              0.7500 |               0.0000 |         6 |
| stable_pair |   1 |         0.0000 |     nan      |              0.5000 |              -1.0000 |         1 |
| stable_pair |   2 |         1.0000 |     nan      |              0.5000 |              -1.0000 |         1 |
### macro - logits / EASY

| subset      |   k |   jaccard_mean |   jaccard_sd |   spearman_all_mean |   spearman_topk_mean |   n_pairs |
|:------------|----:|---------------:|-------------:|--------------------:|---------------------:|----------:|
| all         |   1 |         1.0000 |       0.0000 |              1.0000 |               0.0000 |        10 |
| all         |   2 |         1.0000 |       0.0000 |              1.0000 |               1.0000 |        10 |
| excl_256    |   1 |         1.0000 |       0.0000 |              1.0000 |               0.0000 |         6 |
| excl_256    |   2 |         1.0000 |       0.0000 |              1.0000 |               1.0000 |         6 |
| stable_pair |   1 |         1.0000 |     nan      |              1.0000 |               0.0000 |         1 |
| stable_pair |   2 |         1.0000 |     nan      |              1.0000 |               1.0000 |         1 |
### macro - logits / MEDIUM

| subset      |   k |   jaccard_mean |   jaccard_sd |   spearman_all_mean |   spearman_topk_mean |   n_pairs |
|:------------|----:|---------------:|-------------:|--------------------:|---------------------:|----------:|
| all         |   1 |         1.0000 |       0.0000 |              1.0000 |               0.0000 |        10 |
| all         |   2 |         1.0000 |       0.0000 |              1.0000 |               1.0000 |        10 |
| excl_256    |   1 |         1.0000 |       0.0000 |              1.0000 |               0.0000 |         6 |
| excl_256    |   2 |         1.0000 |       0.0000 |              1.0000 |               1.0000 |         6 |
| stable_pair |   1 |         1.0000 |     nan      |              1.0000 |               0.0000 |         1 |
| stable_pair |   2 |         1.0000 |     nan      |              1.0000 |               1.0000 |         1 |
### macro - logits / HARD

| subset      |   k |   jaccard_mean |   jaccard_sd |   spearman_all_mean |   spearman_topk_mean |   n_pairs |
|:------------|----:|---------------:|-------------:|--------------------:|---------------------:|----------:|
| all         |   1 |         1.0000 |       0.0000 |              1.0000 |               0.0000 |        10 |
| all         |   2 |         1.0000 |       0.0000 |              1.0000 |               1.0000 |        10 |
| excl_256    |   1 |         1.0000 |       0.0000 |              1.0000 |               0.0000 |         6 |
| excl_256    |   2 |         1.0000 |       0.0000 |              1.0000 |               1.0000 |         6 |
| stable_pair |   1 |         1.0000 |     nan      |              1.0000 |               0.0000 |         1 |
| stable_pair |   2 |         1.0000 |     nan      |              1.0000 |               1.0000 |         1 |
### micro - pi / EASY

| subset      |   k |   jaccard_mean |   jaccard_sd |   spearman_all_mean |   spearman_topk_mean |   n_pairs |
|:------------|----:|---------------:|-------------:|--------------------:|---------------------:|----------:|
| all         |  10 |         0.7993 |       0.1642 |              0.7753 |               0.3501 |        10 |
| all         |  20 |         0.8576 |       0.1088 |              0.7753 |               0.7273 |        10 |
| all         |  50 |         0.8727 |       0.1201 |              0.7753 |               0.7755 |        10 |
| excl_256    |  10 |         0.9091 |       0.0996 |              0.9339 |               0.3227 |         6 |
| excl_256    |  20 |         0.9091 |       0.0996 |              0.9339 |               0.8686 |         6 |
| excl_256    |  50 |         0.9615 |       0.0421 |              0.9339 |               0.9343 |         6 |
| stable_pair |  10 |         1.0000 |     nan      |              0.9999 |              -0.0732 |         1 |
| stable_pair |  20 |         1.0000 |     nan      |              0.9999 |               0.8413 |         1 |
| stable_pair |  50 |         1.0000 |     nan      |              0.9999 |               0.9812 |         1 |
### micro - pi / MEDIUM

| subset      |   k |   jaccard_mean |   jaccard_sd |   spearman_all_mean |   spearman_topk_mean |   n_pairs |
|:------------|----:|---------------:|-------------:|--------------------:|---------------------:|----------:|
| all         |  10 |         0.7993 |       0.1642 |              0.7752 |               0.3164 |        10 |
| all         |  20 |         0.8339 |       0.1240 |              0.7752 |               0.7260 |        10 |
| all         |  50 |         0.8254 |       0.1236 |              0.7752 |               0.7838 |        10 |
| excl_256    |  10 |         0.9091 |       0.0996 |              0.9339 |               0.2969 |         6 |
| excl_256    |  20 |         0.8696 |       0.1429 |              0.9339 |               0.8749 |         6 |
| excl_256    |  50 |         0.8621 |       0.1511 |              0.9339 |               0.9487 |         6 |
| stable_pair |  10 |         1.0000 |     nan      |              0.9999 |              -0.0732 |         1 |
| stable_pair |  20 |         1.0000 |     nan      |              0.9999 |               0.8413 |         1 |
| stable_pair |  50 |         1.0000 |     nan      |              0.9999 |               0.9812 |         1 |
### micro - pi / HARD

| subset      |   k |   jaccard_mean |   jaccard_sd |   spearman_all_mean |   spearman_topk_mean |   n_pairs |
|:------------|----:|---------------:|-------------:|--------------------:|---------------------:|----------:|
| all         |  10 |         0.8576 |       0.1088 |              0.8611 |               0.3723 |        10 |
| all         |  20 |         0.8908 |       0.0914 |              0.8611 |               0.8018 |        10 |
| all         |  50 |         0.9111 |       0.0748 |              0.8611 |               0.8594 |        10 |
| excl_256    |  10 |         0.9091 |       0.0996 |              0.9339 |               0.3347 |         6 |
| excl_256    |  20 |         0.9091 |       0.0996 |              0.9339 |               0.8704 |         6 |
| excl_256    |  50 |         0.9615 |       0.0421 |              0.9339 |               0.9345 |         6 |
| stable_pair |  10 |         1.0000 |     nan      |              0.9999 |               0.0610 |         1 |
| stable_pair |  20 |         1.0000 |     nan      |              0.9999 |               0.8611 |         1 |
| stable_pair |  50 |         1.0000 |     nan      |              0.9999 |               0.9835 |         1 |
### micro - logits / EASY

| subset      |   k |   jaccard_mean |   jaccard_sd |   spearman_all_mean |   spearman_topk_mean |   n_pairs |
|:------------|----:|---------------:|-------------:|--------------------:|---------------------:|----------:|
| all         |  10 |         1.0000 |       0.0000 |              0.9999 |               0.3634 |        10 |
| all         |  20 |         1.0000 |       0.0000 |              0.9999 |               0.9059 |        10 |
| all         |  50 |         1.0000 |       0.0000 |              0.9999 |               0.9888 |        10 |
| excl_256    |  10 |         1.0000 |       0.0000 |              0.9999 |               0.2663 |         6 |
| excl_256    |  20 |         1.0000 |       0.0000 |              0.9999 |               0.8915 |         6 |
| excl_256    |  50 |         1.0000 |       0.0000 |              0.9999 |               0.9871 |         6 |
| stable_pair |  10 |         1.0000 |     nan      |              0.9999 |              -0.2561 |         1 |
| stable_pair |  20 |         1.0000 |     nan      |              0.9999 |               0.8142 |         1 |
| stable_pair |  50 |         1.0000 |     nan      |              0.9999 |               0.9780 |         1 |
### micro - logits / MEDIUM

| subset      |   k |   jaccard_mean |   jaccard_sd |   spearman_all_mean |   spearman_topk_mean |   n_pairs |
|:------------|----:|---------------:|-------------:|--------------------:|---------------------:|----------:|
| all         |  10 |         1.0000 |       0.0000 |              0.9999 |               0.2976 |        10 |
| all         |  20 |         1.0000 |       0.0000 |              0.9999 |               0.8961 |        10 |
| all         |  50 |         1.0000 |       0.0000 |              0.9999 |               0.9877 |        10 |
| excl_256    |  10 |         1.0000 |       0.0000 |              0.9999 |               0.2215 |         6 |
| excl_256    |  20 |         1.0000 |       0.0000 |              0.9999 |               0.8849 |         6 |
| excl_256    |  50 |         1.0000 |       0.0000 |              0.9999 |               0.9863 |         6 |
| stable_pair |  10 |         1.0000 |     nan      |              0.9999 |              -0.2561 |         1 |
| stable_pair |  20 |         1.0000 |     nan      |              0.9999 |               0.8142 |         1 |
| stable_pair |  50 |         1.0000 |     nan      |              0.9999 |               0.9780 |         1 |
### micro - logits / HARD

| subset      |   k |   jaccard_mean |   jaccard_sd |   spearman_all_mean |   spearman_topk_mean |   n_pairs |
|:------------|----:|---------------:|-------------:|--------------------:|---------------------:|----------:|
| all         |  10 |         1.0000 |       0.0000 |              0.9999 |               0.2988 |        10 |
| all         |  20 |         1.0000 |       0.0000 |              0.9999 |               0.8963 |        10 |
| all         |  50 |         1.0000 |       0.0000 |              0.9999 |               0.9877 |        10 |
| excl_256    |  10 |         1.0000 |       0.0000 |              0.9999 |               0.2297 |         6 |
| excl_256    |  20 |         1.0000 |       0.0000 |              0.9999 |               0.8861 |         6 |
| excl_256    |  50 |         1.0000 |       0.0000 |              0.9999 |               0.9865 |         6 |
| stable_pair |  10 |         1.0000 |     nan      |              0.9999 |              -0.2195 |         1 |
| stable_pair |  20 |         1.0000 |     nan      |              0.9999 |               0.8197 |         1 |
| stable_pair |  50 |         1.0000 |     nan      |              0.9999 |               0.9786 |         1 |

## 9. Cross-target Stability (pi vs logits, same seed)

| level   | scenario   |   seed |   k |   jaccard |   spearman_all |
|:--------|:-----------|-------:|----:|----------:|---------------:|
| macro   | EASY       |     42 |   1 |    0.0000 |         0.5000 |
| macro   | EASY       |     42 |   2 |    1.0000 |         0.5000 |
| macro   | EASY       |    123 |   1 |    0.0000 |         0.5000 |
| macro   | EASY       |    123 |   2 |    1.0000 |         0.5000 |
| macro   | EASY       |    256 |   1 |    0.0000 |         0.5000 |
| macro   | EASY       |    256 |   2 |    1.0000 |         0.5000 |
| macro   | EASY       |    512 |   1 |    1.0000 |         1.0000 |
| macro   | EASY       |    512 |   2 |    1.0000 |         1.0000 |
| macro   | EASY       |   1024 |   1 |    0.0000 |         0.5000 |
| macro   | EASY       |   1024 |   2 |    1.0000 |         0.5000 |
| macro   | MEDIUM     |     42 |   1 |    0.0000 |         0.5000 |
| macro   | MEDIUM     |     42 |   2 |    1.0000 |         0.5000 |
| macro   | MEDIUM     |    123 |   1 |    0.0000 |         0.5000 |
| macro   | MEDIUM     |    123 |   2 |    1.0000 |         0.5000 |
| macro   | MEDIUM     |    256 |   1 |    0.0000 |         0.5000 |
| macro   | MEDIUM     |    256 |   2 |    1.0000 |         0.5000 |
| macro   | MEDIUM     |    512 |   1 |    1.0000 |         1.0000 |
| macro   | MEDIUM     |    512 |   2 |    1.0000 |         1.0000 |
| macro   | MEDIUM     |   1024 |   1 |    0.0000 |         0.5000 |
| macro   | MEDIUM     |   1024 |   2 |    1.0000 |         0.5000 |
| macro   | HARD       |     42 |   1 |    0.0000 |         0.5000 |
| macro   | HARD       |     42 |   2 |    1.0000 |         0.5000 |
| macro   | HARD       |    123 |   1 |    0.0000 |         0.5000 |
| macro   | HARD       |    123 |   2 |    1.0000 |         0.5000 |
| macro   | HARD       |    256 |   1 |    0.0000 |         0.5000 |
| macro   | HARD       |    256 |   2 |    1.0000 |         0.5000 |
| macro   | HARD       |    512 |   1 |    1.0000 |         1.0000 |
| macro   | HARD       |    512 |   2 |    1.0000 |         1.0000 |
| macro   | HARD       |   1024 |   1 |    0.0000 |         0.5000 |
| macro   | HARD       |   1024 |   2 |    1.0000 |         0.5000 |
| micro   | EASY       |     42 |  10 |    1.0000 |         1.0000 |
| micro   | EASY       |     42 |  20 |    1.0000 |         1.0000 |
| micro   | EASY       |     42 |  50 |    1.0000 |         1.0000 |
| micro   | EASY       |    123 |  10 |    1.0000 |         0.9999 |
| micro   | EASY       |    123 |  20 |    1.0000 |         0.9999 |
| micro   | EASY       |    123 |  50 |    1.0000 |         0.9999 |
| micro   | EASY       |    256 |  10 |    0.6667 |         0.5515 |
| micro   | EASY       |    256 |  20 |    0.8182 |         0.5515 |
| micro   | EASY       |    256 |  50 |    0.7544 |         0.5515 |
| micro   | EASY       |    512 |  10 |    1.0000 |         1.0000 |
| micro   | EASY       |    512 |  20 |    1.0000 |         1.0000 |
| micro   | EASY       |    512 |  50 |    1.0000 |         1.0000 |
| micro   | EASY       |   1024 |  10 |    0.8182 |         0.8680 |
| micro   | EASY       |   1024 |  20 |    0.8182 |         0.8680 |
| micro   | EASY       |   1024 |  50 |    0.9231 |         0.8680 |
| micro   | MEDIUM     |     42 |  10 |    1.0000 |         1.0000 |
| micro   | MEDIUM     |     42 |  20 |    1.0000 |         1.0000 |
| micro   | MEDIUM     |     42 |  50 |    1.0000 |         1.0000 |
| micro   | MEDIUM     |    123 |  10 |    1.0000 |         0.9999 |
| micro   | MEDIUM     |    123 |  20 |    1.0000 |         0.9999 |
| micro   | MEDIUM     |    123 |  50 |    1.0000 |         0.9999 |
| micro   | MEDIUM     |    256 |  10 |    0.6667 |         0.5514 |
| micro   | MEDIUM     |    256 |  20 |    0.8182 |         0.5514 |
| micro   | MEDIUM     |    256 |  50 |    0.7544 |         0.5514 |
| micro   | MEDIUM     |    512 |  10 |    1.0000 |         1.0000 |
| micro   | MEDIUM     |    512 |  20 |    1.0000 |         1.0000 |
| micro   | MEDIUM     |    512 |  50 |    1.0000 |         1.0000 |
| micro   | MEDIUM     |   1024 |  10 |    0.8182 |         0.8680 |
| micro   | MEDIUM     |   1024 |  20 |    0.7391 |         0.8680 |
| micro   | MEDIUM     |   1024 |  50 |    0.7241 |         0.8680 |
| micro   | HARD       |     42 |  10 |    1.0000 |         1.0000 |
| micro   | HARD       |     42 |  20 |    1.0000 |         1.0000 |
| micro   | HARD       |     42 |  50 |    1.0000 |         1.0000 |
| micro   | HARD       |    123 |  10 |    1.0000 |         0.9999 |
| micro   | HARD       |    123 |  20 |    1.0000 |         0.9999 |
| micro   | HARD       |    123 |  50 |    1.0000 |         0.9999 |
| micro   | HARD       |    256 |  10 |    0.8182 |         0.7757 |
| micro   | HARD       |    256 |  20 |    0.9048 |         0.7757 |
| micro   | HARD       |    256 |  50 |    0.8519 |         0.7757 |
| micro   | HARD       |    512 |  10 |    1.0000 |         1.0000 |
| micro   | HARD       |    512 |  20 |    1.0000 |         1.0000 |
| micro   | HARD       |    512 |  50 |    1.0000 |         1.0000 |
| micro   | HARD       |   1024 |  10 |    0.8182 |         0.8680 |
| micro   | HARD       |   1024 |  20 |    0.8182 |         0.8680 |
| micro   | HARD       |   1024 |  50 |    0.9231 |         0.8680 |

## 10. Seed 256 Collapse

| scenario   |   mean_action |   frac_action_13 |   mean_entropy |   mean_max_prob |
|:-----------|--------------:|-----------------:|---------------:|----------------:|
| EASY       |       12.4241 |           0.8848 |         0.0096 |          0.9950 |
| MEDIUM     |       12.4350 |           0.8870 |         0.0095 |          0.9944 |
| HARD       |       12.3850 |           0.8770 |         0.0093 |          0.9945 |

Seed 256 chon action 13 cho phan lon SKU, entropy gan 0. Ket qua SHAP cua seed 256 can duoc bao cao rieng va khong nen dung mot minh de ket luan.


## 11. Figures

![action_distribution_EASY](figs/action_distribution_EASY.png)

![action_distribution_HARD](figs/action_distribution_HARD.png)

![action_distribution_MEDIUM](figs/action_distribution_MEDIUM.png)

![jaccard_macro_logits_EASY_k1](figs/jaccard_macro_logits_EASY_k1.png)

![jaccard_macro_logits_EASY_k2](figs/jaccard_macro_logits_EASY_k2.png)

![jaccard_macro_logits_HARD_k1](figs/jaccard_macro_logits_HARD_k1.png)

![jaccard_macro_logits_HARD_k2](figs/jaccard_macro_logits_HARD_k2.png)

![jaccard_macro_logits_MEDIUM_k1](figs/jaccard_macro_logits_MEDIUM_k1.png)

![jaccard_macro_logits_MEDIUM_k2](figs/jaccard_macro_logits_MEDIUM_k2.png)

![jaccard_macro_pi_EASY_k1](figs/jaccard_macro_pi_EASY_k1.png)

![jaccard_macro_pi_EASY_k2](figs/jaccard_macro_pi_EASY_k2.png)

![jaccard_macro_pi_HARD_k1](figs/jaccard_macro_pi_HARD_k1.png)

![jaccard_macro_pi_HARD_k2](figs/jaccard_macro_pi_HARD_k2.png)

![jaccard_macro_pi_MEDIUM_k1](figs/jaccard_macro_pi_MEDIUM_k1.png)

![jaccard_macro_pi_MEDIUM_k2](figs/jaccard_macro_pi_MEDIUM_k2.png)

![jaccard_micro_logits_EASY_k10](figs/jaccard_micro_logits_EASY_k10.png)

![jaccard_micro_logits_EASY_k20](figs/jaccard_micro_logits_EASY_k20.png)

![jaccard_micro_logits_EASY_k50](figs/jaccard_micro_logits_EASY_k50.png)

![jaccard_micro_logits_HARD_k10](figs/jaccard_micro_logits_HARD_k10.png)

![jaccard_micro_logits_HARD_k20](figs/jaccard_micro_logits_HARD_k20.png)

![jaccard_micro_logits_HARD_k50](figs/jaccard_micro_logits_HARD_k50.png)

![jaccard_micro_logits_MEDIUM_k10](figs/jaccard_micro_logits_MEDIUM_k10.png)

![jaccard_micro_logits_MEDIUM_k20](figs/jaccard_micro_logits_MEDIUM_k20.png)

![jaccard_micro_logits_MEDIUM_k50](figs/jaccard_micro_logits_MEDIUM_k50.png)

![jaccard_micro_pi_EASY_k10](figs/jaccard_micro_pi_EASY_k10.png)

![jaccard_micro_pi_EASY_k20](figs/jaccard_micro_pi_EASY_k20.png)

![jaccard_micro_pi_EASY_k50](figs/jaccard_micro_pi_EASY_k50.png)

![jaccard_micro_pi_HARD_k10](figs/jaccard_micro_pi_HARD_k10.png)

![jaccard_micro_pi_HARD_k20](figs/jaccard_micro_pi_HARD_k20.png)

![jaccard_micro_pi_HARD_k50](figs/jaccard_micro_pi_HARD_k50.png)

![jaccard_micro_pi_MEDIUM_k10](figs/jaccard_micro_pi_MEDIUM_k10.png)

![jaccard_micro_pi_MEDIUM_k20](figs/jaccard_micro_pi_MEDIUM_k20.png)

![jaccard_micro_pi_MEDIUM_k50](figs/jaccard_micro_pi_MEDIUM_k50.png)

![jaccard_vs_k_macro_logits](figs/jaccard_vs_k_macro_logits.png)

![jaccard_vs_k_macro_pi](figs/jaccard_vs_k_macro_pi.png)

![jaccard_vs_k_micro_logits](figs/jaccard_vs_k_micro_logits.png)

![jaccard_vs_k_micro_pi](figs/jaccard_vs_k_micro_pi.png)

![macro_mean_sd](figs/macro_mean_sd.png)

![micro_top20_logits_EASY](figs/micro_top20_logits_EASY.png)

![micro_top20_logits_HARD](figs/micro_top20_logits_HARD.png)

![micro_top20_logits_MEDIUM](figs/micro_top20_logits_MEDIUM.png)

![micro_top20_pi_EASY](figs/micro_top20_pi_EASY.png)

![micro_top20_pi_HARD](figs/micro_top20_pi_HARD.png)

![micro_top20_pi_MEDIUM](figs/micro_top20_pi_MEDIUM.png)


## 12. Key Findings

### 12.1. Mức độ bão hòa của policy

- seed 42: entropy 0.0006, action-13 0.000; seed 123: entropy 0.0117, action-13 0.831; seed 256: entropy 0.0094, action-13 0.883; seed 512: entropy 0.0036, action-13 0.000; seed 1024: entropy 0.0164, action-13 0.664.
- Cả 5 seed đều gần như deterministic; action mode khác nhau. Seed 256 có tỷ lệ action 13 cao nhất và được báo cáo riêng, nhưng seed 42/512 cũng rất bão hòa.

### 12.2. Micro (660 đặc trưng)

- `pi`, k=20: Jaccard all=0.861, excl_256=0.896, stable_pair=1.000; Spearman all=0.804, excl_256=0.934.
- `logits`, k=20: Jaccard all=1.000, excl_256=1.000, stable_pair=1.000; Spearman all=1.000, excl_256=1.000.
- Top-8 micro features (logits, MEDIUM): sales_sku128, sales_sku66, sales_sku14, sales_sku71, sales_sku162, sales_sku65, sales_sku36, sales_sku4.
- Kết luận micro: **ổn định cao**, đặc biệt với logits (Jaccard = 1.000); π vẫn ổn định nhưng nhạy hơn với seed bão hòa.

### 12.3. Macro (3 nhóm)

- `pi`, k=1: Jaccard all=0.600, excl_256=0.500, stable_pair=0.000; Spearman all=0.800, stable_pair=0.500.
- `logits`, k=1: Jaccard all=1.000, excl_256=1.000, stable_pair=1.000; Spearman all=1.000, stable_pair=1.000.
- Top-1 macro `pi` (MEDIUM) theo seed: seed 42=Inventory (0.013); seed 123=Inventory (0.0274); seed 256=Inventory (0.03235); seed 512=Demand (0.07172); seed 1024=Inventory (0.02572).
- Top-1 macro `logits` (MEDIUM) theo seed: seed 42=Demand (20.46); seed 123=Demand (25.33); seed 256=Demand (19.59); seed 512=Demand (38.06); seed 1024=Demand (22.02).
- Kết luận macro: **ổn định có điều kiện**. Logits luôn chọn Demand; π có top-1 dao động giữa Inventory và Demand, đặc biệt seed 512 nghiêng mạnh về Demand.

### 12.4. So sánh π vs logits

- `macro`, k=1: Jaccard trung bình=0.200, Spearman=0.600.
- `micro`, k=20: Jaccard trung bình=0.928, Spearman=0.899.
- Micro: hai target đồng thuận cao ở mức SKU. Macro: top-1 khác nhau, nhưng top-2 đồng thuận (Jaccard = 1.0).

### 12.5. Kết luận tổng

- **Micro SHAP ổn định qua 5 seed**, đặc biệt với logits; đây là bằng chứng mạnh cho tính tái lập ở mức SKU.
- **Macro SHAP chỉ ổn định có điều kiện**: logits nhất quán (Demand), π không ổn định ở top-1 do policy saturation và aggregation.
- **Seed 256** nên báo cáo riêng; không nên dùng một mình để kết luận.
- **Khuyến nghị:** dùng logits làm target chính cho stability, π làm target quyết định; báo cáo top-k (k>=10 cho micro, k=2 cho macro) thay vì top-1.


## 13. Notes

- Seed 42 chi 530 episode (ckpt-53) vs 600 cua cac seed khac; co the la mot phan nguyen nhan khac biet.
- Macro state tong hop lam inventory/waste gan nhu co dinh trong mot scenario; macro SHAP chu yeu phan anh demand.
- PartitionExplainer co yeu to ngau nhien; da co dinh seed va luu raw NPZ. Co the bat noise-floor de do variance.
- Khong dung lai output SHAP cu vi checkpoint 'output Training' khong trung 5 seed nay.
