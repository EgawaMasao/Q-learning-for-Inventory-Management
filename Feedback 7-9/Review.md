
Reviewer: 1

Comments to the Author
This manuscript proposes an Explainable Reinforcement Learning framework for large-scale multi-product inventory management by integrating Reward Difference Explanation, Minimal Sufficient Explanation, and SHAP with DQN and A2C_mod agents. The several major concerns:

1. The manuscript adopts an inventory environment containing 220 products and a 664-dimensional state space, but the action space is compressed into only 14 portfolio-level replenishment strategies. These actions range from conservative to aggressive ordering for the entire portfolio rather than allowing independent order quantities for individual products. Although this design reduces computational complexity, it may oversimplify the multi-product inventory problem and prevent the agent from representing situations in which some products require aggressive replenishment while others require limited or no replenishment.
Suggestion: Clearly explain the mathematical mapping between each of the 14 actions and the replenishment quantities assigned to the 220 products. Include an ablation or sensitivity analysis using different action-space resolutions, such as 7, 14, and 28 strategies, or a clustered product-level action representation. This would demonstrate whether the reported performance and explanation patterns are robust to the selected action-space design.


2. The manuscript introduces several reward components, including service level, holding cost, waste cost, and ordering cost. However, the rationale for selecting their coefficients, normalization ranges, and relative scales is not sufficiently justified. Since RDX and MSX directly interpret the contributions of these reward components, inappropriate scaling could cause one component to dominate the explanations because of its numerical magnitude rather than its genuine operational importance.
Suggestion: Provide empirical, managerial, or literature-based justification for the selected reward weights. Report the numerical range and distribution of each reward component before and after normalization. Include a more systematic sensitivity analysis to evaluate how changes in reward weights affect total reward, selected actions, RDX results, and MSX composition. The current one-at-a-time analysis should also be complemented by experiments that vary multiple reward weights jointly because operational objectives may interact.

3. The mathematical implementation of reward decomposition requires further clarification. The manuscript assumes that the total action-value function can be represented as the sum of component-specific value functions. However, it is unclear whether the agents were explicitly trained with separate reward-component output heads or whether the component values were reconstructed after training.
Suggestion: Describe precisely how each component-specific value function is learned. Include the relevant network architecture, loss functions, update rules, and pseudocode. Report the reconstruction error between the original total value and the sum of the decomposed values. An ablation study comparing the original DRL agent with the reward-decomposed version would help verify that the decomposition does not alter policy performance.

4. The comparison between DQN and A2C_mod is potentially informative, but the manuscript does not sufficiently define A2C_mod or explain how its outputs are transformed into Q-value-equivalent quantities. The manuscript states that A2C_mod values are converted into an equivalent form for comparison with DQN, but the exact transformation and its theoretical validity are not presented.
Suggestion: Provide the complete architecture and algorithmic modifications represented by A2C_mod. Explain the exact transformation used to derive comparable action values and justify why these values can be compared directly with DQN Q-values. Consider using a common quantity, such as action advantage, policy logit, or normalized value difference, for both agents. Without a common output definition, the observed differences may be caused by the explanation target rather than the learning architecture.

5. The SHAP analysis explains the selected-action Q-value for DQN but the selected-action policy probability for A2C_mod. These outputs have different mathematical properties and ranges. A Q-value is unbounded, whereas an action probability is constrained and influenced by all actions through normalization. Therefore, direct comparisons of SHAP magnitude, stability, and feature coverage may not be fair.
Suggestion: Explain equivalent outputs for both architectures, such as action advantages or pre-softmax policy logits. Alternatively, clearly state that the SHAP comparison is qualitative rather than quantitative. Include an additional experiment showing whether the conclusions remain unchanged when SHAP is applied to different actor and critic outputs.

6. The manuscript uses KernelSHAP with a generated background distribution consisting of 200 states that are subsequently reduced to 100 representative samples. However, the generation and selection procedures are not described in sufficient detail. Correlations among inventory, demand, waste, capacity, and other operational features may cause unrealistic perturbed states and unreliable feature attributions. The manuscript appropriately notes that the resulting SHAP values should be interpreted as associative rather than causal, but the background construction still requires stronger validation.
Suggestion: Describe how the background states are generated and how the 100 representative states are selected. Compare synthetic background states with states sampled directly from the training trajectories. Include sensitivity analysis using different background sizes and sampling strategies. The authors should also report whether the perturbed states remain within valid inventory and capacity constraints.

7. The original state space includes 664 product-level and system-level variables, but much of the SHAP analysis aggregates these variables into only inventory, demand, and waste categories. Such aggregation improves readability but may conceal important product-specific and capacity-related decision factors. The conclusion itself acknowledges that aggregated features may cause information loss and recommends finer-grained analysis.
Suggestion: Include additional product-level and system-level SHAP results. Present representative cases showing the highest-ranked products, warehouse utilization, transport capacity, demand volatility, and time until the next ordering period. An ablation study comparing aggregated SHAP, grouped SHAP, and individual-feature SHAP would help determine the effect of feature aggregation on interpretability.

8. The perturbation-based faithfulness evaluation is a positive aspect of the manuscript. The results generally indicate that masking the most relevant SHAP features produces greater output changes than random or least-relevant masking. However, the analysis remains largely descriptive and does not provide sufficient statistical evidence that the observed differences are robust.
Suggestion: Report the area under the perturbation curve, action-switching rates, confidence intervals, and statistical comparisons among MoRF, random, and LeRF masking. Clearly state the number of evaluated states, masking percentages, replacement strategy, and number of repeated runs. Similar statistical analysis should be provided for MSX-guided versus random reward-component masking.

9. The results are reported using mean values and standard deviations in some sections, but it is unclear whether the agents were independently retrained using different random seeds or whether the same trained agents were repeatedly evaluated. Repeating SHAP estimation on a fixed model does not demonstrate robustness to stochastic model training. The manuscript later acknowledges that explanation robustness should be evaluated across multiple random seeds, which indicates that the present architecture-level conclusions may be premature.
Suggestion: Retrain DQN and A2C_mod using multiple independent random seeds and report means, standard deviations, confidence intervals, and effect sizes for both operational and explanation metrics. Appropriate statistical tests, such as paired tests, ANOVA, or non-parametric alternatives, should be used to support claims that one architecture is more stable, adaptive, or explainable than the other.

10. The classical base-stock policy provides an important operational baseline and achieves a higher total reward than both DRL agents, whereas the DRL agents achieve higher service levels and lower stockout rates. This result illustrates a meaningful trade-off, but it also raises questions regarding the practical advantage of the learned policies.
Suggestion: Expand the discussion of why the DRL policies should be preferred when the classical baseline achieves a higher total reward. Explain whether the reward formulation places excessive penalties on holding and waste or whether the DRL agents are inadequately trained. Include additional interpretable baselines, such as an (s,S) policy, a demand-forecasting heuristic, or a rolling-horizon optimization method. The procedure used to choose the base-stock safety factor should also be clearly described and performed using validation data rather than the test set.

11. The experimental dataset contains 900 training cycles and 496 testing cycles, but the manuscript does not clearly explain whether the split is chronological or random. In inventory and demand forecasting applications, random splitting may introduce temporal leakage because future demand patterns can indirectly influence model training.
Suggestion: Clarify how the training, validation, and testing periods are divided. Ensure that the split is chronological and that forecasting models, normalization parameters, reward coefficients, and baseline parameters are fitted using training or validation data only. The authors should also discuss whether the EASY and HARD scenarios represent in-distribution or out-of-distribution evaluation conditions.

12. The paper reports that A2C_mod adapts better in difficult environments, whereas DQN produces more stable but less flexible explanations. However, these conclusions may be affected by differences in model outputs, stochasticity, threshold selection, and the limited number of independently trained models. For example, the RDX results indicate that A2C_mod adjusts operational priorities in the HARD scenario, while DQN continues to emphasize service. Nevertheless, explanation adaptability should not automatically be interpreted as superior operational performance.
Suggestion: Moderate architecture-level conclusions and clearly distinguish among operational performance, policy adaptability, explanation stability, and explanation faithfulness. State that the findings are specific to the current environment, action representation, reward design, and experimental settings unless they are validated using additional algorithms and datasets.

13. Although the framework is presented as improving operational trust and managerial understanding, no domain-expert evaluation or user-oriented interpretability study is included. Perturbation-based faithfulness establishes whether explanations are related to model outputs, but it does not establish whether inventory managers can understand or act on them.
Suggestion: Include several case studies in which RDX, MSX, and SHAP explanations are translated into managerial language. Where possible, obtain feedback from inventory or supply-chain experts regarding clarity, usefulness, and consistency with operational reasoning. If expert validation cannot be conducted, this limitation should be discussed more prominently and claims concerning increased trust should be moderated.

14. Avoid citation of non-peer-reviewed or weakly substantiated works.
Suggestion: Ensure all references are from peer-reviewed sources. Avoid citing arXiv or preprints papers. Remove or replace them with authoritative journal or conference papers.

15. While the manuscript cites several strong foundational works on reinforcement learning, inventory management, and explainable artificial intelligence, it omits recent and relevant studies on deep reinforcement learning in dynamic operational environments, constrained logistics optimization, and trustworthy explanation of learned policies.
Suggestion: Incorporate and discuss the following studies to enrich the literature review and strengthen the contextual framework:
Taherinavid et al. (2023). Automatic transportation mode classification using a deep reinforcement learning approach with smartphone sensors. IEEE Access, 12, 514–533. https://doi.org/10.1109/ACCESS.2023.3346875
→ Demonstrates the application of deep reinforcement learning to heterogeneous and dynamically changing sensor observations and can support the discussion of adaptive sequential decision-making.
Liu et al. (2023). A systematic literature review of vehicle routing problems with time windows. Sustainability, 15(15), 12004. https://doi.org/10.3390/su151512004
→ Provides a comprehensive overview of logistics optimization, scheduling, transportation constraints, and capacity-aware decision-making that is relevant to multi-product inventory management.
Yang et al. (2025). Wastewater treatment monitoring: Fault detection in sensors using transductive learning and improved reinforcement learning. Expert Systems with Applications, 264, 125805. https://doi.org/10.1016/j.eswa.2024.125805
→ Illustrates the use of improved reinforcement learning in a dynamic operational monitoring environment and can strengthen the discussion of policy robustness under changing conditions.
Liu et al. (2026). A transductive learning-based method for vehicle routing problems using off-policy proximal policy optimization and hyperparameter optimization. Information Sciences, 123671. https://doi.org/10.1016/j.ins.2026.123671
→ Presents a recent reinforcement-learning approach for constrained routing and logistics optimization and provides a useful comparison for operational decision-making under complex constraints.
Yang et al. (2026). Explainable deep reinforcement learning for anomaly detection in IoT-enabled metaverse healthcare: Toward trustworthy cyber threat intelligence. Research, 9, 1245. https://doi.org/10.34133/research.1245
→ Directly supports the motivation for integrating explainability with deep reinforcement learning to improve transparency, trustworthiness, and interpretation of learned decisions.
Incorporating these references would better position the manuscript within the current state of the art in reinforcement-learning-based operational optimization and explainable decision-support systems. The authors should critically relate these studies to the proposed inventory-management framework rather than merely adding them to the reference list.

Minor Concerns
16. Ensure consistent use of “Explainable Reinforcement Learning,” “XRL,” and “XAI.” Define each abbreviation at first use and avoid alternating among terms without explanation. Use consistent terminology for “holding cost” rather than switching between “holding,” “storage,” and “hold” cost.

17. Clearly define A2C_mod when it first appears. Explain what has been modified relative to conventional A2C and use the same notation consistently throughout the manuscript.

18. Several equations contain formatting and notation problems, including unclear summation symbols, missing brackets, duplicated terms such as “argarg,” inconsistent subscripts, and possible sign errors in the ordering-cost formula. Conduct a thorough mathematical review and ensure that every symbol is defined immediately after its first occurrence.

19. Report the complete hyperparameter configuration for DQN and A2C_mod, including learning rate, discount factor, optimizer, batch size, replay-buffer size, network architecture, target-network update interval, entropy coefficient, number of episodes, stopping criteria, and random seeds.

20. The manuscript contains grammatical and stylistic issues, including missing hyphens and awkward phrases such as “many products inventory management problems” and “systematic comparative analyzing.” A thorough English-language revision is recommended.



Reviewer: 2

Comments to the Author
Thank you for the careful and substantial revision. The authors have satisfactorily addressed the major concerns raised in the previous review. In particular, the addition of the base-stock baseline, the clarification of the portfolio-level action space, the more detailed SHAP specification, the top-k product-level analysis, and the new faithfulness evaluation considerably strengthen the manuscript. The revised positioning of the study is also more appropriate and balanced. I have no further major methodological concerns. I recommend acceptance.

Reviewer: 3

Comments to the Author
This revision substantially addresses previous concerns by adding a base-stock baseline, top-k SHAP analysis, and faithfulness evaluations. The framework's integration of multiple XRL methods is commendable. However, several fundamental issues remain.

Action Space Abstraction Remains a Critical Limitation. The 14 global ordering actions applied to the entire 220-product portfolio represent an extreme simplification that limits practical relevance. Real inventory systems require SKU-level decisions with heterogeneous demand patterns, costs, and spoilage risks. Acknowledge this as a fundamental limitation in the abstract and conclusion, not just a "future direction"

Clarify whether the global action is the same order quantity for all products or an aggregated portfolio-level strategy If possible, add an experiment with at least 2-3 SKU groups to demonstrate the framework's extensibility

Despite adding top-k analysis, the primary SHAP explanation uses only 3 aggregated features such as Inventory, Demand, Waste. This ignores the 660-dimensional product-level structure that makes the problem challenging.

Make the top-k micro-level analysis (Section 4.5.3) the primary SHAP result, not a supplement

Demonstrate whether the same product-level features are consistently important across scenarios

Provide an explanation for why certain products dominate the top-20 list e.g., high demand, high spoilage

The base-stock policy outperforms DRL agents in total reward across all scenarios (Table 4). This undermines the operational value of the DRL agents being explained.

Directly address why practitioners should use DRL over base-stock when the classical policy achieves higher rewards

Explain whether the DRL agents' near-perfect service levels (0.9999) justify their higher holding/waste costs

The faithfulness experiments use action flip rates and Q-value drops, but the thresholds for "significant" changes are not justified. Define what constitutes a "meaningful" Q-value drop or action flip. Report statistical significance tests comparing MSX-guided vs. random masking

Clarify whether the faithfulness results hold consistently across episodes or vary significantly

Terminology Inconsistencies
"A2C_mod" vs. "A2C mod" used interchangeably
"Multimodel" appears (should be "multi-model")
Section 4.1.3 shows hhhh at table end (formatting error)

Figure References Figures 12-20 are referenced but many captions are incomplete or missing in the PDF. There are only 20 references and they are also not key and latest references. I would highly suggest to see these key papers how research is cited. 10.1016/j.phycom.2018.07.007, https://doi.org/10.1038/s41598-025-34297-5. Cite atleast 50 latest researches and make a story introduction as done in these above mentioned key papers. I would suggest to also consider these key papers. doi: 10.1109/IBCAST47879.2020.9044564, 10.1109/FIT60620.2023.00063, http://www.mdpi.com/2313-7673/11/1/51, http://www.mdpi.com/1424-8220/26/4/1172
https://doi.org/10.1007/s13369-024-08918-6, https://doi.org/10.1038/s41598-026-40798-8, http://www.mdpi.com/1999-4893/19/2/162, http://www.mdpi.com/2079-9292/13/22/4388,

Figure 1 ("Overall XRL workflow") is referenced but not visible in the provided text

Mathematical Notation Issues Equation for MSX in Section 3.2 contains duplicate argarg and missing constraints. Variable definitions for P and N in Section 3.2 could be clearer

Section 4.1.1 states "agents are trained in the publicly available retail dataset Instacart" but the training procedure is not described. How were the 220 products selected? Are they the most frequent items? See these papers to understand how dataset detail is given in research paper. 10.1109/ICCSNT58790.2023.10334545, 10.1109/IBCAST54850.2022.9990102, https://doi.org/10.1049/pel2.70101

Reviewer: 4

Comments to the Author
The manuscript is now generally suitable for publication, but a few issues should be clarified in a minor revision:
- The empirical justification for the EASY/MEDIUM/HARD scenarios remains somewhat qualitative; the selected inventory, demand, and spoilage levels should be more explicitly linked to statistics or percentiles from the Instacart data.
- The SHAP implementation should be made internally consistent. Section 3.3.3 describes KernelSHAP, whereas the micro-level analysis uses the SHAP Partition Explainer; the distinction and corresponding configurations should be clearly stated.
- The dimensionality should be reported consistently as 660 product-level features plus four system-level features, rather than alternating between 660 and 664 product-level dimensions.
- The manuscript should clarify how the Base-Stock safety factor \(k\) is selected and how SKU-level replenishment quantities are mapped to the 14 portfolio-level actions.
- The discussion of the HARD scenario should be checked for consistency, since the text suggests a negative reward for DQN, whereas Table 4 reports positive total rewards for both agents. 

