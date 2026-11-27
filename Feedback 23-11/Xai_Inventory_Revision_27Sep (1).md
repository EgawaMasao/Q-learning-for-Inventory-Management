Explaining Deep RL Agents in Multi-Product Inventory Management: An XRL Framework

**Dac Hoang Nguyen1*, Egawa Masao2, Viet Pham Quoc Le2, Nhu Tai Do2**

*1Department of Information Technology, Tay Do College, 90 National Highway 1A, Ba Lang Ward, Cai Rang District, Can Tho City, Vietnam*

*2Department of Information Technology, Saigon University, 273 An Duong Vuong Street, Cho Quan Ward, Ho Chi Minh City, Vietnam*

*Received ; revised ; accepted*

 

 

***Abstract: ***

**Inventory management requires balancing demand fulfillment, cost reduction, and waste minimization. While Deep Reinforcement Learning (DRL) optimizes these systems, its black-box nature makes its decisions difficult to interpret and audit. This study proposes an Explainable Reinforcement Learning (XRL) framework integrating Reward Difference Explanation (RDX), Minimal Sufficient Explanation (MSX), and SHapley Additive exPlanations (SHAP) to analyze multi-product replenishment system. Utilizing an inventory simulation with DQN and A2C_mod agents, we introduce a reward decomposition mechanism aligned with core business objectives. A classical base-stock policy serves as an operational baseline. Furthermore, SHAP is extended via top-k feature-level analysis, and perturbation-based faithfulness tests are introduced to validate explanations by masking key components. Evaluated on the Instacart dataset (220 products), results reveal distinct operational trade-offs. While the base-stock policy provides a strong reference, DRL agents prioritize service levels and stockout reduction. RDX and MSX clarify the immediate (one-step) reward-space trade-offs behind each decision, whereas SHAP provides complementary feature-space evidence. A comparison with a trained reward decomposition shows that these one-step explanations understate the long-horizon weight of stockout avoidance. Overall, this multi-level XRL framework enhances the transparency of DRL replenishment decisions in a shared-action-space setting. The explanations are assessed for faithfulness and stability with respect to the agents. Their usefulness to inventory managers was not evaluated and is left to a user study.  Actions are selected per SKU by a weight-sharing policy over 14 shared discretized ordering levels; this abstraction trades order-size granularity for scalability, and the findings hold only for the present environment, action, and reward settings.**

***Keywords: *Explainable Reinforcement Learning, Inventory Management, Deep Q-Network, Advantage Actor-Critic, Reward Decomposition, SHAP.**

***Classification numbers: *1.2, 1.3**

***Highlights:***

**•  An XRL framework combining RDX, MSX and SHAP explains DQN and A2C_mod replenishment policies for 220 Instacart products.**

**•  Reward decomposition separates service, holding, waste and ordering objectives; RDX and MSX describe the one-step trade-offs behind each decision.**

**•  A classical base-stock policy attains a higher total reward, whereas the DRL agents prioritise service level and stockout reduction.**

**•   Perturbation-based faithfulness tests and multi-seed analyses assess the reliability of the explanations.**

# Introduction

	large-scale inventory management is a complex decision-making problem. Managers must simultaneously balance multiple conflicting objectives such as meeting customer demand, reducing inventory holding costs, and limiting waste due to spoilage or expiration. At the same time, decisions must comply with constraints on transportation capacity, ordering costs, and storage capacity. As the number of products increases, the state and action spaces expand rapidly, making traditional optimization methods difficult to scale to large real-world problems [1]. DRL has been demonstrated as an effective approach for sequential decision-making problems with large state spaces. In particular, methods such as Deep Q-Network (DQN) and Advantage Actor-Critic (A2C) have shown the ability to learn effective policies in many complex control and optimization tasks [2, 3]. In the context of inventory management, DRL agents such as DQN and A2C_mod can achieve high performance and good scalability for multi-product inventory management problems with variable dimensionality [4]. However, despite achieving high optimal performance in many complex tasks, DRL models often operate as black boxes. This issue makes the agents' decision-making processes difficult to understand and interpret intuitively [5]. This lack of transparency is particularly critical in real-world inventory management systems, where replenishment decisions should be transparent enough for planners to review them and, when necessary, override them. 

Most existing studies on DRL for inventory management focus primarily on performance optimization, such as maximizing cumulative rewards or minimizing long-term average costs, while paying limited attention to explaining the decision-making behavior of agents [4, 5]. In these studies, agents are typically evaluated using quantitative metrics, whereas the trade-offs among different business objectives are not analyzed in a transparent manner. In parallel, the field of XRL has been developed to address the black box issue of reinforcement learning models. Representative approaches include explanations based on reward decomposition such as RDX and MSX, which enable clarification of the role of each reward component in the agent decisions [6]. In addition, feature-level explanation methods such as SHAP have been applied to analyze how state features influence the agent value function or policy [7, 8]. Although XRL has attracted increasing attention, most existing studies still focus on individual explanation methods or are evaluated in simple experimental environments. There remains a lack of studies that comprehensively apply multiple RL explanation methods to large-scale inventory management problems, as well as a lack of systematic comparative analysis of the explainability of decision-making behavior between DRL agents based on different mechanisms, such as DQN and A2C_mod.

Considering the limitations mentioned above, this study proposes an XRL framework for large-scale multi-product inventory management, built upon the simulation environment and DRL agents adopted from prior work [4]. Within this framework, the study systematically integrates three complementary explanation methods to elucidate the decision-making behavior of the agents.The framework is explanatory rather than algorithmic, it does not propose a new replenishment, resource-allocation, or constrained reinforcement learning method, but analyses the decisions of existing DRL agents that operate under shared capacity, cost, and spoilage pressures.

This study makes the following main contributions:

•	Proposes a comprehensive XRL framework for large-scale multi-product inventory management, directly extending DQN and A2C_mod from Meisheri et al (2020). by integrating three complementary explanation methods RDX, MSX, and SHAP.

•	Develops a practical reward decomposition mechanism, enabling RDX to clarify the one-step trade-offs among operational objectives and MSX to identify the core set of reasons that justify each decision, and compares these one-step explanations with long-horizon explanations from a trained decomposition.

•	Uses SHAP to provide aggregated system-level feature attribution and complements it with top-k feature-level analysis.

•	Introduces perturbation-based faithfulness evaluation for the proposed explanations.

•	Conducts a systematic comparative analysis of the explainability of DQN and A2C_mod, thereby elucidating the impact of decision-making mechanisms on the effectiveness and characteristics of each XRL approach.

## Reinforcement Learning for Inventory Management

Inventory management is an important dynamic decision-making problem in supply chains and has been approached through heuristic methods, Approximate Dynamic Programming (ADP), and optimization. Heuristic policies such as the  policy and base-stock policy have advantages in terms of simplicity, interpretability, and low implementation cost [9, 10]. However, these policies often rely on static assumptions and may not fully capture complex interactions in multi-product inventory systems with shared constraints, fluctuating demand, spoilage risk, and limited storage or transportation capacity. ADP methods improve performance by approximating value functions or policies [11]; however, they depend heavily on the dynamics model and face computational challenges at large-scales. Despite these limitations, classical inventory policies remain important operational baselines because they are simple, interpretable, and widely used in inventory control. Therefore, comparing DRL agents against a classical policy such as base-stock is necessary to assess whether the learned policies are operationally competitive, rather than only different from each other. Such a comparison also provides a practical reference point for interpreting whether the explanations produced by XRL methods are attached to policies with meaningful inventory-control performance.

Reinforcement Learning (RL) provides a data-driven approach that allows an agent to learn a policy directly through interaction with the environment without requiring an explicit dynamics model [12]. Inventory problems are commonly modeled as Markov Decision Processes (MDPs). Among modern RL algorithms, Deep Q-Network (DQN) and Advantage Actor-Critic (A2C) are widely used for problems with large state spaces. DQN learns the action-value function [13], while A2C combines policy learning and value estimation to improve training stability [3]. In recent developments, DQN and A2C_mod have achieved promising performance and scalability in large-scale multi-product inventory management problems [4].Beyond inventory control, DQN- and policy-gradient-based DRL agents have shown similar adaptability in other dynamic, high-dimensional operational settings, including sensor-driven mode classification [14] and fault detection under non-stationary monitoring conditions [15].However, these studies focus primarily on aggregate performance, while the decision-making mechanisms and trade-offs among operational objectives are not analyzed transparently. This highlights the need for Explainable Reinforcement Learning methods that can explain both the operational performance and the internal decision logic of DRL agents.

Inventory replenishment is one instance of a wider class of sequential decisions made under limited resources. Constrained Markov decision processes (CMDPs) formalize this class by maximizing the expected return subject to bounds on expected auxiliary costs, such as budget use or capacity occupation [26]. Constrained deep RL methods enforce such bounds during policy learning, for example through trust-region policy updates with near-constraint satisfaction at each iteration [27] or through a Lagrangian penalty signal adapted on a slower timescale [28]. Related work on DRL for resource allocation learns to pack jobs with multiple resource demands under capacity limits [29], and in inventory control a tuned actor-critic agent has been shown to match strong heuristics on lost-sales, dual-sourcing, and multi-echelon problems [30]. A further difficulty is that operating conditions change over time. A policy trained under one demand regime may be deployed under different demand levels or cost structures, which has motivated RL methods for non-stationary environments [31]. The present study differs from this line of work in its aim. Storage capacity enters the environment only as a bound on order quantities, and holding, waste, and ordering costs enter the reward as penalty terms, so the agents are not constrained-RL agents and no new allocation or constrained-optimization algorithm is proposed. Instead, the study explains how existing DQN and A2C_mod policies trade off service against holding, waste, and ordering costs, and how these trade-offs shift under controlled changes in demand and spoilage, a question that constrained or resource-allocation RL methods do not answer by themselves.

## Explainable Reinforcement Learning

XRL aims to enhance the interpretability and transparency of the decision-making behavior of reinforcement learning agents. Due to the characteristics of sequential decision making, delayed rewards, and the presence of dynamic trade-offs among multiple    optimization objectives, explaining the behavior of DRL models becomes particularly challenging compared to supervised learning models [5, 16]. XRL has been developed to provide meaningful explanations, with the aim of supporting verification and human oversight in high-risk domains such as finance [17], autonomous transportation [18], robotic learning [19], and trustworthy anomaly detection in IoT/cyber-physical systems [20].  

XRL methods are commonly divided into two main categories. Intrinsic explanation methods integrate explanatory mechanisms directly into the agent’s learning process, exploiting the internal structure of the RL problem—particularly the reward—to clarify optimization motives and long-term strategies [18]. Within this category, reward-based explanation methods decompose the total reward into semantically meaningful components. In particular, Reward Difference Explanation (RDX) explains differences between actions through the contribution of each reward component, while Minimal Sufficient Explanation (MSX) identifies the smallest subset of core reasons that is still sufficient to justify the agent’s decision [6]. However, these methods are highly dependent on the design and decomposition of the reward, which may limit generalization. In contrast, post hoc explanation methods generate explanations after the RL model has been trained by analyzing the relationship between the agent’s input states and outputs. A common approach is feature attribution, in which SHapley Additive exPlanations (SHAP) are extended from supervised learning to explain DRL models by quantifying the influence of each input feature on the Q value or action policy [5, 18]. Although highly flexible, post hoc methods often struggle to directly reflect long-term optimization objectives and the intrinsic trade-offs encoded in the agent’s reward.

## Explainable Reinforcement Learning in Inventory Management

Although deep reinforcement learning has been applied effectively to many complex inventory management and supply chain problems [21], research adopting the XRL perspective in this domain remains limited. Most existing studies focus on improving optimization performance, while the explanation of agents’ decision-making behavior especially in large-scale multi-product inventory systems with numerous operational constraints has not been adequately addressed. Some early studies have examined policy explanation or the importance of state features in** **relatively simple logistics and supply chain problems, for example through attention mechanisms or feature attribution in vehicle routing and scheduling tasks [22]. Broader operations-research treatments of vehicle routing under time-window and capacity constraints reinforce the relevance of this direction for multi-constraint inventory systems [23, 24]. However, these studies typically rely on a single explanation mechanism and are evaluated in small scale environments, making it difficult to fully reflect the complex trade-offs present in real-world inventory systems. 

In the context of multi-product inventory management, existing XRL studies mainly remain at the level of analyzing state features or providing high level policy explanations, without clarifying the different operational objectives encoded in the agent’s reward. In particular, the key trade-offs among demand fulfillment, inventory holding costs, waste, and ordering costs have not yet been transparently analyzed through explanation mechanisms with clear business semantics. Furthermore, no study has been performed that simultaneously combines XRL methods based on rewards and state characteristics, nor any systematic comparative analysis of explanatory capability across models such as DQN and A2C_mod in large-scale multi-product inventory management problems involving hundreds of Stock Keeping Units (SKU). Motivated by these gaps, the present study builds on the environment and agents proposed by Meisheri et al. (2020) [4] and proposes a comprehensive XRL framework for large-scale multi-product inventory management.

# Materials and Methods

## Inventory Management Environment

This study builds upon the large-scale inventory management environment of many products  in which the problem is modeled as an MDP. At each time step , the agent observes the state* *, selects an action , and receives a reward , following the dynamics of MDP.

The state * *reflects the operational condition of the system and includes both product level and system level features. For each of the 220 products, the state consists of the inventory level , the demand forecast , and the spoilage indicator* *. In addition, the state includes aggregated system level features: warehouse utilization rate , transportation capacity , demand volatility , and the time until the next ordering period . The full state vector is represented as follows:

with a total dimensionality of |S| = 664 (660 product-level features, i.e., 220 SKUs × 3 features, plus 4 system-level features). The action space is discretized into 14 ordering strategies , corresponding to replenishment levels ranging from conservative to aggressive and shared by all SKUs, subject to capacity and storage constraints.  The two agents, DQN and A2C_mod, are trained in the same environment and configuration to ensure fairness in comparing performance and explainability. Figure 1 details the workflow used to explain reinforcement learning decisions within the study’s warehouse management system. Each SKU independently selects one of these 14 shared ordering-rate levels from its own local state through a weight-sharing policy (see Action-to-Replenishment Mapping below); the levels are neither a single portfolio-wide order quantity nor fully independent SKU-specific order sizes. Lower-index actions represent conservative replenishment, while higher-index actions indicate more aggressive ordering to reduce stockout risk. This design keeps the learning problem tractable in a 220-product environment, but it also abstracts away fine-grained SKU-level ordering constraints. 

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 1. Overall XRL workflow for explaining the agent’s decisions regarding inventory management.**

**       Table 1.  Operational interpretation of the 14 discrete action**

**Action range**

**Replenishment level**

**Operational meaning**

a0​

Minimum / no replenishment

Avoids ordering when inventory is sufficient or spoilage risk is high

a1 -  a3

Low replenishment

Conservative ordering to reduce holding and waste costs

a4 -  a8

Medium replenishment

Balanced strategy between demand fulfillment and cost control

a9 -  a13

High replenishment

Aggressive ordering to prevent stockouts under high demand or low inventory

### *Action-to-Replenishment Mapping*

To see how replenishment works in practice, we map these 14 discrete actions to actual order quantities for each of the 220 SKUs. We use a single actor network () with shared weights for all items. At each step, this network runs separately for each SKU  using its local three-dimensional state vector, avoiding the need to process the full system state :

where  ∈ [0,1] is the normalized inventory level of SKU i,  is its forecast demand,  is its fixed shelf-storage capacity, and  is a deterministic estimate of spoilage (2.5% of current stock per cycle). With shared parameters , we run this function 220 times per cycle, once for each SKU, instead of making a single portfolio-wide pass. The network then outputs a categorical distribution over the 14 replenishment actions, which we sample during training and select greedily via argmax at test time:

The selected index is then mapped to a standardized supplementary rate using a fixed category A comprising 14 values ​​shared across all SKUs:

Where A {0, 0.005, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.03, 0.04, 0.08, 0.12, 0.2, 0.5, 1}. And the physical order quantity for SKU i is obtained by rescaling this fraction by that SKUs own capacity:

This setup defines the policy for our primary model, weight-shared A2C_mod. For comparison, the DQN baseline shares the same action space and input representation, but selects actions via an -greedy policy over estimated Q-values instead of softmax sampling. 

Table 2 summarizes the training configuration. Both agents share the same inputs, 14-action menu, discount factor, optimizer family, dropout rate, and 600-episode/900-timestep budget without early stopping, and are evaluated with five seeds (42, 123, 256, 512, 1024); they differ mainly in hidden-layer width, exploration mechanism, and update rule.

**Table 2. Training hyperparameters for DQN and A2C_mod.**

**Hyperparameter**

**DQN**

**A2C_mod**

Hidden layers / width

3 hidden layers, 128 units each

3 hidden layers, 32 units each

Normalization

GroupNormalization (groups = 1) after each hidden layer

GroupNormalization (groups = 1) in the critic only

Output head

Dense(128→14): one Q-value per action, per product

Actor: Dense(32→14) + softmax; Critic: Dense(32→1) state value

Optimizer

Adam

Adam (separate actor/critic instances)

Learning rate

0.001

0.001 (actor), 0.001 (critic)

Discount factor γ

0.99

0.99

Entropy coefficient

(not applicable)

0.001

Experience replay buffer

100,000 transitions (min. 1,000 before training starts)

(on-policy, no replay buffer)

Exploration

ε-greedy, ε: 1.0 → 0.01, linear decay over the first 400 episodes

stochastic softmax policy sampling (no explicit ε)

Target network

Separate target Q-network, hard-synced from the online network every 10 episodes (Double-DQN: online net selects the action, target net evaluates it)

(no target network; single critic)

Loss function

Huber loss on the TD error

Critic: 0.5×(TD error)²; Actor (A2C_mod rule): mean-squared difference between the old and advantage-reweighted policy, p_new = softmax(logπ(a|s) + δ/(|a−a’|+1))

### *Data Split and Chronological Protocol*

The 1,504 six-hour demand cycles (about 376 days) are split strictly chronologically, without shuffling, to avoid temporal leakage: cycles 1–900 form the training segment and cycles 901–1,504 the held-out test segment (604 cycles). The DRL agents are trained on the training segment only and use no separate validation segment. For the base-stock baseline, the training segment is further divided into a fitting part (cycles 1 - 720) and a validation part (cycles 721 - 900), which is used only to select the safety factor (Section 3.1.3). Mean capacity-normalized demand falls from 0.107 in the training segment (0.062 in its validation part) to 0.035 in the test segment, a natural distribution shift; the EASY/MEDIUM/HARD scenarios are controlled demand and waste scalings applied on top of the test segment, not re-splits of the data. The SHAP analyses use the final 504 cycles of the test segment (cycles 1,001–1,504). The data segment and the number of states used in each analysis are listed in Section 4.1.1

### *Leakage Audit and Corrected Procedure*

Four components were audited for test-set leakage (Table 3). Demand forecasting is not a learned component (agents observe capacity-normalized realized demand) and the reward uses fixed unity weights, so neither leaks. Two components initially used test information. Normalization capacity had been derived from the full series and exceeded its train-only counterpart by about one quarter on average (maximum deviation about one half), yet the resulting shift in normalized demand and in reward is under 0.02; capacity is now defined from the training interval alone. The base-stock safety factor had been selected on the test interval; re-selecting it on the validation interval shifts the optimum by only half a unit, with nearly parallel validation and test reward curves, so the validation optimum is reported and the test optimum kept only as a reference. Neither correction requires retraining the DRL agents, and the reported results remain valid.

**Table 3. Leakage audit summary.**

Component

Estimation basis

Leakage

Correction

Demand forecasting

No model, observed demand

No

Documented as observed, train-only if added

Normalization (capacity)

Full series (initial) vs training-only (corrected)

Yes — sizable in constant, modest downstream

Regenerate capacity from training interval only

Reward design

Fixed, unity weights

No

Justified via sensitivity analysis

Baseline (base-stock)

Target on training (correct), safety factor on test (leaked)

Yes → Fixed

Select safety factor on validation interval

Per-product capacity deviations and the safety-factor grid are provided in the Supplementary Material.

### *Product-Group Construction for Scalability Test*

For the scalability test (Section 3.11), the 220 SKUs were partitioned by mean demand into a Fast group (73 SKUs), a Medium group (73 SKUs), and a Slow group (74 low-volume, high-variability SKUs; 220 is not divisible by three), and each group was trained independently with 1,000 chronological periods.

## Reward-based Explanations

### *Reward Decomposition*

To support business level explanations of the agent’s decision-making behavior, this study applies a reward decomposition mechanism, in which the total reward at time  is represented as the sum of semantically meaningful reward components:

where

•  : level of demand satisfaction and stockout penalty

•	: inventory holding cost

•	: loss due to spoilage or expiration

•	: ordering and transportation cost

•  wc​ are weighting coefficients that reflect the relative priority level of each operational objective.

Similarly, the action value function is also decomposed into the sum of individual value components.

where each * *denotes the contribution of a specific reward component to the action value; in this study it is evaluated post hoc as a one-step component value (see “Learning Procedure for the Decomposed Values”), not learned as a separate long-term value.

To ensure mathematical rigor as well as consistency with the explanation mechanism, each component is specifically defined as follows.

## •    Service: Reflects the level of demand fulfillment at each time step. A high value indicates that the system performs well in serving demand and limiting stockouts.

Where:

•    is the available inventory at time  ,

•    is the realized demand.

## •   Holding: Represents the cost of maintaining inventory. This component helps prevent excessive stock accumulation.

Where:

•   is the holding cost per unit,

•   is the ending inventory level.

## •   Waste: Reflects losses due to damaged or expired items, encouraging efficient inventory management.

Where:

•   is the cost per unit of damaged or expired items,

•   ​ is the quantity of damaged items at time .

## •   Ordering: Represents the cost incurred when placing orders, including both fixed and variable costs, helping balance order frequency and order size.

Where:

•   is the order quantity,

•   is the variable cost per unit ordered,

•   is the fixed cost per order,

•   is an indicator variable equal to 1 if an order is placed and 0 otherwise.

### *Learning Procedure for the Decomposed Values*

During evaluation, for each state-action pair , the value of component  is computed as , where . Here,  is obtained directly from the corresponding closed-form reward equation defined above. This calculation uses the trained model only to determine the evaluated actions and does not introduce any additional parameters or training steps. Thus, the component values are obtained through post-processing of the reward function rather than through an additional network head. 

The proposed procedure differs from the decomposed Q-learning method of Juozapaitis et al. (2019) [6]. In their approach, each reward component is represented by a separate value function  and trained using a component-specific Bellman target The component value functions are trained together with the main Q-function, such that their sum approximates the learned , subject to training error. In the present study,  represents the immediate reward of component  rather than a learned long-term value. Therefore, holds exactly for the four reward components. This equality applies to the one-step reward and does not imply that the sum is equal to the network's discounted multi-step . The difference between these two quantities is evaluated in Section 3.3.3.Section 3.3.4 implements this trained decomposition for both agents and uses it to measure how far the one-step explanations depart from long-horizon ones.

### Reward Difference Explanation

RDX is used to explain why the agent prefers the action  over  in the same state , that is, when . According to Juozapaitis et al. (2019) [6], RDX is defined as the difference in values between the two actions:

with the reward decomposition mechanism, the action value is represented as the sum of component values (one-step values evaluated post hoc; see “Learning Procedure for the Decomposed Values”) ; therefore, RDX can be decomposed according to individual operational objectives:

Where

•  : component * *supports choosing 

•  : component * *opposes 

In the inventory management problem, each  corresponds directly to a business objective (service/stockout, holding cost, waste, order cost). As a result, RDX helps reveal the trade-offs by showing which objectives favor the selection of  and which objectives incur disadvantages, thus enabling readers to more easily compare the contributions between different objectives.

### Minimal Sufficient Explanation:

Although RDX provides detailed information on the contribution of each reward component, the number of reasons can become large and difficult to interpret in practice. To address this limitation, MSX is proposed to identify the smallest subset of key reasons that is still sufficient to justify the agent’s decision. From RDX, we consider two sets of reward components:

where  and  respectively represent the reasons supporting and opposing the selected action* *. The total disadvantage of action  relative to  is determined by:

 is defined as the smallest subset  such that the total advantage of the components in  exceeds the entire disadvantage :

The  represents the main reasons that lead the agent to prefer action 

 is the smallest subset * *such that:

  indicates which disadvantages make the reasons in  necessary. MSX provides a compact justification set for the agent’s decision by identifying the key reward components that support the selected action. In this study, the faithfulness of MSX is further evaluated through reward-component masking experiments, which test whether removing MSX-identified components changes the selected action or affects the estimated value of the decision.

## Reward Coefficient Justification

The composite reward combines service, holding, waste, and balance terms with a constant base that normalizes the scale. Because the public retail dataset has no monetary cost annotations, unit weights are adopted as a neutral reference, not an empirically optimal choice; capacity is a multiple of average demand and the waste rate lies within the range reported for perishable goods. How the conclusions change when the weights deviate is examined in Section 3.5.

### A2C_mod Algorithm and Its Relation to Conventional A2C

A2C_mod follows Meisheri et al. [4] for 220 products, each described by three features (inventory, demand, waste) and fourteen ordinal replenishment levels. A per-product actor (four-layer feed-forward network with dropout) outputs a distribution over the levels and a shallower two-layer critic estimates the state value; both are trained for 600 episodes with a discount factor close to one. The critic keeps the temporal-difference update, whereas the actor is trained by squared-error distillation towards a soft target obtained by shifting the log-probabilities of all levels in proportion to the advantage divided by a distance-dependent term and renormalizing with a softmax. Because the levels are ordered, this distance weighting spreads the advantage to neighboring levels and attenuates distant ones (a Laplacian-like smoothing), favoring gradual adjustments, and the squared (Brier) loss is a proper scoring rule with bounded gradients that reduces variance at modest reward scales.

Relative to the conventional synchronous A2C, entropy is computed but not optimized, advantages are single-step temporal differences without normalization, and actor and critic use separate optimizers (the full comparison is provided in the Supplementary Material). The absence of entropy optimization favors exploitation over exploration and is a conservative bias, not an improvement.

### Common Quantity for Cross-Agent Comparison

Because the value-based and actor-critic agents output quantities of different type and scale, a dual common quantity places both on an equal footing without retraining. Feature-level attribution needs a vector over the fourteen actions: the portfolio-averaged softmax of the action values for DQN and the portfolio-averaged policy for A2C_mod, averaging over the 220 products being justified by per-product weight sharing; for A2C_mod the equivalent action value (log-policy plus state value) is an order-preserving proxy, not an estimate of the true Q-value. Reward-level decomposition needs a scalar, so a model-agnostic one-step lookahead simulates the transition under the best and second-best actions and splits the reward gap by objective. No single transformation serves both families, so the duality is intentional. Its validity rests on order preservation (softmax and logarithm are monotonic), scale compatibility (normalization to the probability simplex), and smoothness (Laplace-like weighting along the ordered action line).

On fifty held-out states per scenario (fixed seed), the optimal action is preserved in 100% of states in all three scenarios, while the rank correlation between the two families over the remaining actions is only 0.29 to 0.32, as expected from their distinct learning mechanisms; preserving the optimum is the minimal requirement for a fair comparison.

## Feature-based Explanation with SHAP  

In addition to reward space based explanation methods, this study employs SHAP to analyze the agent’s decision-making behavior in the input state feature space. SHAP enables quantification of the contribution of each state feature to the agent’s value estimate or policy, thereby clarifying the key signals that the DRL model relies on when making decisions.

### SHAP for DQN 

For the DQN agent, SHAP is applied to explain the action value function . Specifically, the  value of an action in state  is approximated as the sum of the contribution of features:

where  denotes the contribution of the state feature to the evaluation of the action , and  is the baseline value. This approach makes it possible to identify which state factors have the greatest influence on the DQN agent’s preference for a particular action, thereby providing an intuitive explanation at the input feature-level.

### SHAP for A2C_mod

For the agent A2C_mod, SHAP is used to explain the output of the actor or critic network, including the action policy  or the state value function . Similarly to DQN, the model output is represented as the sum of feature contributions:

where  corresponds to the probability of action or the estimate of the value of the state. Applying SHAP to A2C_mod helps clarify which state features govern the probability distribution of action or the estimation of values, thus supporting the analysis of the behavior of the Actor-Critic agent at the level of observable features.

### SHAP Implementation Details

SHAP is implemented using KernelSHAP with shap.KernelExplainer, which treats the trained DRL agents as black-box models. This model-agnostic setting allows the same attribution procedure to be applied to both DQN and A2C_mod, despite their different learning architectures. For DQN, SHAP values are computed with respect to the selected-action Q-value , where . For A2C_mod, SHAP values are computed with respect to the selected-action policy output . The background distribution is constructed from representative normalized inventory states. Inventory and demand-related variables are sampled within their operational ranges, while waste-related variables are generated conditionally on inventory to preserve the main dependency structure observed in inventory systems. A total of 200 background states are generated and then subsampled to 100 representative samples for SHAP estimation. During masking, KernelSHAP marginalizes missing features using the background distribution rather than replacing them with zero, since zero may correspond to meaningful operational conditions such as no inventory or no demand. Because inventory, demand, waste, and capacity-related variables may be correlated, SHAP values are interpreted as associative feature attributions rather than causal effects. The use of a structured background distribution helps reduce unrealistic out-of-distribution perturbations, but it does not fully remove the influence of feature dependence. Therefore, SHAP is used as a feature-space interpretation tool and is complemented by perturbation-based faithfulness evaluation. The main implementation settings of SHAP are summarized in Table 4.

**Table 4. SHAP implementation configuration.**

**Component**

**Configuration**

SHAP variant

KernelSHAP using shap.KernelExplainer

Explained output for DQN

Selected-action Q-value 

Explained output for A2C_mod

Selected-action policy output 

Background distribution

Representative normalized inventory states

Background size

200 generated states, subsampled to 100 representative samples

Masking strategy

Marginalization using the background distribution

Aggregated features

Inventory, Demand, Waste

Additional analysis

Top-k SKU-level/system-level SHAP analysis

Correlated features

Interpreted as associative attribution, not causal effect

### SHAP Experimental Setup and Validity Verification

SHAP values are treated as associative, not causal, attributions of each feature to the model output for a given state; both agents are handled as black-box functions so the same principle applies to the value-based and actor-critic models, and correlated inventory, demand, waste, and capacity variables are interpreted accordingly.

*2.3.4.1 SHAP Background Construction*

For the macro analysis each state has three aggregated features (inventory, demand, waste, with waste a bounded linear function of inventory plus controlled noise). Two hundred synthetic states are generated with a fixed seed and 100 are extracted for explanation, either by random sampling (primary baseline) or as KMeans centroids (ablation); the two subsets have nearly identical statistics and pairwise-distance distributions, so the random baseline is sufficiently representative.

*2.3.4.2 Background Sensitivity*

Stability was tested on a grid over background origin (synthetic uniform versus trajectory from the first test steps), size (three levels, baseline 100), and sampling strategy (random versus KMeans), for both agents, at the macro level (three features, KernelSHAP) and the micro level (660 features, Partition Explainer), on the same medium-scenario test states. At the macro level, rankings are insensitive to size and sampling (rank correlation 1.00 within synthetic backgrounds) but shift when the origin changes to trajectory (0.87, a single rank swap), and feature coverage at a fixed threshold is higher for synthetic backgrounds. At the micro level, magnitudes stay similar (similarity 0.94 to 0.99) but rankings are unstable (mean rank correlation near zero for every background), and the full grid on 50 states took over four hours, so micro-level rankings are treated as indicative.

*2.3.4.3 Validity of Perturbed States*

Perturbed states were checked against the physical constraints of the inventory problem (bounds on inventory, demand, and waste, and the waste-inventory relation within tolerance) on 5,000 states per scenario for the macro case and one thousand median-masked states for the micro case. Trajectory backgrounds are fully valid in all scenarios; synthetic backgrounds reach 95.5%, 98.8%, and 84.5% in the EASY, MEDIUM, and HARD scenarios (the hard case is a cautionary note), and micro-level median masking is fully valid.

*2.3.4.4. Consistent SHAP Implementation*

The macro analysis applies KernelSHAP to the aggregated three-feature states (exact, seconds per evaluation) and the micro analysis applies Partition Explainer to raw 660-dimensional states (near-linear scaling, minutes per evaluation). Both use 100 representative background states drawn from 200 synthetic states with a fixed seed, complemented by trajectory ablations; the macro level shows which feature group drives the decision and the micro level which product within it contributes most.

# Results and Discussion

## Evaluation Schema

The evaluation schema in this study is designed to simultaneously assess the decision-making effectiveness of DRL agents and the quality of XRL explanations (RDX, MSX, SHAP) in multi-product inventory management problems. The evaluation process is not only based on traditional operational performance metrics but also considers the transparency and interpretability of agent behavior. The evaluation framework is composed of three core elements: the dataset, the training–testing setup, and the scheme used to assess explanation methods.

### *Dataset*

This study directly adopts the DRL agents DQN and A2C_mod that have been trained in the multi-product inventory management environment. These agents are trained in the publicly available retail dataset Instacart Market Basket Analysis[25] , as presented in Table 5.

The 220 products and simulation inputs were generated by a preprocessing pipeline adapted from Meisheri et al. (2020) [4]. Products were ranked by order frequency over the complete Instacart history, the top 20% (about 10,000 products) were restricted to the 12 grocery departments considered here, and 220 were sampled uniformly at random. Because Instacart has no inventory records, inventory variables were generated synthetically: order timestamps were reconstructed from day-of-week, hour, and inter-order gap and discretized into 6-hour cycles, demand is the number of order occurrences per cycle, shelf capacity follows a heuristic of about three days of average throughput, and initial inventory is drawn from Uniform(0,1).Instacart records purchases only. It contains no inventory levels, replenishment orders, shelf capacities, costs, or shelf-life information. Table 5 therefore separates the variables observed in the data from those derived from it and those simulated, gives the reason for each assumption, and lists the data segment and number of states used in each analysis.

**Table 5. Data sources, simulation assumptions, and data segments used in each analysis.**

**Item**

**Value or construction**

**Source and justification**

**Observed in Instacart [25]**

Records used

Orders and order product lines (prior and train files) for the selected products

Observed. Purchase records only: no inventory, replenishment, capacity, cost, or shelf-life data.

Products

220 products: top 20% by order frequency, restricted to 12 grocery departments, sampled uniformly at random

Observed identifiers; selection procedure adapted from [4].

Order timing

Day of week, hour of day, and days since the customer’s previous order

Observed (relative timing only; no calendar dates).

**Derived from the observed records**

Timestamps and decision cycle

First order of each customer placed on a random date in the first week of January 2017 matching its recorded weekday; later orders placed by the recorded inter-order gaps; timestamps grouped into 6-hour cycles

Derived with a random calendar anchor, because Instacart provides relative timing only.

Demand dᵢ,ₜ

Number of order lines containing product i in cycle t, each counted as one unit; 47.7% of SKU-cycles are zero in training and 69.7% in testing

Derived. Instacart records no purchase quantities, so one unit per order line is assumed.

Horizon and split

1,504 cycles (≈376 days); training cycles 1- 900 (60%), test cycles 901 - 1,504 (604 cycles, 40%)

Chronological split without shuffling (Section 2.1.2).

**Simulated or assumed**

Shelf capacity Cᵢ

⌈12 × mean demand per cycle⌉, about three days of average throughput; 4 - 602 units (mean 25.1)

Simulated heuristic following [4]; no capacity data. Computed on the full series (see leakage audit, Table 3).

Initial inventory

Uniform(0, 1) fraction of capacity per SKU, redrawn at every training episode; one stored draw used for all test runs, scaled by the scenario inventory factor (Table 8)

Simulated; no inventory records. Random starts expose the agents to a wide range of stock levels during training.

Replenishment lead time

Zero: an order placed in a cycle is available in the same cycle

Assumption; no delivery data. Keeps the state Markovian without an order pipeline.

Spoilage

ρ = 2.5% of current stock per cycle; 1%, 5%, and 15% in the EASY, MEDIUM, and HARD scenarios

Simulated; no shelf-life data. Scenario rates bracket the training rate (Table 8).

Costs and reward weights

Dimensionless, capacity-normalized penalty terms with unit weights

Assumption; no monetary costs in the data. Sensitivity to the weights is examined in Section 3.5.

**Data segment and number of states used in each analysis**

Agent training

600 episodes × 900 steps on training cycles 1 - 900 (Table 2)

Training segment only.

Base-stock calibration

Fit on cycles 1-720, safety factor selected on cycles 721 - 900, targets recomputed on cycles 1- 900

Training segment only (Section 3.1.3).

Operational evaluation and reconstruction

Full test segment: 604 cycles; 604 × 220 = 132,880 state - action pairs per scenario and agent

Tables 6, 7, and 9.

RDX/MSX analyses

496 states from the test segment per scenario

Sections 3.3 - 4.5 and 3.7.3.

**Trained runs (seeds) used in each analysis**

Operational comparison

5 seed:42, 123, 256, 512, 1024 for both agents

Tables 6 - 7.

Reconstruction error

One run per agent: seed-42 checkpoints (DQN and A2C_mod)

Table 9.

Trained reward decomposition

Five  runs per agent, retrained with seeds 42, 123, and 256, 512, 1024, compared with the original runs of the same seeds

Table 10.

RDX/MSX ablation, λ sensitivity, Stability

One run per agent: DQN run main run; A2C_mod main run

Sections 3.3 - 4.5 and 3.7.

Reward-weight effect on total reward; joint weights

One run per agent: seed-42 checkpoints

Section 3.5.

Global and local SHAP

One run per agent: a DQN trained on the 3-feature per-product state (32 hidden units); A2C_mod main run

Section 3.6.1 - 3.6.2.

FCS, micro-level SHAP, SHAP faithfulness

One run per agent: DQN variant with 32 hidden units; A2C_mod main run; FCS repeated 5 times on the same run

Tables 13 - 15 ;Sections 3.6.3 - 3.6.7 and 3.8.3.

Action-space resolution

One run per agent and resolution (7, 14, 28 levels)

Section 3.9; Table 17.

Scalability by product group

One run per agent and product group

Section 3.11.

### *Experimental Design*

The experimental design evaluates, in parallel, the performance and explainability of the two agents, DQN and A2C_mod, within the same inventory management environment. Agents are trained once on the training segment and then evaluated, without retraining, on the held-out test segment and on three controlled stress-test scenarios derived from it.. During evaluation, agent behavior is analyzed using XRL methods: RDX and MSX to explain decisions in the reward space, and SHAP to analyze the influence of state features, creating a direct linkage between behavior and explanation. This procedure enables a direct connection between the agent’s decision-making behavior and the corresponding explanations throughout the evaluation process.

*3.1.3. Classical Inventory Baseline*

To provide an operational reference for evaluating the DRL agents, this study introduces a classical base-stock policy as a non-learning inventory-control baseline. This baseline is used to examine whether the learned DRL policies are competitive with a standard inventory replenishment rule. For each product , the target inventory level is defined as:

where and denote the mean and standard deviation of historical demand for product , respectively. The parameter is a safety factor that controls the trade-off between stockout risk and inventory cost. At each time step , the replenishment quantity is computed as:

where is the current inventory level of product . The resulting replenishment quantity is clipped to the normalized capacity range and mapped to the nearest available discrete replenishment action. In this study, is selected from:

To avoid using the test data when calibrating the baseline policy, the safety factor  was determined using a validation period taken exclusively from the training data. Specifically, the first 80% of the training period, corresponding to 720 out of 900 cycles, was used to estimate the mean demand () and standard deviation () for each product. The remaining 180 cycles were reserved for validation. We then considered five candidate values of the safety factor, , and simulated the corresponding base-stock policies over the validation period. The value providing the highest average validation reward was selected, resulting in .

After the safety factor was selected, each product’s target inventory level was recalculated from demand statistics over the full 900-cycle training period, and the resulting base-stock policy was compared with DQN and A2C_mod on the untouched test set under identical test cycles, capacity constraints, and metrics, so that the test period is used solely for final evaluation.

##  Operational Performance Comparison with Classical Baseline

To evaluate the operational competitiveness of the learned DRL policies, this study compares DQN and A2C_mod with the classical Base-Stock policy under the same testing conditions. The comparison is conducted for the overall test set, as the mean over five independently trained seeds per model, for the overall setting and for the three controlled stress-test scenarios, EASY, MEDIUM, and HARD. The results are summarized in Table 6.

**Table 6. Performance comparison with the classical Base-Stock baseline(k=1.0).**

**Testing scenario**

**Evaluation metric**

**Base-Stock**

**DQN**

**A2C_mod**

**All scenarios**

Total Reward ↑

+0.8567

+0.7197 ± 0.0396

+0.6939 ± 0.1463

Service Level ↑

+0.9800

+0.9999 ± 0.0001

+0.9997 ± 0.0007

Holding Cost ↓

+0.0000

+0.1110 ± 0.0630

+0.0935 ± 0.1495

Waste Cost ↓

+0.0111

+0.0672 ± 0.0008

+0.0672 ± 0.0008

Ordering Cost ↓

+0.1121

+0.1020 ± 0.0258

+0.1478 ± 0.0301

Stockout Rate ↓

+0.0200

+0.0001 ± 0.0001

+0.0003 ± 0.0007

**EASY scenario** 

Total Reward ↑

+0.9320

+0.8304 ± 0.0533

+0.8194 ± 0.1546

Service Level ↑

+0.9786

+1.0000

+1.0000 ± 0.0001

Holding Cost ↓

+0.0000

+0.1145 ± 0.0718

+0.1010 ± 0.1579

Waste Cost ↓

+0.0005

+0.0098 ± 0.0001

+0.0098 ± 0.0002

Ordering Cost ↓

+0.0461

+0.0453 ± 0.0258

+0.0698 ± 0.0424

Stockout Rate ↓

+0.0214

+0.0000

+0.0000 ± 0.0001

**MEDIUM scenario**

(x=60%,sales=50%,waste=5.0%)

Total Reward ↑

+0.8621

+0.7484 ± 0.0498

+0.7160 ± 0.1557

Service Level ↑

+0.9808

+0.9999 ± 0.0001

+0.9996 ± 0.0008

Holding Cost ↓

+0.0000

+0.1111 ± 0.0665

+0.0964 ± 0.1553

Waste Cost ↓

+0.0057

+0.0484 ± 0.0007

+0.0475 ± 0.0010

Ordering Cost ↓

+0.1131

+0.0921 ± 0.0183

+0.1396 ± 0.0359

Stockout Rate ↓

+0.0192

+0.0001 ± 0.0001

+0.0004 ± 0.0008

**HARD scenario** 

Total Reward ↑

+0.7761

+0.5803 ± 0.0197

+0.5462 ± 0.1303

Service Level ↑

+0.9805

+0.9997 ± 0.0002

+0.9994 ± 0.0011

Holding Cost ↓

+0.0000

+0.1076 ± 0.0534

+0.0831 ± 0.1354

Waste Cost ↓

+0.0272

+0.1432 ± 0.0018

+0.1363 ± 0.0040

Ordering Cost ↓

+0.1772

+0.1686 ± 0.0392

+0.2339 ± 0.0180

Stockout Rate ↓

+0.0195

+0.0003 ± 0.0002

+0.0006 ± 0.0011

Base-Stock remains a strong classical baseline. It attains the highest total reward overall (+0.8567, versus +0.7197 ± 0.0396 for DQN and +0.6939 ± 0.1463 for A2C_mod) and in every scenario (EASY +0.9320, MEDIUM +0.8621, HARD +0.7761). The operational metrics differ: Base-Stock incurs no holding cost and a lower waste cost (0.0111 versus 0.0672 for both agents overall), but its service level is lower (0.9800, against above 0.999 for both agents in all scenarios) and its stockout rate higher (0.0200 versus 0.0001 for DQN and 0.0003 for A2C_mod). The trade-off persists when the safety factor is selected on a separate validation period: the static order-up-to policy has lower inventory-related costs but avoids stockouts less effectively than the state-dependent DRL agents.

### *Why Prefer a DRL Policy When Base-Stock Attains a Higher Total Reward?*

A higher total reward does not by itself make DQN or A2C_mod unsuitable, for three reasons. First, the unit-weighted reward does not represent a grocery retailer’s cost structure: Base-Stock stocks out in roughly 2% of product-periods, which for perishable and fast-moving items means lost sales and lower customer satisfaction, whereas excess stock can partly be recovered through markdowns, so service may be treated as a minimum operational constraint rather than a cost to trade freely. Second, Base-Stock uses a fixed target computed from historical demand statistics, whereas the DRL agents act on the current state, which is reflected in stockout rates one to two orders of magnitude lower even after validation-based calibration. Third, the study aims to interpret how DRL agents use state information with RDX, MSX, and SHAP, which a fixed-formula policy does not offer; Base-Stock is a classical reference, not a replacement target.

### *Quantified Trade-off Between Service Level and Holding/Waste Cost*

Table 7 reports the differences relative to the corrected Base-Stock reference (k = 1.0). DQN reduces the stockout rate by 0.0199 (about 99.5%) and raises service by 2.0 percentage points, at the cost of +0.1110 holding and +0.0561 waste cost, so total reward falls by 0.1370; A2C_mod behaves similarly (−0.1628). Fewer stockouts therefore do not numerically offset the added inventory costs under the unit-weighted reward, but this depends on a reward design that does not encode how a retailer values a stockout relative to excess inventory or waste.

***Table 7. Change relative to the corrected Base-Stock baseline.***

**Metric**

**Base-Stock (k=1.0)**

**DQN**

**Δ (DQN − Base-Stock)**

**A2C_mod**

**Δ (A2C_mod − Base-Stock)**

Service Level ↑

0.9800

0.9999

+0.0199

0.9997

+0.0197

Stockout Rate ↓

0.0200

0.0001

−0.0199

0.0003

−0.0197

Holding Cost ↓

0.0000

0.1110

+0.1110

0.0935

+0.0935

Waste Cost ↓

0.0111

0.0672

+0.0561

0.0672

+0.0561

Ordering Cost ↓

0.1121

0.1020

−0.0101

0.1478

+0.0357

Total Reward ↑

0.8567

0.7197

−0.1370

0.6939

−0.1628

## Reward Difference Explanation

### Controlled stress-test scenarios* and experimental setup*

To examine how the agents’ decisions and their explanations change under increasing pressure, the study defines three controlled stress-test scenarios with increasing demand and spoilage. Table 8 summarizes the detailed configurations of each scenario and their corresponding evaluation objectives. The three evaluation scenarios are designed to test the stability and discriminative ability of the XRL pipeline under controlled changes in demand and spoilage. They are derived from the empirical characteristics of the Instacart-based inventory data. The MEDIUM scenario uses the original demand distribution as the calibration baseline. The EASY scenario reduces demand intensity and spoilage risk to represent more stable products with lower operational pressure, while the HARD scenario increases both demand intensity and spoilage risk to simulate high-pressure perishable-product conditions. This design allows the same trained agents to be evaluated under controlled changes in demand and waste levels, thereby testing whether RDX, MSX, and SHAP explanations remain meaningful when the operating environment becomes more difficult. The scenarios are transformations of the same held-out Instacart-based test segment (scaling of initial inventory, demand, and spoilage rate in Table 8), not independent datasets or environments. They therefore probe robustness to controlled shifts in demand intensity and spoilage, not generalization to other retailers, product assortments, demand processes, or lead-time structures.

**Table 8.  Configuration of the controlled stress-test scenarios		**

**Attribute**

**Value**

**Characteristics**

Easy Scenario

Inventory 30%, Demand 20%, Spoilage rate 1%

This is an ideal situation to order aggressively to meet demand without significant risk of excessive inventory.

Medium Scenario

Inventory 60%, Demand 50%, Spoilage rate 5%

This is a balanced situation that requires the agent to carefully trade-off between ordering in anticipation of demand and the risk of holding inventory.

Hard Scenario

 Inventory 90%, Demand 80%, Spoilage rate 15%

This is a critical situation. Although demand is high, inventory is already near capacity, and the spoilage rate is high. The agent faces a major conflict: ordering more increases holding and disposal costs, while not ordering leads to missed sales.

**Empirical basis for the scenario settings.** The percentages in Table 8 are scaling factors, not percentile thresholds. Demand settings (20%, 50%, 80%) scale the Instacart-derived test demand (604 cycles × 220 products, mean 0.0354, 69.7% zeros) and give scaled means at about the 70th to 73rd percentile of real observations. Inventory settings (30%, 60%, 90%) scale only the synthetic initial stock and land near the nominal percentiles (0.270, 0.584, 0.889), later levels being endogenous. Spoilage settings (1%, 5%, 15%) cannot be derived from Instacart, which has no shelf-life data, and are scenario overrides spanning the trained 2.5% waste rate.

### *Experimental Results and Analysis*

Figure 2 presents the RDX results for the two agents, DQN and A2C_mod, in the three evaluation scenarios EASY, MEDIUM and HARD by decomposing the total reward into the components of service, storage, waste and order.** **

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 2. Reward Difference Explanation (RDX) analysis results for DQN and A2C_mod.**

In the EASY scenario, the reward structures of the two models are almost identical, with the service component playing the dominant role. Operational costs have only a minor impact, indicating that DQN and A2C_mod exhibit similar decision-making behavior in a stable environment. In the MEDIUM scenario, differences begin to emerge as A2C_mod shows a greater sensitivity to holding and ordering costs, while DQN continues to maintain a service-prioritized strategy. This reflects differences in how the two agents balance service quality and operational costs as uncertainty increases. In the HARD scenario, A2C_mod reduces its holding and ordering exposure whereas DQN keeps prioritizing service and incurs higher costs; both nevertheless obtain positive total rewards (Table 6: 0.5803 for DQN, 0.5462 for A2C_mod, five-seed means), so the difference lies in the priorities revealed by RDX, not in the sign or ranking of total reward, and is not by itself evidence of better operational performance. In general, the RDX results show that the two agents behave in a similar way in simple scenarios, but A2C_mod shifts its reward priorities more as scenario pressure increases. RDX analysis helps clarify the decision-making mechanisms and business trade-offs of each model. The comparison with the Base-Stock policy provides an operational reference for interpreting the RDX results. When a DRL agent achieves better service-level performance or lower stockout rate than the classical baseline, the RDX decomposition helps identify which reward components support this improvement. Conversely, when the DRL agent obtains a lower total reward than Base-Stock, RDX indicates whether this result is caused by higher holding, waste, or ordering-related penalties. Therefore, RDX not only explains the differences between DQN and A2C_mod, but also clarifies how the learned DRL policies differ from a conventional replenishment strategy.

Figure 3 compares the distribution of Q values for all actions between the two agents, A2C_mod and DQN, in the three scenarios EASY, MEDIUM and HARD. For A2C_mod, since the algorithm is based on a policy value mechanism, the Q values are transformed into an equivalent form to ensure direct comparability with DQN.

In the EASY scenario, A2C_mod relatively clearly distinguishes between the selected action and the remaining actions, while DQN exhibits high and uniform Q values between actions, indicating that the differentiation of actions is not pronounced in a simple environment. In the MEDIUM and HARD scenarios, A2C_mod shows increasingly clear separation, with the selected action having a markedly higher * *value, while DQN continues to maintain relatively uniform * *values. This reflects a stable but less flexible decision-making strategy as environmental complexity increases. These results indicate that A2C_mod more clearly differentiates between effective and less effective actions in challenging environments, whereas DQN exhibits threshold-like decision-making behavior with limited contextual adaptability.

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 3. Comparison of Q-value distributions of A2C_mod and DQN across different levels of environmental difficulty.**

Figure 4 presents the RDX results by comparing the selected action with alternative actions in individual reward components, including service, holding, waste, and order. The values in the chart represent the reward differences between the selected action and other options, thus indicating the relative advantages or disadvantages of the current choice for each component.

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 4. The heatmap shows the reward component differences in RDX.**

Positive difference values indicate that the selected action yields higher benefits than alternative actions for the corresponding component, while negative values indicate that there are still more advantageous options. The results show that in the EASY and MEDIUM scenarios, both models still exhibit some competing  actions that are more favorable in terms of cost, reflecting a decision space that is not yet clearly differentiated. In contrast, in the HARD scenario, A2C_mod shows a clear advantage, with the selected action showing positive differences between most components, while DQN still has alternative actions that are more favorable in terms of holding and ordering costs. Overall, the RDX plots provide an intuitive view of the rationality of the selected action compared to other options, while clarifying how the models balance the reward components under different environmental conditions.

### *Reconstruction Consistency of the Decomposed Value*

Section 2.2 defines each component value as a one-step reward rather than a separately learned value, so the four components sum exactly to the immediate total reward; this does not imply that their sum equals the action value learned by the agent (Q for DQN and, for A2C_mod, the Q-value-equivalent quantity of Section 2.2.7, also used for cross-architecture SHAP). We therefore evaluate the reconstruction error directly:

Both agents use a single seed and the checkpoints of the other experiments. The real Instacart-derived test set (604 timesteps × 220 products) is replayed with greedy actions under the three scenario transformations of Table 8, giving 132,880 state-action pairs per scenario and model (Table 9).

The reconstruction error is large relative to the one-step reward scale (mean reward per step of about 0.7 to 0.95), and the correlation between the learned value and the component sum is weak for DQN (r = 0.02 to 0.21) and negative for A2C_mod (r = −0.05 to −0.13). This is consistent with Section 2.2: component values are immediate rewards, whereas the learned Q-value is the expected discounted return (γ = 0.99), so the two are not expected to coincide.

RDX and MSX therefore describe the immediate trade-offs among the four reward components for the selected action, not a decomposition of the full multi-step Q-value; a separately trained decomposition, and the agreement between one-step and long-horizon explanations, are evaluated in Section 3.3.4.

**Table 9. Reconstruction error between the network’s learned action value and the one-step decomposed sum**

**Model**

**Scenario**

**Q mean**

**Σc Qc mean**

**Recon. error (mean ± SD)**

**RMSE**

**Corr(Q, Σc Qc)**

DQN

EASY

+18.163

+0.949

+17.214 ± 0.065

17.214

+0.019

MEDIUM

+18.131

+0.845

+17.286 ± 0.075

17.286

+0.181

HARD

+18.091

+0.676

+17.415 ± 0.094

17.415

+0.210

A2C_mod

EASY

+259.209

+0.811

+258.399 ± 78.498

270.059

-0.015

MEDIUM

+244.334

+0.782

+243.551 ± 75.519

254.991

-0.048

HARD

+238.527

+0.695

+237.832 ± 72.643

248.679

-0.126

### *Ablation Validation with a Genuinely Trained Reward Decomposition*

## Section 3.3.3 used a post-hoc decomposition from single-head checkpoints never trained to produce component values, and found a large reconstruction gap (Table 9). To test whether this gap is an artifact of the post-hoc procedure, a genuinely trained decomposition was implemented following Juozapaitis et al. (2019): DQN receives four parallel Q-heads and A2C_mod four component critics (service, holding, waste, ordering), so the components sum to the action value by construction. Trunks, hyperparameters, and budget are unchanged, only the output head is decomposed, and each algorithm was retrained from five seeds (42, 123, 256, 512, 1024); Table 10 compares mean total reward per step with the five original runs.

To test whether learning a decomposed value changes the policy, the decomposed runs were compared with the five original single-value runs (Table 10). For each model and scenario we report the difference in mean total reward with a 95% percentile-bootstrap interval over seeds (20,000 resamples), Hedges’ g, and a two one-sided test (TOST) of equivalence with a margin of ±0.10 reward per step. No difference between the original and decomposed agents was detected: every bootstrap interval includes zero, for both models and all three scenarios. The point estimates favour the decomposed DQN in MEDIUM and HARD (+0.09 and +0.17; g = 0.31 and 0.88) and the original DQN in EASY (−0.05). For A2C_mod the differences are small (−0.05 to +0.02; |g| ≤ 0.12) relative to the spread between runs. Equivalence could not be established either: with five seeds per arm and between-seed standard deviations of 0.10–0.32 for DQN and 0.38–0.41 for A2C_mod, no TOST p-value fell below 0.05 at a margin of ±0.10 or ±0.05.

A2C_mod collapsed in two of five runs in each arm (original: seeds 123 and 256; decomposed: seeds 256 and 1024), so the decomposition neither causes nor removes this failure mode. We therefore conclude only that decomposing the value function did not produce a detectable change in policy performance; establishing equivalence would require more seeds.

## 

**Table 10. Mean episodic total reward of the trained-decomposition checkpoints versus the original single-head checkpoints**

**Model**

**Scenario**

**Original**

**Decomposed**

**Δ [95% CI]**

**Hedges’ g**

**TOST p**

DQN

EASY

0.835 ± 0.102

0.789 ± 0.188

−0.046 [−0.218, 0.118]

−0.27

0.30

MEDIUM

0.643 ± 0.319

0.732 ± 0.184

+0.089 [−0.170, 0.403]

0.31

0.47

HARD

0.537 ± 0.198

0.703 ± 0.137

+0.166 [−0.014, 0.356]

0.88

0.72

A2C_mod

EASY

0.472 ± 0.406

0.420 ± 0.382

−0.053 [−0.485, 0.387]

−0.12

0.43

MEDIUM

0.382 ± 0.398

0.364 ± 0.388

−0.019 [−0.454, 0.421]

−0.04

0.38

HARD

0.240 ± 0.385

0.259 ± 0.395

+0.019 [−0.417, 0.450]

0.04

0.38

The decomposed agents also allow a direct check of how far the one-step explanations used in this paper depart from long-horizon ones. For the same state and the same pair of actions (greedy and second-best), we compared the one-step RDX with a long-horizon RDX built from the trained component values: ΔQ_c = Q_c(s, a₁) − Q_c(s, a₂) from the four DQN heads and, because the A2C_mod critics are state-value functions, ΔQ_c = [r_c(s, a₁) + γV_c(s′_a₁)] − [r_c(s, a₂) + γV_c(s′_a₂)]. MSX was computed with the same rule for both. States were taken every sixth step of the real test trajectory under each agent’s own policy (101 steps × 220 SKUs per run).

The two explanations agree only partly (Table 11). The dominant component coincides in 47–55% of DQN decisions and in 28–64% of decisions for non-collapsed A2C_mod runs, and the MSX sets are identical in only 10–13% (DQN) and 22–49% (A2C_mod) of cases. The disagreement is systematic. Service is almost never dominant in the one-step RDX (at most 2% of decisions), because a one-step stockout difference appears only when exactly one of the two actions empties the SKU in the next period. In the long-horizon RDX, service dominates 41–51% of DQN decisions and 29–56% of A2C_mod decisions, while the share dominated by the order (inventory-spread) component falls from 60–97% to 7–58%. The two collapsed A2C_mod runs agree almost perfectly because both explanations point to holding when the policy orders the maximum, so they are excluded from Table 11. The one-step RDX and MSX reported in this paper should therefore be read as explanations of the immediate trade-off; they understate how much the agents’ longer-horizon value relies on avoiding future stockouts. These results are obtained on the decomposed agents, whose performance did not differ detectably from the original agents (Table 10).

**Table 11. Agreement between one-step (post-hoc) and long-horizon (trained-decomposition) reward explanations for the same decision (best vs. second-best action).**

**Model**

**Scenario**

**Same dominant component**

**MSX Jaccard**

**MSX identical**

**Service dominant: one-step → long**

**Order dominant: one-step → long**

DQN

EASY

0.47 ± 0.36

0.33 ± 0.21

0.10 ± 0.05

0.00 → 0.51

0.92 → 0.48

DQN

MEDIUM

0.55 ± 0.30

0.40 ± 0.18

0.13 ± 0.08

0.00 → 0.43

0.97 → 0.56

DQN

HARD

0.55 ± 0.28

0.40 ± 0.16

0.12 ± 0.06

0.02 → 0.41

0.93 → 0.58

A2C_mod

EASY

0.64 ± 0.37

0.63 ± 0.32

0.49 ± 0.41

0.00 → 0.29

0.60 → 0.27

A2C_mod

MEDIUM

0.35 ± 0.42

0.34 ± 0.40

0.29 ± 0.42

0.00 → 0.56

0.71 → 0.08

A2C_mod

HARD

0.28 ± 0.34

0.36 ± 0.26

0.22 ± 0.37

0.00 → 0.52

0.73 → 0.07

- ***Minimal Sufficient Explanation*** 

MSX is constructed based on RDX to identify the minimal set of core reward components that is still sufficient to justify the agent’s decision. While RDX provides a detailed analysis of differences between actions between individual reward components, MSX focuses on clarifying the indispensable factors in the decision-making rationale. In this study, MSX is applied to both DQN and A2C_mod in the EASY, MEDIUM and HARD scenarios to analyze how the structure of the agents’ reasoning changes as environmental complexity increases.

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 5. MSX evaluation based on the influence level of each reward component.**

# The results shown in Figure 5 indicate that, in all three scenarios, sequentially removing each individual reward component does not change the agent’s decision for either model. This observation suggests that the decision does not rely on a single factor but is instead formed through a stable combination of multiple reward components. Even in the HARD scenario, where risk levels and objective conflicts are higher, no component, when removed, can reverse the decision. Therefore, there exists no single reward component that is minimal and sufficient to explain the decision, thus confirming the multi-factor nature of the decision-making process in the inventory management problem. 

# However, when MSX is examined from the perspective of the minimal subset of reward components, the models exhibit markedly different reasoning structures, as shown in Figure 6. For A2C_mod, in the EASY and MEDIUM scenarios, the MSX set includes only the service component, indicating that demand fulfillment serves as the core factor in low and medium risk environments. When moving to the HARD scenario, the MSX set shifts to the holding component, reflecting an adjustment of the model’s core reasoning factor under increased inventory cost pressure. In contrast, for DQN, the MSX set in all three scenarios consistently includes only the holding component, indicating a stable minimal explanation structure that is largely independent of environmental context. These results clarify the difference in decision making flexibility between the two models, with A2C_mod demonstrating better context dependent adaptability than DQN.

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 6. Qualitative–quantitative summary chart of the MSX results.**

## Sensitivity Analysis

**	**To evaluate the degree of dependence of reward-based explanations (RDX and MSX) on weight design, the study conducts a sensitivity analysis on the weighting coefficients of each reward component. Below, the sensitivity analysis framework used to examine the stability of explanations when changing the priority levels among operational objectives is presented.

The total reward at time *t* is determined by the following formula:

	Baseline setup:

	The sensitivity analysis is conducted using the One-At-a-Time (OAT) method, in which only one weight is adjusted at each step while the remaining weights are kept unchanged. Specifically, for a component , the weight is adjusted as follows:

Where:

	The total reward after adjustment becomes:

	Since RDX reflects the contribution of each component to the Q-value, when the weight changes, the contribution of component  changes proportionally with :

	This procedure is performed sequentially for each component and for all three scenarios EASY, MEDIUM, and HARD presented in Table 12.

**Table 12. Sensitivity analysis setup by scenario and architecture.**

**Scenario**

**Fixed settings**

**Algorithm**

**Varied reward weight(s)**

**Weight factors tested (λ)**

EASY

Inv 30%, Dem 20%, Spoil 1%

A2C_mod

(

{0.5, 1.0, 1.5, 2.0}

DQN

MEDIUM

Inv 60%, Dem 50%, Spoil 5%

A2C_mod

DQN

HARD

Inv 90%, Dem 80%, Spoil 15%

A2C_mod

DQN

	![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 7. Sensitivity analysis of the average RDX with respect to the coefficient λ in the EASY, MEDIUM, and HARD scenarios for DQN and A2C_mod.**

	Figure 7 illustrates the variation of the average RDX value with respect to the adjustment coefficient λ across three scenarios (EASY, MEDIUM, HARD) for the two architectures DQN and A2C_mod. Overall, RDX increases with λ, indicating that the explanation structure depends on the amplification level of the reward weights. In all scenarios, the ordering cost component (worder)  exhibits the highest sensitivity, with a steep slope as λ increases, while the holding cost (whold) increases at a lower rate. In contrast, the service (wsrv) and waste (wwaste) components maintain small RDX values with minimal fluctuation, reflecting relative stability under weight changes. A comparison between the two architectures shows that DQN exhibits significantly larger variation amplitudes, particularly in the MEDIUM and HARD scenarios, where RDX increases sharply as λ increases. Meanwhile, A2C_mod maintains lower and more stable RDX values, indicating that its explanations are less sensitive to reward weight adjustments. As environmental complexity increases, the variation level of RDX also increases, most noticeably for DQN. These results suggest that the RDX explanation structure is primarily influenced by the ordering cost component, and that the stability of explanations depends considerably on the reinforcement learning architecture employed. In addition to analyzing the variation magnitude of RDX values, the study further examines changes in the MSX structure as λ varies. This analysis aims to assess whether the minimal explanation set maintains consistency when the priority levels among operational objectives are adjusted.![placeholder](https://markdowntoword.io/placeholder.png)

**Fig 8. Frequency of occurrence of reward components in the MSX set with respect to the coefficient  across the EASY, MEDIUM, and HARD scenarios for DQN and A2C_mod.**

	Figure 8 presents the frequency of occurrence of reward components in the MSX set with respect to the coefficient  across three scenarios and two architectures. The results show that the ordering cost (worder) dominates in most configurations, with frequencies ranging from approximately 50% to nearly 100%, particularly high in the EASY scenario of DQN. The holding cost (whold) plays a secondary role, becoming more pronounced in the MEDIUM and HARD scenarios of DQN (around 30–40%), while the service (wsrv) and waste (wwaste) components appear at relatively low frequencies. From an architectural perspective, DQN exhibits a redistribution of components as environmental complexity increases: the proportion of worder  gradually decreases from EASY to HARD, while whold increases correspondingly. However, within each individual scenario, the MSX structure remains relatively stable with respect to λ. In contrast, A2C_mod shows a clearer context dependence: in the EASY scenario, the MSX structure varies with λ, whereas in the HARD scenario, the component frequencies remain nearly unchanged, reflecting high stability under weight adjustments. The MSX results complement the RDX analysis by demonstrating that explanation magnitude and explanation structure are two distinct aspects. While RDX measures the intensity of contribution changes, MSX reflects the stability of the core reasoning set, thereby clarifying behavioral differences between DQN and A2C_mod under different environmental conditions.

### *Total reward and action invariance under single-weight reweighting*

Because the weights are applied after the trained policy has selected its action, the selected action is unchanged under reweighting (verified for all 96 combinations of agent, scenario, component, and weight factor), so the analysis isolates the effect of the weights on reward values and on RDX/MSX with the policy fixed. Reweighting service produces the largest change in total reward for both agents: because stockouts are rare, the service term is close to 1.0 per product, so scaling its weight by λ changes total reward by approximately λ − 1 (Figure 9). Reweighting ordering produces the second-largest change, in the opposite direction, consistent with its large RDX magnitude in Figure 7, whereas reweighting holding has little effect on total reward although holding terms make up a large share of the MSX set in Figure 7. The magnitude of RDX/MSX values and the sensitivity of total reward to a weight thus capture different aspects of the same decomposition.

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 9. One-at-a-time sensitivity of average reward to the four weighting coefficients for the two learning agents and the base-stock policy.**

### *Joint (multi-weight) sensitivity and interaction effects*

OAT does not capture interactions, so all six pairs of the four components were perturbed jointly over {0.5, 1.0, 1.5, 2.0}² (16 combinations per pair; same policies, trajectories, and MSX threshold; agents and scenarios pooled). The holding × order pair shows a pronounced joint effect: holding barely affects total reward alone and order has a large RDX magnitude, yet jointly the MSX-inclusion rate ranges from 0.000 to 0.988 and falls below 1% whenever the order weight is 2.0, a pattern OAT cannot reveal. Pairs involving service give the widest total-reward ranges (up to about 1.94 against a unit-weight baseline of 1.0), reflecting a service term close to +1.0. Joint weight changes can thus alter RDX/MSX explanations in ways not visible when components are varied separately.

## SHapley Additive exPlanations

To address aggregation bias while maintaining strategic interpretability, this study adopts a two-level SHAP framework. 

The aggregated 3-feature view provides a strategic cognitive framework for quickly identifying overall behavioral trends, while the micro-level analysis on the 660-dimensional raw feature space addresses heterogeneity. This structure maintains compatibility with prior FCS/ablation analyses while the micro-level results are treated as the primary SHAP findings.

SHAP for DQN is computed on softmax(Q) and for A2C_mod on the policy π(a*|s), two quantities on different scales (Q unbounded; π bounded in [0,1] and interdependent via the softmax), so magnitude and stability comparisons between agents in Figures 10–12 are descriptive and qualitative, not quantitative. A sensitivity analysis on common-target logits (Section 3.6.6; Supplementary Material) supports this: Jaccard similarities are only 0.25–0.29 (DQN) and 0.25–0.67 (A2C_mod) across evaluation-set sizes.

### *Problem setup*

       The original state space consists of 664 features (660 product-level features and 4 system-level features). However, applying SHAP directly to such a high-dimensional space with strong correlations among variables may result in dispersed and difficult-to-interpret explanation values. Therefore, the study adopts a purposive dimensionality reduction strategy, aggregating the 660 product-level features into three representative portfolio-level variables: Inventory (I), Demand (D), and Waste (W). This aggregation aligns with the reward function structure and helps improve the signal-to-noise ratio, while providing clearer managerial implications at the system level However, this approach may reduce the ability to capture heterogeneity across SKUs and may overlook cross-product interactions. Therefore, the explanation results should be interpreted at the system level, acknowledging the trade-off between interpretability and granularity.

### *Experimental Results and Analysis*

For A2C_mod (Figure 9), the SHAP values of the three features Inventory, Demand, and Waste are all concentrated very close to zero with low dispersion. This indicates that the contribution of each feature to the model output is small and relatively balanced, reflecting a decision-making mechanism that is not strongly dominated by any single factor at the global level. In contrast, DQN exhibits a more pronounced separation in the SHAP distributions, particularly for Inventory and Demand, with larger dispersion and a clear influence on output. Meanwhile, Waste shows a very small contribution concentrated around zero. These results indicate that DQN’s global decision making is primarily driven by inventory and demand. 

Comparing the two models, the global SHAP analysis shows that A2C_mod has a more distributed and stable contribution structure, while DQN relies heavily on a few dominant features, highlighting differences in the overall decision-making mechanisms between the two approaches. Because the two explained outputs differ in scale (softmax(Q) versus π(a*|s)), the smaller SHAP magnitudes of A2C_mod partly reflect its bounded probability output, so this comparison is qualitative (Section 3.6.6).

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 10. Global SHAP of A2C_mod and DQN.**

To analyze decision-making behavior at the local level, the study constructs five representative scenarios corresponding to key operational states of the inventory system, as shown in Table 13.

**Table 13. Scenarios for local SHAP analysis.**

**Scenario**

**System State**

**Evaluation Purpose**

High Inventory

Inv = 0.9,     Demand = 0.5

Tests the ability to “stop ordering” when inventory is high.

Low Inventory

Inv = 0.1, 

Demand = 0.5

Tests the response to “urgent replenishment” when inventory is low.

High Demand 

Inv = 0.5, Demand = 0.9

Tests sensitive to market  opportunity.

Low Demand

Inv = 0.5, 

Demand = 0.1

Tests the ability to reduce costs when demand is weak

Critical State

Inv = 0.05, Demand = 0.8

Extreme situation: inventory is nearly depleted while demand is high.

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 11. Local SHAP of A2C_mod and DQN.**

The local SHAP results (Figure 11) reveal clear differences in state-dependent responses between A2C_mod and DQN. Local SHAP analysis shows distinct differences in how A2C_mod and DQN react to system states. In the High Inventory and Low Inventory scenarios, DQN is strongly influenced by the Waste component, while A2C_mod exhibits very small SHAP values close to zero, indicating that its decisions are not dominated by any single factor. In the High Demand and Low Demand scenarios, DQN continues to show pronounced sensitivity to Demand and Waste, reflecting a decision-making behavior that reacts to market fluctuations. In contrast, A2C_mod maintains low and dispersed SHAP contributions, indicating a more stable decision-making mechanism. Notably, in the Critical State scenario, DQN exhibits SHAP values with large magnitudes, while A2C_mod still maintains values close to zero, demonstrating its ability to sustain stable decisions even under extreme conditions. Overall, the local SHAP analysis suggests that, for the explained outputs and in this setting, A2C_mod is more state stable, whereas DQN relies heavily on certain features, especially in high-risk scenarios; given the different output scales, this comparison is qualitative.

### *Top-k Micro-Level SHAP Analysis*

To complement the aggregated SHAP analysis, this study further performs a top-k micro-level SHAP analysis on the 660-dimensional product-level state space, consisting of 220 products and three feature groups: Inventory, Sales/Demand, and Waste. While the aggregated SHAP analysis provides a macro-level interpretation, the micro-level analysis identifies which specific product-level variables most strongly influence the agents’ decisions. Because the raw state space is high-dimensional, the SHAP Partition Explainer is used to improve computational efficiency by exploiting the hierarchical structure of the input features. The importance of each feature is computed using the mean absolute SHAP value: 

where denotes the SHAP value of feature for action at state , is the number of evaluated states, and is the action set. After ranking all features by their importance scores, the top-20 most influential variables are extracted for analysis. As shown in Figure 12, Sales/Demand features dominate the top-ranked positions in most scenarios, indicating that both agents are strongly influenced by product-level demand fluctuations. In the EASY scenario, the decisions of DQN and A2C_mod are mainly driven by Sales/Demand features, while Waste-related variables have limited influence. In the MEDIUM and HARD scenarios, Waste-related features appear more frequently among the top-20 variables, suggesting that the agents begin to consider spoilage risk when the environment becomes more uncertain and costly. For DQN, Sales/Demand remains the dominant factor across scenarios, but Waste features become more visible in the HARD scenario. For A2C_mod, a similar pattern is observed; however, in the HARD scenario, Waste-related variables receive higher importance, suggesting a more risk-aware decision pattern. Overall, the top-k micro-level SHAP analysis complements the aggregated Inventory–Demand–Waste explanation by showing that the agents’ decisions are also influenced by specific product-level variables. These SHAP results should be interpreted as associative feature attributions rather than causal effects.

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 12. Top 20 SHAP feature importance over the 660-dimensional product-level state space for DQN and A2C_mod across EASY, MEDIUM, and HARD scenarios.**

While Section 3.6.2 provides an aggregated overview, this section presents the primary SHAP analysis on the 660-dimensional product-level state space (50 states per scenario, PartitionExplainer, background 100). At the macro level, Sales/Demand dominates for both agents, but the micro level reveals extreme concentration on 5-8 specific SKUs rather than uniform importance across 220 SKUs, evidencing aggregation bias; SKUs 64, 163, 46, and 155 appear in the Top-5 of every agent-scenario case.

### *Consistency of Top-k Features across EASY/MEDIUM/HARD*

For Top-20, DQN shows Jaccard 1.00 and Spearman 1.00 across all scenario pairs (stable, 8 SKUs), whereas A2C_mod shows Jaccard 0.29–0.33 and Spearman 0.20 (context-dependent shift, core Top-5 remains). The DQN stability must be read together with its sensitivity to the background distribution and the number of explained states (Sections 2.3.4.2 and 3.6.6), where rankings changed markedly, so the identity of the dominant SKUs is indicative only; full k = 10/20/50 results are in the Supplementary Material, and the comparison is qualitative (Section 2.3.3).

### *Why Dominant Products Dominate Top-20 - Case Analysis*

The top-demand SKUs (57, 81, 108) are not in the Top-8 SHAP (r = 0.04). The dominant SKUs are low-volume, high-CV (1.1-1.46), small-capacity items (e.g., SKU 64: CV 1.46, capacity 7; SKU 163: CV 1.27, capacity 12), which are volatile and stockout-sensitive; SKU 215 (volume 8.68, capacity 81) is an exception.

### *Common Target Sensitivity Analysis*

For a common target, pre-softmax logits were used (linear Q-value outputs for DQN, penultimate-layer outputs for A2C_mod), explained with PartitionExplainer over 100 background samples on n = 10, 20, and 50 states (n = 10 as a ~25-minute feasibility benchmark, to be validated by a Jaccard index above 0.8 against n = 50). Rankings proved highly sensitive to sample size: the Jaccard similarity between n = 10 and larger sets is only 0.250–0.290 for DQN and 0.250–0.667 for A2C_mod, so a 10-state subset cannot reliably capture global importance and attributions are reported across all three regimes. Logit attributions also differ from post-activation targets (Jaccard 0.667 for DQN and 0.250 for A2C_mod in EASY), which supports treating the cross-architecture comparison as qualitative.

### *Actor vs Critic Sensitivity*

For A2C_mod, explanations can target the policy output π, the pre-softmax logits, or the state value V(s). On a 10-state set per scenario, attributions varied markedly with the target despite the shared representation (rankings in the Supplementary Material): in EASY, π versus logits overlap in only 2 of 5 top features (Top-20 Jaccard 0.250), and π versus V(s) gives 0.250. Actor and critic thus prioritize different signals, so interpretability studies of multi-head architectures should distinguish policy-level from value-level targets rather than assume invariance.

### *Reward Component Range and Distribution*

Reward components were computed per product and period from the normalized state (demand divided by capacity, a multiple of average training demand) on the original records (1,000 training and 504 testing periods, 220 products), with the same transition and reward as in training. Normalization reduces mean and dispersion by more than an order of magnitude (mean demand from 2.11 to 0.10 in training and from 0.73 to 0.034 in testing), while capacity is heterogeneous but about ten times average demand.

Three points follow. First, capacity-based normalization is effective: the distributional distance between training and testing falls from 1.37 for raw demand to 0.066 for normalized demand (about 95%), mitigating covariate shift without refitting on test data and hence without leakage. Second, waste lies in [0, 0.025] and balance spread in [0, 0.89], but the service component is close to 1 per product while cost components are about an order of magnitude smaller (Table 6), so service dominates total-reward changes under reweighting (Section 3.5) and unit weights are a neutral reference, not a calibrated choice. Third, the contrast between policies supports the reward design: without replenishment stockouts are almost certain (0.97 on the test set) and no inventory is retained, whereas base-stock replenishment cuts the stockout rate to 0.022 and reaches a stable, high-reward operating point; overstock is policy-dependent (it appears for the aggressive value-based agent) rather than data-dependent.

## Comparative Evaluation

### *Design*

To evaluate the role and contribution level of each explanation mechanism within the proposed XRL framework, the study conducts an ablation-oriented analysis by comparing three different explanation configurations. The objective of this design is to determine whether each individual method is sufficient to provide comprehensive interpretability, or whether their combination is necessary to achieve holistic transparency. Specifically, the three configurations considered are:

(i) **Reward-only (RDX + MSX):** Explanations are constructed entirely based on reward decomposition. RDX quantifies the contribution of each business objective, while MSX identifies the minimal set of components sufficient to justify the action.

(ii) **Feature-based only (SHAP):** Explanations rely on SHAP values to measure the influence of state features on the Q-value or action policy, without considering the reward objective structure.

(iii) **Hybrid framework:** Combines RDX/MSX and SHAP to simultaneously analyze explanations at both the business-objective level and the input-feature-level

### *Metrics*

      To quantitatively evaluate the explanation mechanisms within the proposed XRL framework, the study employs a set of metrics to measure objective coverage, feature utilization, cross-domain consistency, and the stability of the explanation structure under weight adjustments.

##        Objective Coverage Score (OCS): OCS measures the coverage level of business objectives in reward-based explanations (RDX/MSX). 

## Where:

## •   is the number of business objectives (Service, Holding, Waste, Order),

## •   is the contribution of objective ,

## •   is the threshold for significant contribution 

## •    is the indicator function.

## Feature Coverage Score (FCS): FCS measures the proportion of state features that significantly influence the Q-value in SHAP-based explanations.

Where:

**•  ** is the number of input features,

**•  ** is the SHAP value of feature .

## Cross-domain Alignment Score (CAS): CAS measures the consistency between the business objective domain (RDX/MSX) and the state feature domain (SHAP).

Where:

**•  **The set of dominant features includes features whose SHAP values exceed the threshold,

**•  **The set of dominant objectives includes objectives with high RDX contributions,

**•  ** is the number of important factors considered.

Stability evaluates the robustness of the MSX structure when the weight coefficient λ changes.

This metric reflects the proportion of states in which the MSX set remains unchanged when the weights are adjusted within a reasonable range.

Dominance Ratio (DR): DR measures the relative contribution of each reward component.

## %MSX-change

%MSX-change=100%−Stability

## This metric reflects the degree of change in the MSX structure when the weights are adjusted.

### *RDX/MSX-only*

The results in Figure 13 reveal the existence of a structural trade-off mechanism between the Objective Coverage Score (OCS) and the size of the minimal explanation set (MSX-size) as the parameter λ varies from 0.5 to 2.0. Specifically, as λ increases, the threshold for determining a “significant” contribution in OCS becomes more stringent, leading to a tendency for the number of objectives exceeding the threshold to decrease or remain low. At the same time, the MSX mechanism requires a higher level of Q-gap coverage, forcing the algorithm to expand the objective set to ensure complete explanation. Consequently, MSX-size increases with λ, while OCS does not increase correspondingly, reflecting an inverse relationship between the two metrics

![placeholder](https://markdowntoword.io/placeholder.png)

## Fig. 13. Dependency relationship with respect to  between OCS and MSX-size for DQN and A2C_mod.

The difference between the two architectures is clearly reflected in how they respond to the λ constraint. For DQN, both OCS and MSX-size vary in a continuous and gradual manner, indicating that Q-values are relatively evenly distributed across multiple objectives. This structure reflects a stable multi-objective decision-making mechanism, in which multiple objectives jointly contribute to shaping the action value. In contrast, A2C_mod exhibits more pronounced nonlinear behavior. At low λ levels, MSX-size remains small, implying that decisions are primarily dominated by a few prominent objectives. However, when λ exceeds a moderate threshold, MSX-size abruptly increases to its maximum level, indicating that the model must incorporate additional objectives to satisfy the Q-gap coverage requirement. This suggests that the value structure of A2C_mod is more concentrated and more sensitive to explanation constraints. These findings indicate that the RDX/MSX mechanism not only reflects the number of components involved in an explanation but also reveals the intrinsic value distribution structure of each architecture. DQN demonstrates greater stability and a more distributed multi-objective allocation, whereas A2C_mod tends to concentrate strongly on dominant objectives and only expands the explanation set when constraints become stricter.

Figure 14 illustrates the variation of the Stability metric with respect to the constraint coefficient λ across the three scenarios EASY, MEDIUM, and HARD. Stability measures the extent to which the MSX set structure is preserved when λ changes relative to the reference configuration λ = 1.0, thereby reflecting the intrinsic consistency of the explanation mechanism under hyperparameter perturbations.

The results show that at λ = 1.0, both DQN and A2C_mod achieve Stability close to 100%, confirming that the MSX set under the baseline configuration is internally consistent. However, when λ increases beyond 1.0, the two architectures exhibit clearly divergent behaviors. For DQN, Stability decreases in a continuous and gradual manner. When λ increases to 1.5, Stability drops to approximately 52–53%, and further declines to around 44–46% at λ = 2.0 across all three scenarios. This relatively smooth downward trend indicates that the MSX structure is adjusted in an incremental and inheritable manner: dominant objectives are retained, while additional objectives are incorporated to satisfy the higher coverage threshold. In contrast, A2C_mod demonstrates an abrupt decline. Once λ exceeds 1.0, Stability sharply drops to approximately 27–29% and remains at this low level as λ continues to increase. This pattern is consistently observed across all three environmental scenarios, indicating that changes in the MSX structure occur in a stepwise rather than gradual manner. Notably, the shape of the Stability curves is nearly identical across EASY, MEDIUM, and HARD for each model. This suggests that the decline in Stability is primarily driven by the XRL constraint (λ), rather than environmental complexity. In other words, the stability of the explanation set reflects the architectural characteristics of value learning more than the nature of the task itself. Overall, DQN maintains significantly higher Stability than A2C_mod as λ increases, with nearly double the value when λ ≥ 1.5. These results indicate that the value-based architecture of DQN produces explanation sets that are more inheritable and robust under strict explanation constraints, whereas A2C_mod is more sensitive to variations in the XRL hyperparameter.

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 14. Robustness of the MSX set with respect to the threshold coefficient λ across the EASY, MEDIUM, and HARD scenarios for DQN and A2C_mod**

### *SHAP-only*

To evaluate the effectiveness of the SHAP-only configuration, the study establishes experiments under multiple different settings. Table 14 summarizes the complete experimental setup used in this analysis. The FCS results corresponding to the 24 experimental configurations are summarized in Table 15, showing the variation in feature coverage with respect to λ and across different agent architectures.

Figure 15 presents the variation of the Feature Coverage Score (FCS) with respect to the parameter λ for the two architectures, DQN and A2C_mod, across three environmental scenarios (EASY, MEDIUM, HARD). The results are computed over 496 test states and averaged over 5 repeated evaluations of one trained checkpoint per configuration (no retraining in Table 14) . Overall, both agents maintain very high FCS values (>0.90) across all configurations, indicating that the three input features (inventory, sales, and waste) make significant contributions to the decision-making process. As λ increases (corresponding to a higher threshold ε according to the relation ε = 0.01λ), FCS decreases monotonically for both models, reflecting increasingly stringent criteria for identifying “significant” features. However, the magnitude of decline is greater for A2C_mod than for DQN, resulting in a widening gap between the two models as λ increases.

This difference indicates that the SHAP value distribution of A2C_mod tends to be more concentrated around the threshold ε, making features more likely to be excluded as λ increases, whereas DQN maintains contributions that consistently exceed the threshold. At the same time, DQN exhibits almost perfect reproducibility (standard deviation approximately 0), while A2C_mod shows small variance (<0.3%), confirming the high stability of both models but with different levels of sensitivity to threshold changes. Notably, FCS in the HARD scenario is not lower than in MEDIUM; in some configurations, it is even comparable or higher, indicating that increased environmental difficulty does not reduce feature coverage. Overall, the results confirm that both agents rely on the entire feature space; however, DQN is more robust to variations in the parameter λ, and λ = 1.0 can be considered a reasonable balance between sensitivity and specificity in SHAP-based explanation analysis.

**  Table 14. Experimental Parameter Configuration Design**

**Component**

**Setting**

Agents

DQN, A2C_mod

Scenarios

EASY, MEDIUM, HARD

Number of λ levels

0.5, 1.0, 1.5, 2.0

Threshold ε (FCS)

0.5%, 1.0%, 1.5%, 2.0%

Total configurations

2 agents × 3 scenarios × 4 λ = 24 configurations

Number of runs

5 repeated evaluations of one trained checkpoint per configuration (no retraining)

**Table 15. Comparison of FCS (Mean ± Std) across λ and ε levels under different environmental scenarios.**

**Agent**

**Scenario**

**λ**

**ε**

**FCS (Mean ± Std)**

**DQN**

EASY

0.5

0.005

0.9906 ± 0.0000

1.0

0.010

0.9825 ± 0.0000

1.5

0.015

0.9731 ± 0.0000

2.0

0.020

0.9610 ± 0.0000

MEDIUM

0.5

0.005

0.9825 ± 0.0000

1.0

0.010

0.9684 ± 0.0000

1.5

0.015

0.9563 ± 0.0000

2.0

0.020

0.9483 ± 0.0000

HARD

0.5

0.005

0.9872 ± 0.0000

1.0

0.010

0.9792 ± 0.0000

1.5

0.015

0.9731 ± 0.0000

2.0

0.020

0.9704 ± 0.0000

**A2C_mod**

EASY

0.5

0.005

0.9805 ± 0.0004

1.0

0.010

0.9636 ± 0.0010

1.5

0.015

0.9366 ± 0.0007

2.0

0.020

0.9199 ± 0.0008

MEDIUM

0.5

0.005

0.9777 ± 0.0017

1.0

0.010

0.9598 ± 0.0003

1.5

0.015

0.9429 ± 0.0010

2.0

0.020

0.9223 ± 0.0017

HARD

0.5

0.005

0.9886 ± 0.0013

1.0

0.010

0.9753 ± 0.0005

1.5

0.015

0.9647 ± 0.0007

2.0

0.020

0.9185 ± 0.0014

	

The results in Figure 16, showing the distribution of the Feature Coverage Score (FCS), evaluate the statistical stability of the SHAP-only method under the influence of agent architecture, environmental difficulty, and classification threshold. The results indicate that DQN achieves nearly perfect reproducibility across all 12 configurations, with boxplots collapsing into thin horizontal lines and no observable dispersion across runs. This suggests that FCS is not affected by randomness in the SHAP estimation process and reflects the high determinism of the trained Q-network. In contrast, A2C_mod exhibits small but finite dispersion, reflected in narrow interquartile ranges and very low variance. However, this level of variation remains negligible compared to the mean FCS values (above 0.91 in all configurations), indicating that the explanation method remains statistically stable. Notably, no abrupt increase in variance is observed when transitioning from the EASY to HARD scenario for either agent, suggesting that environmental difficulty does not compromise measurement reproducibility. The influence of the parameter λ is also clearly observed. While DQN maintains invariant stability across λ values, A2C_mod shows a slight increase in dispersion as λ increases, particularly at λ = 2.0. This trend may be explained by the presence of features with SHAP values close to the classification threshold, leading to minor fluctuations in FCS across runs. However, the magnitude of variation remains very limited and does not alter the overall trend of the results. 

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 15. Effect of the parameter λ on the FCS of DQN and A2C_mod across the three environmental scenarios.**

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 16. Distribution of the Feature Coverage Score (FCS) of DQN and A2C_mod across different λ levels in the three environmental scenarios.**

### *Combined frame*

The results in Figure 17 provide a comprehensive view of the agent’s decision-making structure at both the objective level and the feature-level. The OCS results indicate that environmental difficulty directly affects the number of objectives dominating the policy. In the EASY scenario, both agents maintain a stable single-objective structure (OCS = 0.25) across the entire λ range, reflecting a focused strategy with no significant competition among reward components. In contrast, in the MEDIUM and HARD scenarios, particularly at low λ values, a multi-objective structure (OCS = 0.50) emerges, indicating the presence of trade-offs among objectives as the level of competition increases. However, when compared with CAS, most configurations still yield a value of 0, implying that although the agent may simultaneously optimize multiple objectives, direct alignment between feature-level explanations (SHAP) and objective-level explanations (RDX) does not always exist. Alignment becomes more evident in the HARD environment and at sensitive λ thresholds, especially for A2C_mod, suggesting that under high-pressure conditions, the linkage structure between features and objectives becomes more consistent. A comparison between the two architectures shows that A2C_mod maintains both a multi-objective structure and more stable alignment as λ increases, whereas DQN is more sensitive to threshold changes and tends to lose its multi-objective structure at high λ values. Overall, these findings emphasize that multi-objective optimization does not necessarily imply direct compatibility between the two explanation layers. Therefore, jointly analyzing OCS and CAS is necessary to uncover potential decoupling between feature-level reasoning and objective-level reasoning in multi-objective reinforcement learning systems.

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 17. Sensitivity of Objective Coverage Score (OCS) to Threshold Parameter λ across Scenarios for DQN and A2C_mod**

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 18. Cross-domain Alignment Score (CAS) across Threshold Levels (λ), Agents, and Environmental Scenarios**.

Figure 18 presents the Cross-domain Alignment Score (CAS) between feature-level explanations (SHAP) and objective-level explanations (RDX) across different threshold values λ in the three scenarios EASY, MEDIUM, and HARD for the two agents DQN and A2C_mod. The results show that most configurations yield CAS = 0, reflecting the lack of direct compatibility between the two explanation layers. This implies that the objectives activated at the RDX level are not necessarily directly mapped from features with high SHAP contributions. In the EASY scenario, CAS remains 0 across the entire λ range for both agents, indicating that although the policy may optimize a dominant objective, the feature-level structure does not exhibit clear overlap with objective-level reasoning. In the MEDIUM scenario, alignment appears only at low λ values: DQN achieves CAS = 0.25 at λ = 0.5, while A2C_mod maintains CAS = 0.25 at λ = 0.5 and 1.0 before dropping to 0 as λ increases. This suggests that when the evaluation threshold is sufficiently sensitive, overlap between the two explanation sets exists; however, as λ becomes more stringent, alignment declines because only the strongest contributing components are retained. Notably, in the HARD scenario, A2C_mod maintains CAS = 0.25 across the entire λ range, whereas DQN achieves alignment only at λ ≤ 1.0 and loses alignment at higher values. These results indicate that under high-pressure environments, the Actor–Critic architecture demonstrates a more robust ability to maintain linkage between feature-level and objective-level reasoning. Overall, the figure confirms that compatibility between the two explanation layers is not an inherent property of the learned policy, but rather depends on environmental complexity, agent architecture, and the strictness level of the λ threshold.

## Faithfulness Evaluation of Explanations

To complement the descriptive comparison metrics in Section 3.7, this study further evaluates the faithfulness of the generated explanations through perturbation-based tests. The goal is to examine whether the reward components or state features identified as important by RDX/MSX and SHAP are functionally related to the agents’ outputs. If masking explanation-identified components or features causes larger changes than masking random or less relevant ones, the explanation can be considered more faithful to the model behavior.

### *Faithfulness Evaluation for RDX/MSX*

For RDX/MSX, the evaluation is conducted by masking the reward components selected by MSX and observing whether the agent’s selected action or value estimate changes. Given the decomposed value function:

the MSX-identified component set is removed, and the remaining value is recomputed. The action flip rate of MSX masking is then compared with a random masking baseline , which masks the same number of reward components randomly. The faithfulness ratio is defined as: 

where indicates that MSX-selected components are more influential than randomly selected components. 

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 19. Action flip rate comparison between MSX-guided masking and random masking across EASY, MEDIUM, and HARD scenarios under different thresholds.**

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 20. RDX/MSX faithfulness analysis based on faithfulness ratio and Q-value gap drop under MSX-guided masking and random masking.**

The results show that MSX-guided masking becomes more faithful when the threshold is sufficiently strict. In particular, when , the MEDIUM and HARD scenarios show substantially higher action flip rates under MSX-guided masking than under random masking. In the HARD scenario, masking MSX-identified reward components changes the selected action in nearly all evaluated states, whereas random masking produces a much lower action flip rate. This indicates that MSX identifies reward components that are functionally important to the agent’s decision.

In contrast, the EASY scenario shows almost no action changes under either MSX-guided or random masking. This is expected because the reward components are less conflicting in simple environments, and many actions have similar reward values. Overall, the RDX/MSX faithfulness results suggest that the explanation is more informative in scenarios with stronger reward conflicts, especially in MEDIUM and HARD settings.

A one-sided paired test on the MSX-minus-Random flip rate difference confirms this pattern quantitatively: The faithfulness advantage of MSX-guided masking is statistically significant only once the masking threshold λ is strict enough (λ≥1.5: p<0.001 in both MEDIUM and HARD,  λ≤1.0: not significant, p=1.0),  EASY produces zero flips under either strategy at every λ, so no test could be computed there. 

### *Faithfulness Evaluation for SHAP*

For SHAP, faithfulness is evaluated by perturbing the top-ranked features identified by SHAP and measuring the resulting changes in the model outputs. Three masking strategies are compared: MoRF, Random, and LeRF. MoRF masks the most relevant SHAP features first, LeRF masks the least relevant features first, and Random serves as a baseline. For DQN, the output change is measured by the relative decrease in the selected-action Q-value:

     For A2C_mod, the output change is measured by the decrease in the selected-action probability:

     The action switching rate is also used to measure how often the final selected action changes after feature masking. Figure 21 presents the perturbation curves for SHAP faithfulness verification under MoRF, Random, and LeRF masking strategies. 

![placeholder](https://markdowntoword.io/placeholder.png)

**Fig. 21. Perturbation curves for SHAP faithfulness verification under MoRF, Random, and LeRF masking.**

The results generally follow the expected faithfulness pattern:

    For DQN, masking the most relevant SHAP features produces larger output changes than masking random or least relevant features. This effect becomes clearer in the MEDIUM and HARD scenarios, where the selected action is more likely to change after MoRF masking. This suggests that the top-ranked SHAP features are functionally related to DQN’s Q-value estimation and action selection. For A2C_mod, the selected action remains more stable, with limited action switching across scenarios. However, the selected-action probability still decreases more strongly under MoRF than under Random or LeRF masking. This indicates that SHAP-identified features affect the confidence of the policy, even when the final selected action does not change. Overall, the perturbation-based results provide empirical evidence that the proposed explanations are not merely descriptive. RDX/MSX identifies reward components whose removal can change the selected action, while SHAP identifies state features whose perturbation reduces Q-values or policy probabilities. Therefore, the proposed XRL framework is supported by faithfulness evidence, particularly under more difficult inventory scenarios.

### *Perturbation Protocol*

The protocol isolates the effect of the importance ranking on the model output while preserving the rest of the state distribution. Fifty test states per scenario (150 per agent) are drawn from the normalized test sequence, with thirty repetitions per random condition. Features are ranked by mean absolute SHAP value from the micro-level analysis and three strategies are compared on identical states: Most Relevant First, Least Relevant First, and random removal of the same number of features as a calibrated baseline. One to ten features (about 0.15–1.52% of the 660-dimensional state) are replaced by the median of 100 background states, with a fixed seed. The output change is the relative decrease of the Q-value of the originally optimal action (value-based agent) or the absolute decrease of its policy probability (actor-critic agent); the switching rate is the share of cases in which the optimal action changes.

### *SHAP Faithfulness: Area Under Perturbation Curve and Confidence Intervals*

The area under the perturbation curve (trapezoidal integration over the ten masking levels) summarizes the cumulative effect, with 95% bootstrap intervals (1,000 resamples over states) and Wilson intervals for switching rates. The expected faithfulness ordering holds in all conditions (Table 16): Most Relevant First gives a positive, monotonically increasing change, random stays near zero, and Least Relevant First is negative or near zero, with the actor-critic agent showing changes about twice as large as the value-based agent (indicative only, since the scales differ). Intervals do not overlap between Most Relevant First and random, and the switching rate is zero in all eighteen conditions (Wilson upper bound about 7%), so the discrete decision is robust within the examined range.

**Table 16. Area under perturbation curve by strategy (95% interval width <0.002).**

Agent

Scenario

Most Relevant First

Random

Least Relevant First

Value-based

Easy

0.014

-0.001

-0.010

Value-based

Medium

0.011

-0.001

-0.010

Value-based

Hard

0.010

-0.001

-0.011

Actor-critic

Easy

0.024

0.007

-0.020

Actor-critic

Medium

0.024

0.006

-0.0003

Actor-critic

Hard

0.025

0.005

0.0002

### *Statistical Comparisons: Ordered Masking and Minimal Sufficient Guidance*

Superiority of the ordered conditions was tested per state with a paired non-parametric test, effect sizes, and correction for multiple comparisons. All comparisons of Most Relevant First against random and against Least Relevant First remain significant after correction, with large to very large effects (the only moderate effect is random versus Least Relevant First for the actor-critic agent in the hard scenario). Guidance by a minimal sufficient set also outperforms random masking of equal size, linking feature-level and reward-level faithfulness: with size five, both agents and all scenarios give corrected p < 0.001 and large effects (about 2.8–3.9), and with size two the advantage is significant but smaller (details in the Supplementary Material).

### *Threshold for Meaningful Change*

A meaningful change is defined relative to the granularity of the action space (the smallest step between consecutive levels is about 0.25% of storage capacity) and the natural noise of the system: a one-percent relative change in Q-value or absolute change in policy probability, or any change in the optimal action index. These are conventional sensitivity thresholds, not values calibrated to an operating cost. No state exceeds the one-percent threshold in any of the eighteen conditions (average change at maximum masking 0.16–0.40%), and the discrete threshold gives the same result; this does not indicate unfaithful rankings, since the ordering between strategies remains highly significant, but that a lower threshold or larger masking would be needed, so the threshold is reported as a sensitivity parameter.

### *Consistency Across States*

Consistency was assessed from the distribution of the change across the fifty states of each condition (mean, SD, coefficient of variation, IQR at maximum masking). The actor-critic agent shows low dispersion (CV 0.02–0.17), whereas the value-based agent becomes more heterogeneous with difficulty (CV 0.19 in EASY to 0.48 in HARD). Scenario differences are significant for both agents, so faithfulness is not entirely invariant to operating conditions, but the ordering of the three strategies is preserved in every scenario. Together, these analyses show that the feature ranking is functionally faithful in ordering, that its advantage over random and reverse baselines is statistically robust, and that minimal sufficient guidance adds complementary evidence, within the examined masking range (at most about 1.5% of the state features).

## Sensitivity to Action Space Resolution

To assess whether the conclusions depend on the granularity of the replenishment decision, three resolutions were compared: coarse (7 levels), baseline (14), and fine (28). Dynamics, reward, and state representation were fixed, each agent family was trained independently under each resolution with its tuned hidden size, and evaluation was deterministic (greedy policy, fixed seed) on the held-out test sequence. Because training stochasticity is not averaged over seeds, differences are indicative rather than conclusive, and the weaker A2C_mod results at 7 and 28 levels may partly reflect seed-specific failures such as the seed-specific collapse to maximum replenishment reported in Section 3.3.4. Explanation quality is measured by SHAP feature coverage and, from a one-step lookahead, by objective coverage, minimal sufficient set size, and its stability.

Performance is non-monotonic in resolution and differs by family (Table 17). The value-based agent keeps a high reward at all resolutions, whereas the actor-critic agent peaks at the baseline resolution and degrades at both coarser and finer discretizations because of a sharp increase in overstock (0.96 and 0.84 versus 0.00), which is operationally meaningful rather than a reward-scaling artifact.

**Table 17. Operational performance on the held-out test set across resolutions.**

Agent

Resolution

Reward

Overstock

Waste

Actor-critic

7

-0.13

0.96

0.024

Actor-critic

14

0.38

0.00

0.013

Actor-critic

28

-0.12

0.84

0.024

Value-based

7

0.80

0.00

0.024

Value-based

14

0.60

0.00

0.019

Value-based

28

0.82

0.01

0.024

Explanation robustness does not follow the performance trend. Feature coverage stays sparse in all conditions (at most 0.02); the fine resolution for the actor-critic agent needs more objectives to reach the same threshold (minimal set size 2.68 versus about 1.0), and stability is highest for the coarse actor-critic agent (90.7) and within a moderate band (about 52–60) for the value-based agent.

Overall, effectiveness and explanatory characteristics respond differently to resolution: fourteen levels is a balanced choice for the actor-critic family, whereas the value-based family tolerates coarser and finer discretizations. Sparsity and moderate stability of explanations are largely preserved, supporting the generalizability of the earlier conclusions, while the larger minimal set at very fine granularity marks a practical limit for the actor-critic agent.

## Scope and Interpretation of the Study's Claims

The results in Sections 3.1–3.9 address four distinct aspects of the evaluated policies. Operational performance uses the five-seed aggregates in Tables 6 and 7; policy adaptability uses the behavior across the controlled stress-test scenarios in Figures 2 and 6 and Table 8 (one selected checkpoint per architecture); explanation stability uses the Stability/Jaccard metric (Figure 14) and the FCS dispersion (Table 15); explanation faithfulness is assessed separately in Section 3.8 (Figures 19–21). These aspects should be interpreted independently: strong operational performance does not by itself establish explanation stability or faithfulness, and vice versa.

**Boundary conditions.** The four claim types are evaluated under a common setting (220 Instacart-derived products, a shared 14-level action space, a unit-weighted reward), so the findings characterize the specific DQN and A2C_mod implementations used here, not DQN or A2C as algorithmic classes.

## Scalability to Product-Group Level

Scalability was evaluated by training A2C_mod and DQN independently on the three groups under identical settings (600 episodes of 900 steps, 14 actions); all six models completed training. DQN outperformed A2C_mod in every group: the mean reward over the last 100 episodes was 0.7709, 0.6508, and 0.5740 for the Fast, Medium, and Slow groups versus 0.1807, 0.0477, and 0.1514 for A2C_mod, with DQN stockout rates near zero and slightly higher waste. A2C_mod improved by only 1.48% from the first 50 episodes in the Fast group and declined by 79% in the Medium group, which therefore needs separate tuning. The pipeline is thus not hard-coded to 220 SKUs and can be applied at product-group granularity; applying RDX, MSX, and SHAP per group is left for future work.

## Managerial Case Studies and Interpretation

The analyses in Sections 3.3–3.8 and the scorecard in Table 18 (Section 3.13) establish properties of the explanations with respect to the agents. Whether they are stable when analysis choices change and whether the reward components and features they highlight are functionally related to the agents’ outputs. Faithfulness to the model does not establish usefulness to practitioners. That would require evidence that the explanations help inventory managers make better or faster decisions, or rely on the agent appropriately, and no such study with domain experts was conducted. The three cases below therefore illustrate how the explanations can be read in operational terms. They are the authors’ interpretations, offered to make the results concrete, not evidence of practical usefulness, and the claims of this study are accordingly limited to transparency of the agents’ decisions.

Faithfulness to model behavior does not entail usefulness to practitioners, and domain-expert evaluation was not feasible within the present scope. Three illustrative managerial cases therefore translate the findings into operational language, covering reward-space trade-offs, minimal justification sets, and feature-space attribution; as an interim bridge between statistical faithfulness and practical usefulness, they moderate claims regarding trust to transparency, with usefulness deferred to future expert assessment.

Case 1: Balancing service fulfillment against holding costs

Under stable conditions with ample capacity and low perishability, both architectures converge on aggressive replenishment and service dominates the reward difference. Under pressure (high inventory, elevated demand, higher perishability), the actor-critic agent attenuates ordering intensity and holding-cost exposure whereas the value-based agent keeps prioritizing service; the explanation makes the compromise explicit (cost containment versus stockout avoidance) without prescribing a universally correct choice.

Case 2: Concise justification for order decisions

Minimal sufficient sets give a concise rationale that can be checked against the reward design: the actor-critic agent needs only the service component under low and moderate pressure and shifts to holding under high pressure, whereas the value-based agent stays centered on holding in all regimes.

Case 3: Identifying products that drive decisions

Aggregated views suggest uniform importance across the portfolio, whereas the disaggregated analysis shows strong concentration on a few volatile, small-capacity SKUs rather than on the highest-volume items (Section 3.6.5), which suggests where monitoring could be focused; this suggestion has not been validated with practitioners.

From explanation to action: what the explanations can and cannot support

A faithful explanation is not automatically an actionable one. Contrastive methods like RDX and MSX align well with managerial intuition answering *why order more rather than less?* by isolating the reward components favoring a given order level and can be audited against reward weights. Yet their scope is restricted to single-step trade-offs, overlooking downstream policy effects (Section 3.3.3). Input-based attributions, by contrast, stop short of prescribing action. SHAP values capture statistical sensitivity rather than causal impact, leaving unclear what a manager should change or what a different order would yield. Attention weights [22] suffer from a related flaw: high internal weighting does not guarantee that changing those inputs would alter the final decision [32]. In practice, then, feature attributions work best as diagnostic tools to uncover unexpected model dependencies (Case 3), while contrastive explanations are stronger candidates for justifying specific orders. Bridging the gap to decision support will require counterfactual analysis such as determining the minimal demand or inventory shift needed to change an order and direct testing with practitioners. 

## Reliability and Stability of the Explanations

The reliability checks on the explanations are reported in several places (Sections 2.3.4 and 3.3–3.8). Table 18 collects them into a single scorecard organized by criterion: faithfulness (do the highlighted reward components and state features matter to the agent’s output?), stability (does the explanation change when analysis choices that should not matter are varied?), consistency (does it change across scenarios, explained targets, or the learned value?), reproducibility across independently trained agents, and practical interpretability (does it help a person decide?). For each criterion the table gives the metric, the result for each agent, and where it is reported; criteria that have not been measured are marked as such.

Four points follow from the scorecard. First, faithfulness holds for the ordering of explanations but with small effects: masking MSX-selected reward components changes the action far more often than random masking once the threshold is strict (λ ≥ 1.5, MEDIUM and HARD), and masking top-ranked SHAP features lowers the output more than random or least-relevant masking in every condition, yet masking up to ten SHAP features never changes the selected action and never moves the output by more than 1%. The SHAP rankings are therefore faithful as rankings, not evidence that any single feature determines the decision. Second, stability depends on what is varied. Macro-level SHAP rankings are unaffected by the size and sampling of the background but change when its origin changes, and micro-level SKU rankings are unstable across backgrounds and across the number of explained states, so they are reported as indicative only. Third, the MSX set depends on the threshold λ: when λ moves away from 1.0 it changes in roughly half of the states for DQN and in about 70% of the states for A2C_mod, so an MSX should always be reported together with its threshold. Fourth, the attributions of A2C_mod depend on the explained target (policy, logits, or state value), so a single A2C_mod attribution should not be read as the agent’s explanation.

**Table 18. Interpretability scorecard: reliability and stability evidence for the explanations.**

**Criterion**

**Metric**

**DQN**

**A2C_mod**

**Reported in**

**Faithfulness**

Reward-space faithfulness

Action flip rate, MSX-guided vs. random masking of reward components

Agents pooled: MSX > random for λ ≥ 1.5 in MEDIUM and HARD (p < 0.001); not significant for λ ≤ 1.0; no flips in EASY

Section 3.8.1, Figs. 19–20

Feature-space faithfulness

Area under perturbation curve: MoRF / Random / LeRF

0.010–0.014 / −0.001 / −0.010 to −0.011

0.024–0.025 / 0.005–0.007 / −0.020 to 0.0002

Table 16

Ordering significance

Paired test with correction; effect size

MoRF > Random and MoRF > LeRF significant in all conditions; large to very large effects

Section 3.8.5

Decision-level effect of feature masking

Action switching rate; output change at maximum masking (10 of 660 features)

No action switch in any of 18 conditions (Wilson upper bound ≈ 7%); output change 0.16–0.40%, below the 1% threshold

Sections 3.8.4, 3.8.6

Consistency across states

Coefficient of variation of output change at maximum masking

0.19 (EASY) to 0.48 (HARD)

0.02–0.17

Section 3.8.7

**Stability**

MSX threshold λ

Share of states whose MSX equals the λ = 1.0 set (one checkpoint)

52–53% (λ = 1.5); 44–46% (λ = 2.0)

27–29% for all λ > 1.0

Section 3.7.3, Fig. 14

Joint reward-weight changes

MSX inclusion rate, holding × order pair

0.000–0.988 (agents and scenarios pooled)

Section 3.5.2

SHAP background, macro level

Rank correlation: across size and sampling; across origin

1.00; 0.87 (single rank swap)

Section 2.3.4.2

SHAP background, micro level (660 features)

Magnitude similarity; mean rank correlation

0.94–0.99; ≈ 0 (rankings indicative only)

Section 2.3.4.2

Number of explained states

Top-20 Jaccard, n = 10 vs. n = 20 and 50 (logit target)

0.25–0.29

0.25–0.67

Section 3.6.6

Validity of perturbed states

Share of physically valid states, synthetic background (EASY / MEDIUM / HARD); trajectory background

95.5% / 98.8% / 84.5%; 100%

Section 2.3.4.3

Repeated SHAP estimation

SD of FCS over 5 evaluations of one checkpoint

0.0000

0.0003–0.0017

Tables 14–15

**Consistency**

Across scenarios

Top-20 micro-level SHAP, Jaccard / Spearman between scenario pairs

1.00 / 1.00

0.29–0.33 / 0.20

Section 3.6.4

Across explained targets

Top-20 Jaccard: logits vs. post-activation output (EASY); policy vs. state value

0.667; not applicable

0.250; 0.250

Sections 3.6.6–3.6.7

With the learned value

Correlation between Q and the one-step component sum

0.02–0.21

−0.13 to −0.02

Table 9

Action-space resolution

MSX stability at 7 / 14 / 28 levels (one seed)

about 52–60

up to 90.7 (7 levels)

Section 3.9

**Practical interpretability**

Usefulness to inventory managers

Study with users (decision quality, appropriate reliance, understanding)

Not measured (listed as a limitation)

Section 3.12; Limitations

# Conclusion

This study proposed an Explainable Reinforcement Learning (XRL) framework that combines Reward Difference Explanation (RDX), Minimal Sufficient Explanation (MSX), and SHAP to explain the replenishment decisions of two DRL agents, DQN and A2C_mod, in a 220-product inventory environment derived from Instacart data. Operationally, the classical base-stock policy attains the highest unit-weighted total reward, whereas both agents almost eliminate stockouts at the price of higher holding and waste costs (Section 3.2); which policy is preferable therefore depends on how a retailer values a stockout relative to excess inventory and waste. In the reward space, RDX and MSX show that A2C_mod shifts the weight of its objectives as the stress-test scenarios become harder, whereas DQN keeps a narrower and more stable set of dominant objectives (Sections 3.3–3.4). In the feature space, SHAP indicates that DQN relies mainly on inventory and demand signals, whereas the attributions of A2C_mod are spread more evenly across features; because the two explained outputs differ in scale, this contrast is qualitative (Section 3.6).

The reliability checks qualify these findings (Section 3.13, Table 18). The explanations are faithful as rankings: masking MSX-selected reward components changes the action more often than random masking at strict thresholds, and masking top-ranked SHAP features lowers the output more than random or least-relevant masking, although masking up to ten features never changes the selected action (Section 3.8). The one-step RDX and MSX understate the long-horizon weight of service and agree with a trained decomposition on the dominant component in only about half of the DQN decisions (Section 3.3.4). MSX Stability varies widely across five independently trained runs of each agent, with no significant difference between the agents, so stability values from a single checkpoint should not be read as properties of an architecture, and SKU-level SHAP rankings are indicative only. The differences reported here therefore characterize the evaluated DQN and A2C_mod implementations under the present environment, action, and reward settings, not the algorithm families (Section 3.10). Finally, the explanations have been assessed against the agents, not against their users; whether they help inventory managers make better or faster replenishment decisions remains to be tested (Section 5).

# Study Limitations and Future Work

Although the proposed framework provides structured insights into reasoning in both reward space and feature space, several limitations point to directions for future research. First, the reward explanations are computed one step ahead; Section 3.3.4 shows that they understate the long-horizon contribution of service, so long-horizon decompositions should be trained and explained directly, and explanation stability, which varies strongly between training runs (Section 3.13), should be assessed with more runs than the five used here. Second, future studies should extend SHAP analysis beyond aggregated features to more fine-grained product-level and system-level representations to mitigate information loss due to dimensionality reduction. Third, the explanations were evaluated only against the agents (faithfulness and stability), not against their intended users: no study with inventory managers was conducted, so it remains unknown whether RDX, MSX, or SHAP outputs improve the quality or speed of replenishment decisions, or whether they help planners rely on the agent appropriately. That is, follow it when it is right and override it when it is wrong. Such a study would present planners with agent recommendations with and without each explanation type and measure decision quality, appropriate reliance, and self-reported understanding, and would also test whether the identified trade-offs and feature allocations match real-world managerial reasoning. In addition to methodological extensions, applying the framework to other DRL algorithms such as PPO or SAC would enable assessment of the generalizability of the explanation mechanisms across policy-gradient and actor-critic variants. Fourth, the action space remains a coarse abstraction: all SKUs share one weight-sharing policy over 14 ordering-rate levels, so the framework cannot represent continuous or SKU-specific ordering rules; extending it to continuous or SKU-specific parameterized action spaces is a significant direction for future research. Fifth, the EASY, MEDIUM, and HARD scenarios are controlled transformations of a single Instacart-based test segment rather than independent environments; generalization to other retailers, assortments, demand processes, and lead-time structures has not been tested and requires evaluation on independent datasets. Furthermore, incorporating richer operational constraints such as budget limits, delivery lead times, and supply chain risks would enhance the practical applicability of the model.

# Data and code availability

The code, the released demand, capacity, and initial-stock files derived from the Instacart Market Basket Analysis data [25], the trained checkpoints, and the reproducibility material are available at https://github.com/EgawaMasao/QLKHO-RL, archived on Zenodo. The repository documents the product selection procedure, the reconstruction of timestamps and demand, the simulated variables (capacity, initial inventory, spoilage), the reward and metrics as implemented, all hyperparameters and training seeds, the chronological split, and the evaluation protocol of each table and figure. It includes a script that verifies the released data files against published checksums, a script that reproduces the Base-Stock baseline, and a script that recovers the Instacart product identifiers of the 220 products from the public Instacart files, which are not redistributed.

# Credit author statement

Dac Hoang Nguyen: Conceptualization of the review article, Literature search, Writing the manuscript as the main author and corresponding author; Egawa Masao, Viet Pham Quoc Le: Reviewing, Data synthesis, Critical feedback; Nhu Tai Do: Editing, Refining the discussion and conclusions.

# Competing interests

The authors declare that there is no conflict of interest regarding the publication of this article.

# References

[1]	T. M. Whitin, “Inventory Control Research: A Survey,” Management Science, vol. 1, no. 1, pp. 32–40, 1954.

[2]	G. Liu, W. Deng, X. Xie, L. Huang, and H. Tang, “Human-Level Control Through Directly Trained Deep Spiking Q-Networks,” IEEE Transactions on Cybernetics, vol. 53, no. 11, pp. 7187–7198, Oct. 2023, doi: 10.1109/TCYB.2022.3198259.

[3]	V. Mnih et al., “Asynchronous Methods for Deep Reinforcement Learning,” in Proceedings of The 33rd International Conference on Machine Learning, PMLR, Jun. 2016, pp. 1928–1937. Accessed: Mar. 05, 2026. [Online]. Available: https://proceedings.mlr.press/v48/mniha16.html

[4]	H. Meisheri, V. Baniwal, N. N. Sultana, H. Khadilkar, and B. Ravindran, “Using Reinforcement Learning for a Large Variable-Dimensional Inventory Management Problem.”

[5]	E. Puiutta and E. M. S. P. Veith, “Explainable Reinforcement Learning: A Survey,” in Machine Learning and Knowledge Extraction, A. Holzinger, P. Kieseberg, A. M. Tjoa, and E. Weippl, Eds., Cham: Springer International Publishing, 2020, pp. 77–95, doi: 10.1007/978-3-030-57321-8_5.

[6]	Z. Juozapaitis, A. Koul, A. Fern, M. Erwig, and F. Doshi-Velez, “Explainable Reinforcement Learning via Reward Decomposition,” in Proc. IJCAI/ECAI 2019 Workshop on Explainable Artificial Intelligence (XRL-2019), Macao, China, Aug. 2019.

[7]	S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” in Advances in Neural Information Processing Systems 30 (NeurIPS 2017), 2017, pp. 4765–4774.

[8]	D. Beechey, T. M. S. Smith, and Ö. Şimşek, “Explaining Reinforcement Learning with Shapley Values,” in Proceedings of the 40th International Conference on Machine Learning, PMLR, Jul. 2023, pp. 2003–2014. Accessed: Mar. 05, 2026. [Online]. Available: https://proceedings.mlr.press/v202/beechey23a.html

[9]	M. Sieke, “Foundations of Inventory Management,” in Supply Chain Contract Management: A Performance Analysis of Efficient Supply Chain Contracts, M. Sieke, Ed., Wiesbaden: Springer Fachmedien, 2008, pp. 9–36. doi: 10.1007/978-3-658-24382-1_2.

[10]	E. A. Silver, D. F. Pyke, and D. J. Thomas, Inventory and production management in supply chains, Fourth edition, First issued in paperback. Boca Raton, FL London New York, NY: CRC Press, Taylor & Francis Group, 2021.

[11]	J. Si, A. G. Barto, W. B. Powell, and D. Wunsch, “The Linear Programming Approach to Approximate Dynamic Programming,” in Handbook of Learning and Approximate Dynamic Programming, IEEE, 2004, pp. 153–178. doi: 10.1109/9780470544785.ch6.

[12]	R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. Cambridge, MA: MIT Press, 2018.

[13]	V. Mnih et al., “Human-level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–533, Feb. 2015, doi: 10.1038/nature14236.

[14]	Taherinavid et al., “Automatic Transportation Mode Classification Using a Deep Reinforcement Learning Approach With Smartphone Sensors,” IEEE Access, vol. 12, pp. 514–533, 2024, doi: 10.1109/ACCESS.2023.3346875.

[15]	Yang et al., “Wastewater Treatment Monitoring: Fault Detection in Sensors Using Transductive Learning and Improved Reinforcement Learning,” Expert Systems with Applications, vol. 264, p. 125805, 2025, doi: 10.1016/j.eswa.2024.125805.

[16]	A. Barredo Arrieta et al., “Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI,” Information Fusion, vol. 58, pp. 82–115, Jun. 2020, doi: 10.1016/j.inffus.2019.12.012.

[17]	A. de-la-Rica-Escudero, E. C. Garrido-Merchán, and M. Coronado-Vaca, “Explainable Post Hoc Portfolio Management Financial Policy of a Deep Reinforcement Learning Agent,” PLOS ONE, vol. 20, no. 1, p. e0315528, Jan. 2025, doi: 10.1371/journal.pone.0315528.

[18]	A. Kuznietsov, B. Gyevnar, C. Wang, S. Peters, and S. V. Albrecht, “Explainable AI for Safe and Trustworthy Autonomous Driving: A Systematic Review,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 12, pp. 19342–19364, Oct. 2024, doi: 10.1109/TITS.2024.3474469.

[19]	D. Das, S. Banerjee, and S. Chernova, “Explainable AI for Robot Failures: Generating Explanations that Improve User Assistance in Fault Recovery,” in Proc. 2021 ACM/IEEE Int. Conf. on Human-Robot Interaction (HRI ’21), 2021, doi: 10.1145/3434073.3444657.

[20]	Yang et al., “Explainable Deep Reinforcement Learning for Anomaly Detection in IoT-Enabled Metaverse Healthcare: Toward Trustworthy Cyber Threat Intelligence,” Research, vol. 9, Art. no. 1245, 2026, doi: 10.34133/research.1245.

[21]	R. N. Boute, J. Gijsbrechts, W. van Jaarsveld, and N. Vanvuchelen, “Deep reinforcement learning for inventory control: A roadmap,” European Journal of Operational Research, vol. 298, no. 2, pp. 401–412, Apr. 2022, doi: 10.1016/j.ejor.2021.07.016.

[22]	W. Kool, H. van Hoof, and M. Welling, “Attention, Learn to Solve Routing Problems!,” in Proc. 7th International Conference on Learning Representations (ICLR 2019), New Orleans, LA, USA, 2019.

[23]	Liu et al., “A Systematic Literature Review of Vehicle Routing Problems with Time Windows,” Sustainability, vol. 15, no. 15, p. 12004, 2023, doi: 10.3390/su151512004.

[24]	Liu et al., “A Transductive Learning-Based Method for Vehicle Routing Problems Using Off-Policy Proximal Policy Optimization and Hyperparameter Optimization,” Information Sciences, Art. no. 123671, 2026, doi: 10.1016/j.ins.2026.123671.

[25]	“The Instacart Online Grocery Shopping Dataset 2017,” Instacart. Accessed: Mar. 2026. [Online]. Available: https://www.instacart.com/datasets/grocery-shopping-2017 (distributed via Kaggle: [https://www.kaggle.com/c/instacart-market-basket-analysis/data](https://www.kaggle.com/c/instacart-market-basket-analysis/data)).

[26]E. Altman, Constrained Markov Decision Processes. Boca Raton, FL: Chapman & Hall/CRC, 1999, doi: 10.1201/9781315140223.

[27]J. Achiam, D. Held, A. Tamar, and P. Abbeel, “Constrained Policy Optimization,” in Proceedings of the 34th International Conference on Machine Learning, PMLR, vol. 70, 2017, pp. 22–31.

[28]C. Tessler, D. J. Mankowitz, and S. Mannor, “Reward Constrained Policy Optimization,” in Proc. 7th International Conference on Learning Representations (ICLR 2019), New Orleans, LA, USA, 2019.

[29]H. Mao, M. Alizadeh, I. Menache, and S. Kandula, “Resource Management with Deep Reinforcement Learning,” in Proc. 15th ACM Workshop on Hot Topics in Networks (HotNets 2016), 2016, pp. 50–56, doi: 10.1145/3005745.3005750.

[30]J. Gijsbrechts, R. N. Boute, J. A. Van Mieghem, and D. J. Zhang, “Can Deep Reinforcement Learning Improve Inventory Management? Performance on Lost Sales, Dual-Sourcing, and Multi-Echelon Problems,” Manufacturing & Service Operations Management, vol. 24, no. 3, pp. 1349–1368, 2022, doi: 10.1287/msom.2021.1064.

[31]S. Padakandla, “A Survey of Reinforcement Learning Algorithms for Dynamically Varying Environments,” ACM Computing Surveys, vol. 54, no. 6, pp. 1–25, 2021, doi: 10.1145/3459991.

[32]S. Jain and B. C. Wallace, “Attention is not Explanation,” in Proc. 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT 2019), 2019, pp. 3543–3556, doi: 10.18653/v1/N19-1357.