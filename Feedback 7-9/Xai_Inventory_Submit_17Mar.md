# Explaining Deep RL Agents in Multi - Product Inventory Management:

# An XRL Framework

1* 2 2 2

Dac Hoang Nguyen, Egawa Masao, Viet Pham Quoc Le, Nhu Tai Do

1 Dept. Information Technology, Tay Do College, Can Tho, Vietnam 2 Dept. Information Technology, Saigon University, Ho Chi Minh, Vietnam

Abstract:

Inventory management requires balancing demand fulfillment, cost reduction, and waste minimization. While Deep Reinforcement Learning (DRL) optimizes these systems, its black - box nature limits interpretability and operational trust. This study proposes an Explainable Reinforcement Learning (XRL) framework integrating Reward Difference Explanation (RDX), Minimal Sufficient Explanation (MSX), and SHapley Additive exPlanations (SHAP) to analyze multi - product replen ishment system. Utilizing an inventory simulation with DQN and A2C_mod agents, we introduce a reward decomposition mechanism aligned with core business objectives. A classical base - stock policy serves as an operational baseline. Furthermore, SHAP is extended via top - k feature - level analysis, and perturbation - based faithfulness tests are introduced to validate explanations by masking key components. Evaluated on the Instacart dataset (220 products), results reveal distinct operational trade - offs. While the base - stock policy provides a strong reference, DRL agents prioritize service levels and stockout reduction. RDX and MSX effectively clarify reward - space trade - offs, whereas SHAP provides complementary feature - space evidence. Overall, this multi - level XRL framework enhances the transparency of portfolio - level DRL replenishment analysis.

Keywords: Explainable Reinforcement Learning, Inventory Management, Deep Q - Network, Advantage Actor - Critic, Reward Decomposition, SHAP.

Classification numbers: -

decisions must be understandable to increase system

1. Introduction

trustworthiness.

Large scale inventory management is a complex

Most existing studies on DRL for inventory

decision - making problem. Managers must simultaneously

management focus primarily on performance optimization,

balance multiple conflicting objectives such as meeting

such as maximizing cumulative rewards or minimizing long

customer demand, reducing inventory holding costs, and

term average costs, while paying limited attention to

limiting waste due to spoilage or expiration. At the same

explaining the decision making behavior of agents [4, 5]. In

time, decisions must comply with constraints on

these studies, agents are typically evaluated using

transportation capacity, ordering costs, and storage

quantitative metrics, whereas the trade - offs among different

capacity. As the number of products increases, the state and

business objectives are not analyzed in a transparent manner.

action spaces expand rapidly, making traditional

In parallel, the f ield of XRL has been developed to address

optimization methods difficult to scale to large real world

the black box issue of reinforcement learning models.

problems [1]. DRL has been demonstrated as an effective

Representative approaches include explanations based on

approach for sequential decision - making pro blems with

reward decomposition such as RDX and MSX, which enable

large state spaces. In particular, methods such as Deep Q

clarification of the role of each reward componen t in the

Network (DQN) and Advantage Actor Critic (A2C) have

agent decisions [6]. In addition, feature level explanation

shown the ability to learn effective policies in many

methods such as SHAP have been applied to analyze how

complex control and optimization tasks [2, 3]. In the

state features influence the agent value function or policy [7,

context of inventory management, DRL agents such as

8]. Although XRL has attracted increasing attention, most

DQN and A2C_mod can achieve high performance and

existing studies still focus o n individual explanation

good scalability for multi - product inventory management

methods or are evaluated in simple experimental

problems with variable dimensionality [4]. However,

environments. There remains a lack of studies that

despite achievi ng high optimal performance in many

comprehensively apply multiple XAI RL explanation

complex tasks, DRL models often operate as black boxes.

methods to large scale inventory management problems, as

This issue makes the agents' decision - making processes

well as a lack of systema tic comparative analyzing the

difficult to understand and interpret intuitively [5]. This

explainability of decision - making behavior between DRL

lack of transparency is particularly critical in real - world

agents based on different mechanisms, such as DQN and

inventory management systems, where operational

A2C_mod.

Considering the limitations mentioned above, this Reinforcement Learning (RL) provides a data - driven study proposes an XRL framework for large scale multi - approach that allows an agent to learn a policy directly prod uct inventory management, built upon the simulation through interac tion with the environment without requiring environment and DRL agents adopted from prior work [4]. an explicit dynamics model [12]. Inventory problems are

commonly modeled as Markov Decision Processes (MDPs).

Within this framework, the study systematically integrates

Among modern RL algorithms, Deep Q - Network (DQN)

three complementary explanation methods to elucidate the

and Advantage Actor - Critic (A2C) are widely used for

decision - making behavior of the agents.

problems with large state spaces. DQN learns the action -

This study makes the following main contributions: value function 𝑸 (𝒔, 𝒂) [13], while A2C combines policy

learning and value estimation to improve training stability

- Proposes a comprehensive XRL framework for

[3]. In recent developments, DQN and A2C_mod have

large scale multi - product inventory management, directly

achieved promising performance and scalability in large -

extend ing DQN and A2C_mod from Meisheri et al (2020). scale multi - product inventory management problems [4].

by integrating three complementary explanation methods However, these studies focus primarily on aggregate RDX, MSX, and SHAP. performance, while the decision - making mechanisms and

trade - offs among operational objectives are not analyzed

- Develops a practical reward decomposition

transparently. T his highlights the need for Explainable

mechanism, enabling RDX to clarify trade - offs among

Reinforcement Learning methods that can explain both the

operational objectives and MSX to ide ntify the core set of operational performance and the internal decision logic of reasons that justify the agent decisions. DRL agents.

- Uses SHAP to provide aggregated system - level 2.2. Explainable Reinforcement Learning

feature attribution and complements it with top - k feature - XRL aims to enhance the interpretability and level analysis. tra nsparency of the decision - making behavior of

reinforcement learning agents. Due to the characteristics of

- Introduces perturbation - based faithfulness

sequential decision making, delayed rewards, and the

evaluation for the proposed explanations. presence of dynamic trade - offs among multiple

optimization objectives, explaining the behavior of DRL

- Conducts a systematic comparative analysis of the

models becomes particula rly challenging compared to

explainability of DQN and A2C_mod, thereby elucidating

supervised learning models [5, 14]. XRL has been

the impact of decision - making mechanisms on the

developed to provide meaningful explanations, thereby

effectiveness and characteristics of each XRL approach.

increasing trustworthiness, verifiability, and decision support in high - risk domains such as finance [15],

2. Literature Review

autonomous transportation [1 6], and robotic learning [17].

2. 1. Reinforcement Learning for Inventory XRL methods are commonly divided into two main

Management categories. Intrinsic explanation methods integrate

explanatory mechanisms directly into the agent’s

Inventory management is an important dynamic

learning process, explo iting the internal structure of the

decision - making problem in supply chains and has been

RL problem — particularly the reward — to clarify

approached through heuristic methods, Approximate Dynamic Programming (ADP), and optimization. Heuristic optimization motives and long - term strategies [18].

,

pol icies such as the (𝒔 𝑺) policy and base - stock policy have Within this category, reward - based explanations advantages in terms of simplicity, interpretability, and low methods decompose the total reward into semantically implementation cost [9, 10]. However, these policies often meaningful co mponents. In particular, Reward rely on static assumptions and may not fully capture complex Difference Explanation (RDX) explains differences interactions in multi - product inventory systems with shared between actions through the contribution of each reward constraints, fluctuating demand, spoilage risk, and limited component, while Minimal Sufficient Explanation storage or transportation capacity. ADP methods improve (MSX) identifies the smallest subset of core reasons that performance by approximating value functions or policies is still suf ficient to justify the agent’s decision [6].

[11]; however, they depend heavily on t he dynamics model However, these methods are highly dependent o n the and face computational challenges at large scales. Despite design and decomposition of the reward, which may these limitations, classical inventory policies remain

limit generalization. In contrast, post hoc explanation

important operational baselines because they are simple,

methods generate explanations after the RL model has

interpretable, and widely used in inventory control.

been trained by analyzing the relationship between the

Therefore, com paring DRL agents against a classical policy

agent’s input states and outputs. A common approach is

such as base - stock is necessary to assess whether the learned

featuring attribution, in which SHapley Additive

policies are operationally competitive, rather than only

exPlanations (SHAP) are extended from supervised

different from each other. Such a comparison also provides

learning to explain DRL models by quantifying the

a practical reference point for i nterpreting whether the

influence of each input feature on the Q value or action

explanations produced by XRL methods are attached to

policy [8, 18]. Although highly flexible, post hoc

policies with meaningful inventory - control performance.

methods often struggle to directly reflect long term

optimization objectives and the intrinsic trade - offs agent observes the state 𝑠 𝑡, selects an action 𝑎 𝑡 ∈ 𝐴, and encoded in the agent’s reward. receives a reward 𝑅

𝑡, following the dynamics of MDP.

2. 3 Explainable Reinforcement Learning in Inventory 𝑠 𝑡 + 1 ∼ 𝑃 (𝑠 𝑡, 𝑎 𝑡), 𝑅 𝑡 = 𝑅 (𝑠 𝑡, 𝑎 𝑡)

Management

The state 𝑠 𝑡 reflects the operational condition of the

Although deep reinforcement learning has been system and includes both product level and system level applied effectively to many complex inventory features. For each of the 220 products, the state consists of management and supply chain problems [20], research the inventory level 𝐼 𝑖, 𝑡, the demand forecast 𝐷 𝑖, 𝑡, and the adopting the XRL perspective in this domain remains spoilage indicator 𝑊

𝑖, 𝑡. I n addition, the state includes

limited. Most existing studies focus on improving aggregated system level features: warehouse utilization optimization performance, while the explanation of

rate 𝑈 𝑡, transportation capacity 𝐶 𝑡, demand volatility 𝑉 𝑡,

agents’ decision - making behavior especially in l arge scale

and the time until the next ordering period 𝑇 𝑡. The full

multi product inventory systems with numerous

state vector is represented as follows:

operational constraints has not been adequately addressed.

𝑠 𝑡 = [𝐼 1, 𝑡, 𝐷 1, 𝑡, 𝑊 1, 𝑡, …, 𝐼 220, 𝑡, 𝐷 220, 𝑡, 𝑊 220, 𝑡, 𝑈 𝑡, 𝐶 𝑡, 𝑉 𝑡, 𝑇 𝑡]

Some early studies have examined policy explanation or

with a total dimensionality of |S| = 664. The action

the importance of state features in relatively simple logistics and supply chain prob lems, for example through space is discretized into 14 ordering strategies 𝐴 = attention mechanisms or feature attribution in vehicle 𝑎 0, 𝑎 1, …, 𝑎 13, corresponding to replenishment levels routing and scheduling tasks [21]. However, these studies ranging from co nservative to aggressive for the entire typically rely on a single explanation mechanism and are product portfolio, subject to capacity and storage evaluated in small scale environments, making it difficult constraints. The two agents, DQN and A2C_mod, are to fully reflect the complex trade - offs present in real world trained in the same environment and configuration to inventory systems. ensure fairness in comparing performance and

explainability. Figure 1 details the workflow used to

In the context of multi product i nventory management,

explain reinforcement learning decisions within the

existing XRL studies mainly remain at the level of

study’s warehouse management system. The action space

analyzing state features or providing high level policy

is discretized into 14 ordering strategies, corresponding to

explanations, without clarifying the different operational

replenishment levels ranging from conservative to

objectives encoded in the agent’s reward. In particular, the

a ggressive for the entire product portfolio. These actions

key trad e - offs among demand fulfillment, inventory

are interpreted as portfolio - level replenishment strategies

holding costs, waste, and ordering costs have not yet been

rather than fully independent SKU - level order quantities.

transparently analyzed through explanation mechanisms with clear business semantics. Furthermore, no study has Lower - index actions represent conservative been performed that simultaneously combines XRL replenishment, while higher - index act ions indicate more methods based on rewards and state characteristics, nor aggressive ordering to reduce stockout risk. This design any systematic comparative analysis of explanatory keeps the learning problem tractable in a 220 - product capability across such as DQN and A2C_mod in large environment, but it also abstracts away fine - grained SKU - scale multi product inventory management problems level ordering constraints.

involving hundreds of Stock Keeping Units (SKU).

Motivated by these gaps, the present study builds on the environment and agents proposed by Meisheri et al.

(2020) [4] and proposes a comprehensive XRL framework for large scale multi product inventory management.

3. Methodology
3. 1. Inventory Management Environment

This study builds upon the large - scale inventory

Fig. 1. Overall XRL workflow for explaining the agent’s

management environment of many products in which the decisions regarding inventory management.

problem is modeled as an MDP. At each time step 𝑡, the Table 1. Operational interpretation of the 14 discrete actions

Action range Replenishment level Operational meaning a 0 Minimum / no replenishment Avoids ordering when inventory is sufficient or spoilage risk is high a 1 - a 3 Low replenishment Conservative ordering to reduce holding and waste costs a 4 - a 8 Medium replenishmen Balanced strategy between demand fulfillment and cost control a 9 - a 13 High replenishment Aggressive ordering to prevent stockouts under high demand or low inventory

3.1.3 Data Split and Chronological Protocol

A central concern in inventory forecasting is temporal leakage, where future demand patterns inadvertently inform training. To avoid this, the demand series is partitioned strictly chronologically rather than randomly. The underlying time index is defined in fixed intervals, and the sequence is divided into contiguous blocks: an initial training period, a subsequent validation period carved from the tail of the training sequence, and a held-out test period. No shuffling or random sampling is applied, and the training and testing intervals correspond to two disjoint physical partitions of the time series. This design ensures that the model is always evaluated on genuinely future observations, analogous to learning from historical records to predict unseen future demand, without access to the answer beforehand.

A validation segment is reserved chronologically from the end of the training interval specifically for hyperparameter selection. In particular, the safety factor that governs the classical base-stock policy is tuned on this validation interval and never on the test interval, preserving the test set as a final, unbiased evaluation ground. The three operational scenarios examined in the ablation studies do not constitute re-splits of the data; rather, they are controlled demand and waste scalings applied on top of the test interval to create conditions of varying difficulty.

Quantitatively, the training interval spans one thousand steps, the validation interval two hundred steps, and the test interval roughly five hundred steps. Normalized demand, defined as raw demand divided by product capacity, averages an order of magnitude lower in the held-out interval than in the training interval, while per-product correlations between training and test are moderately negative. This indicates a natural distributional drift from training to test, so that the more demanding scenarios represent controlled out-of-distribution conditions on top of an already shifted baseline. Capacity itself is heterogeneous across products and, by construction, an order of magnitude larger than average demand.

Table 1b. Chronological partition and demand characteristics.

| Partition | Time interval | Steps | Mean normalized demand | Interpretation |
| :--- | :--- | ---: | ---: | :--- |
| Training | Early period | 1000 | 0.10 | Primary learning interval |
| Validation | Tail of training | 200 | 0.055 | Hyperparameter selection, chronologically after training |
| Test | Later period | 504 | 0.034 | Held-out future evaluation, naturally shifted |

[Figure A — Chronological timeline] - output_audit_33_timeline.png

*Figure A illustrates the chronological timeline with three contiguous blocks: training, validation, and test. No shuffling is applied, ensuring that future information never leaks into training.*

[Figure B — Demand distribution] - output_audit_33_histogram.png

*Figure B compares the distribution of normalized demand between training and test and the per-product mean scatter. The held-out interval is systematically lower and the per-product correlation is negative, evidencing natural out-of-distribution drift.*

3.1.4 Leakage Audit and Corrected Procedure

To address concerns that model development may have inadvertently benefited from test information, four components are audited for leakage: demand forecasting, normalization, reward design, and baseline calibration. The audit examines for each component where its parameters are estimated, whether test information is involved, and, if so, what correction restores a strict train-only protocol. The rationale is to demonstrate that any influence of the test set is either absent or, where initially present, is small, correctable, and does not require a full retraining of the deep reinforcement learning agents for the present submission.

Demand forecasting is not a learned component in this study; the agents observe realized demand directly, normalized by capacity, without a separate forecasting model that requires fitting. Reward design is likewise fixed by construction: the composite reward combines service, holding, waste, and balance terms with a constant base and unity weights, the waste rate being a fixed perishable decay. Because no coefficients are estimated from data, neither forecasting nor reward introduces leakage; the choice of unity weights is justified independently through a one-at-a-time sensitivity analysis on held-out data reported elsewhere.

Two components initially involved test information and were therefore examined quantitatively. First, normalization relies on product capacity as a scaling constant. The current capacity was originally derived from the full time series, so its average exceeds the train-only counterpart by roughly one quarter, with a vast majority of products deviating by more than ten percent and a maximum deviation of one half. Despite this sizable discrepancy in the constant itself, the downstream effect on normalized demand is modest: the mean shift is only about two hundredths on training and less than one hundredth on testing, translating into a reward impact of less than two hundredths. The corrected procedure defines capacity solely from the training interval as a multiple of average training demand and regenerates the capacity record from training data alone. Given the limited downstream impact, the present reinforcement learning results remain valid, with the corrected capacity to be used in the revision.

Second, baseline calibration was examined. The target level of the base-stock policy is correctly estimated from training demand, but its safety factor was formerly selected on the test interval, which constitutes leakage. Re-evaluation on the validation interval carved chronologically from the end of training gives an optimum that differs by only half a unit from the test optimum, and the reward curves as a function of the safety factor are nearly parallel across validation and test. The correction is therefore to report the validation optimum and to retain the test optimum only as a reference, which removes leakage without requiring retraining of the deep agents. Conceptually, this corresponds to choosing the answer based on homework rather than on the final examination, while noting that the two choices remain close.

[Figure C — Capacity audit] - output_audit_34_capacity_diff.png

*Figure C shows per-product capacity deviation and its histogram. While the constant itself differs substantially when computed from training data alone, the resulting shift in normalized demand remains modest.*

[Figure D — Baseline safety factor] - output_audit_34_k_curve.png

*Figure D plots average reward against the safety factor on validation and on test. The two curves are nearly parallel and their maxima differ by only half a unit, demonstrating that selection on validation is safe and does not constitute overfitting to the test set.*

Table 1c. Leakage audit summary.

| Component | Estimation basis | Leakage | Correction |
| :--- | :--- | :--- | :--- |
| Demand forecasting | No model, observed demand | No | Documented as observed, train-only if added |
| Normalization (capacity) | Full series (initial) vs training-only (corrected) | Yes — sizable in constant, modest downstream | Regenerate capacity from training interval only |
| Reward design | Fixed, unity weights | No | Justified via sensitivity analysis |
| Baseline (base-stock) | Target on training (correct), safety factor on test (leaked) | Yes → Fixed | Select safety factor on validation interval |

Full quantitative details, including per-product capacity deviations and the safety-factor grid, are provided as supplementary material.

### 3.1.5 Product-Group Construction for Scalability Test (Task 5)

To test scalability to the product-group level, the 220 SKUs were partitioned into three groups based on mean demand. The Fast group contains 73 high-velocity SKUs with high mean demand and rapid turnover, the Medium group contains 73 SKUs with intermediate demand, and the Slow group contains 74 low-demand SKUs with high coefficient of variation and low volume. Each group was trained independently with 1,000 chronological periods. The Slow group contains 74 SKUs because 220 is not divisible by three, ensuring no SKUs are discarded.

[Figure 1]
The learning curve shows the episode-averaged reward for A2C across the three groups. The Fast group remains stable around 0.18 with very low variance, indicating limited learning gain. The Slow group converges steadily around 0.15. The Medium group exhibits high variance and a clear degradation from early to late training, suggesting instability under the current hyperparameters.

[Figure 2]
The learning curve shows the episode-averaged reward for DQN across the three groups. All groups start from low or negative initial rewards and improve strongly over 600 episodes, reaching approximately 0.77 for Fast, 0.65 for Medium and 0.57 for Slow. The consistent upward trend demonstrates effective learning in each group, with Fast converging to the highest final reward and Slow the lowest.

ℎ 𝑜𝑙𝑑

3. 2. Reward - based Explanations 𝑟

𝑡 = − ℎ 𝐼 𝑡

3. 2.1. Reward Decomposition:

Where:

To support business level explanations of the agent’s decision - making behavior, this study applies a reward • ℎ is the holding cost per unit, decomposition mechanism, in which the total reward at

- 𝐼 𝑡 is the ending inventory level.

time 𝑡 is represented as the sum of semantically meaningful reward components: • Waste: Reflects losses due to damaged or expired items,

𝑠𝑟𝑣 ℎ 𝑜𝑙𝑑 𝑤𝑎𝑠𝑡𝑒

𝑟 𝑡 = 𝑤 𝑠𝑟𝑣 𝑟 𝑡 + 𝑤 ℎ 𝑜𝑙𝑑 𝑟 𝑡 + 𝑤 𝑤𝑎𝑠𝑡𝑒 𝑟 encouraging efficient inventory management. 𝑡

𝑜𝑟𝑑𝑒𝑟

- 𝑤 𝑜 𝑟 𝑑𝑒𝑟 𝑟 𝑡 𝑤𝑎𝑠𝑡𝑒

𝑟 𝑡 = − 𝑐 𝑤 𝑊 𝑡

where

Where:

𝑠𝑟𝑣

- 𝑟 𝑡: level of demand satisfaction and stockout penalty

ℎ 𝑜𝑙𝑑 • 𝑐 𝑤 is the cost per unit of damaged or expired items,

- 𝑟 𝑡: inventory holding cost

𝑤𝑎𝑠𝑡𝑒 • 𝑊 𝑡 is the quantity of damaged items at time 𝑡.

- 𝑟 𝑡: loss due to spoilage or expiration

𝑜𝑟𝑑𝑒𝑟 • Ordering: Represents the cost incurred when p lacing

- 𝑟 𝑡: ordering and transportation cost

orders, including both fixed and variable costs, helping

- w c are weighting coefficients that reflect the relative balance order frequency and order size.

priority level of each operational objective.

𝑜𝑟𝑑𝑒𝑟 𝑟 𝑡 = − (𝑐 𝑜 𝑄 𝑡 + 𝑐 𝑓 𝟏 𝑄

𝑡 > 0)

Similarly, the action value function is also decomposed into the sum of individual value components. Where:

𝐾 (𝑘)

𝑄 (𝑠, 𝑎) = ∑ 𝑄 (𝑠, 𝑎), • 𝑄

𝑡 is the order quantity,

𝑘 = 1

- 𝑐 𝑜 is the variable cost per unit ordered,

(𝑘)

where each 𝑄 (𝑠, 𝑎) represents the long - term contribution of a specific reward component. • 𝑐 𝑓 is the fixed cost per order, To ensure mathematical rigor as well as consistency • 𝟏 𝑄

𝑡 > 0 is an indicator variable equal to 1 if an

with the explanation mechanism, each component is

order is placed and 0 otherwise.

specifically defined as follows.

3. 2.2. Reward Difference Explanation
- Service: Reflects the level of demand fulfillment at each

RDX is used to explain why the agent prefers the action

time step. A high value indicates that the system performs

𝑎 1 over 𝑎 2 in the same state 𝑠, that is, when 𝑄 (𝑠, 𝑎 1) >

well in serving demand and limiting stockouts.

𝑄 (𝑠, 𝑎 2). According to Juozapaitis et al. (2019) [6], RDX is defined as the difference in values between the two

min (𝐼 𝑡, 𝐷 𝑡) actions:

𝑠𝑟𝑣, if 𝐷 𝑡 > 0 𝑟 𝑡 = {𝐷 𝑡 𝛥 (𝑠, 𝑎

1, 𝑎 2) = 𝑄 (𝑠, 𝑎 1) − 𝑄 (𝑠, 𝑎 2)

1, if 𝐷 𝑡 = 0

with the reward decomposition mechanism, the action value is represented as the sum of component values

Where:

(𝑘) 𝑄 (𝑠, 𝑎); therefore, RDX can be decomposed according to individual operational objectives:

- 𝐼 𝑡 is the available inventory at time 𝑡,

𝐾

- 𝐷 𝑡 is the realized demand. (𝑘)

𝛥 (𝑠, 𝑎 1, 𝑎 2) = ∑ 𝛥 (𝑠, 𝑎 1, 𝑎 2),

- Holding: Represents the cost of maintaining inventory. 𝑘 = 1

(𝑘) (𝑘) (𝑘)

This component helps prevent excessive stock 𝛥 (𝑠, 𝑎 1, 𝑎 2) = 𝑄 (𝑠, 𝑎 1) − 𝑄 (𝑠, 𝑎 2).

accumulation.

Where

(𝑘)

- 𝛥 > 0: component 𝑘 supports choosing 𝑎 In addition to reward space based explanation 1

methods, this study employs SHAP to analyze the agent’s

(𝑘)

- 𝛥 < 0: component 𝑘 opposes 𝑎 1 decision making behavior in the input state feature space.

SHAP enables quantification of the contribution of each

(𝑘)

In the inventory management problem, each 𝛥 state feature to the agent’s value estimate or policy, corresponds directly to a business objective thereby clarifying the key signals that the DRL model (service/stockout, holding cost, waste, order cost). As a relies on when making decisions.

result, RDX helps reveal the trade - offs by showing which

3. 3.1. SHAP for DQN

objectives favor the selection of 𝑎 1 and which objectives

For the DQN agent, SHAP is applied to explai n the

incur disadvantages, thus enabling r eaders to more easily

action value function 𝑄 (𝑠, 𝑎). Specifically, the 𝑄 value of

compare the contributions between different objectives.

an action in state 𝑠 is approximated as the sum of the

3. 2.3. Minimal Sufficient Explanation: contribution of features:

𝑁

Although RDX provides detailed information on the

𝑄 (𝑠, 𝑎) ≈ 𝜙 0 + ∑ 𝜙 𝑖

contribution of each reward component, the number of

𝑖 = 1

reasons can become large and difficult to interpret in

where 𝜙 denotes the contribution of the 𝑖 − 𝑡 ℎ state

practice. To address this limitation, MSX is proposed to 𝑖

feature to the evaluation of the action 𝑎, and 𝜙 is the

identify the smallest subset of key reasons that is still 0

baseline value. This approach makes it possible to identify

sufficient to justify the agent’s decision. From RDX, we

which state factors have the greatest influence on the DQN

consider two sets of reward components:

agent’s preference for a particular action, thereby

(𝑘)

𝑃 = {𝛥 (𝑠, 𝑎 1, 𝑎 2) > 0},

providing an intuiti ve explanation at the input feature

(𝑘)

𝑁 = 𝑘 ∣ 𝛥 (𝑠, 𝑎 1, 𝑎 2) < 0 level.

where 𝑃 and 𝑁 respectively represent the reasons 3.3.2. SHAP for A2C_mod supporting and opposing the selected action 𝑎 1. The total

For the agent A2C_mod, SHAP is used to explain the

disadvantage of action 𝑎 1 relative to 𝑎 2 is determined by:

output of the actor or critic network, including the action policy 𝜋 (𝑠 | 𝑎) or the state value function 𝑉 (𝑠). Similarly

(𝑘)

𝑑 = ∑ | 𝛥 (𝑠, 𝑎 1, 𝑎 2) |

to DQN, the model output is represented as the sum of

𝑘 ∈ 𝑁 feature contributions:

+ 𝑀𝑆 𝑋 is defined as the smallest subset 𝑀 ⊆ 𝑃 such 𝑁 that the total advantage of the components in 𝑀 exceeds 𝑓 (𝑠) ≈ 𝜙

0 + ∑ 𝜙 𝑖,

the entire disadvantage 𝑑: 𝑖 = 1

- (𝑘) where 𝑓 (𝑠) corresponds to the probability of action or the

𝑀𝑆 𝑋 = argarg s.t. ∑ 𝛥 (𝑠, 𝑎 1, 𝑎 2) > 𝑑

estimate of the value of the state. Applying SHAP to

𝑘 ∈ 𝑀

A2C_mod helps clarify which state features govern the probability distri bution of action or the estimation of values, thus supporting the analysis of the behavior of the

+

The 𝑀𝑆 𝑋 represents the main reasons that lead the actor critic agent at the level of observable features.

agent to prefer action 𝑎 1.

3. 3.3. SHAP Implementation Details

(𝑘) (𝑘)

𝑣 = ∑ (𝛥 (𝑠, 𝑎 1, 𝑎 2) − 𝛥 (𝑠, 𝑎 1 ′, 𝑎 2)) SHAP is implemented using KernelSHAP with 𝑘 ∈ 𝑀𝑆 𝑋 + shap.KernelExplainer, which trea ts the trained DRL agents − as black - box models. This model - agnostic setting allows the 𝑀𝑆 𝑋 is the smallest subset 𝑀 ⊆ 𝑁 such that:

same attribution procedure to be applied to both DQN and

− (𝑘) 𝑀𝑆 𝑋 = 𝑎𝑟𝑔𝑎𝑟𝑔 s.t. ∑ − 𝛥 (𝑠, 𝑎 1, 𝑎 2) > 𝑣 A2C_mod, despite their different learning architectures. For

𝑘 ∈ 𝑀 DQN, SHAP values are computed with respect to the

− ∗ ∗ 𝑀𝑆 𝑋 indicates which disadvantages make the se lected - action Q - value 𝑄 (𝑠, 𝑎), where 𝑎 =

+

reasons in 𝑀𝑆 𝑋 necessary. MSX provides a compact arg max 𝑎 𝑄 (𝑠, 𝑎). For A2C_mod, SHAP values are justification set for the agent’s decision by identifying the computed with respect to the selected - action policy output

∗

key reward components that support the selected action. In 𝜋 (𝑎 ∣ 𝑠). The background distribution is constructed from this study, the faithfulness of MSX is further evaluated representative normalized inventory states. Inventory and through reward - component masking experiments, which demand - related variables are sampled within their test whether removing MSX - identified components operational ranges, while waste - related variables are changes the selected action or affects the estimated value generated conditionally on inventory to pres erve the main of the decision. dependency structure observed in inventory systems. A total

of 200 background states are generated and then subsampled

3. 2.4. Reward Coefficient Justification

The composite reward integrates four operational objectives. For each product, the reward is defined as a sum of a service term, a holding term, a waste term and a balance term, with a constant base reward that normalizes the overall scale. The weights are set to unity as a neutral baseline, while the waste rate is calibrated to a perishable decay rate observed in the data.

Table 3. Reward components and their justification.

| Component | Weight | Theoretical basis | Managerial interpretation |
| :--- | :---: | :--- | :--- |
| Service (stockout) | 1.0 | Newsvendor model [22] | Stockout cost substantially exceeds holding cost; strong service incentive |
| Holding (overstock) | 1.0 | Economic order quantity [23] | Holding cost of a few percent per period |
| Waste | 0.025 | Perishable inventory [24] | Spoilage rate of a few percent per period, calibrated from demand statistics |
| Balance (quantile spread) | 1.0 | Risk-averse control [25] | Inventory balancing across the portfolio |

The dataset is a public retail transaction dataset that does not contain explicit monetary cost annotations. Capacity is therefore defined as a multiple of average demand and the waste rate is set within the range reported for perishable goods. Sensitivity to the weighting coefficients is assessed with a one-at-a-time analysis on held-out data, varying each coefficient around the baseline. The base-stock policy consistently outperforms the deep reinforcement learning agents, while the two learning agents exhibit complementary sensitivities: overstock is the most influential factor for the value-based agent, whereas balance is the most influential for the actor-critic agent. Stockout exhibits negligible sensitivity and waste shows moderate sensitivity. The unitary weight lies in the interior of the tested range and yields neither the minimal nor the maximal performance, supporting its choice as a balanced baseline.

3.3. Feature-based Explanation with SHAP

This study employs SHAP to analyze the influence of state features on the agent's policy and value estimates, thereby linking observable inventory conditions to decision behavior. All models are treated as black-box functions, allowing the same attribution principle to be applied to both the value-based and the actor-critic agents. SHAP values are interpreted as associative feature attributions that quantify the contribution of each input feature to the model's output for a given state, rather than as causal effects, since inventory, demand, waste and capacity related variables may be correlated.

#### 3.3.1 SHAP Background Construction

The background distribution is designed to reflect operational conditions while ensuring full reproducibility. For the macro-level analysis, each state is represented by three aggregated features capturing inventory level, demand and waste, where waste is modeled as a linear function of inventory with controlled random variation and bounded within an admissible interval. An initial set of 200 synthetic states is generated to balance diversity of the state space with the computational cost of KernelSHAP. From this set, 100 representative states are extracted for explanation. The primary baseline is obtained through controlled random sampling with a fixed seed, ensuring exact reproducibility. A complementary set of 100 centroids obtained via KMeans clustering is constructed as an ablation to maximize spatial coverage. Maintaining both extraction strategies allows testing whether the random baseline is already sufficiently representative.

Table 2 summarizes the descriptive statistics of the background sets. The two 100-state subsets exhibit closely aligned means and dispersions, and their pairwise distance distributions are nearly identical, providing quantitative evidence for their equivalent diversity.

Table 2a. Descriptive statistics of the synthetic background and its representative subsets.

| Dataset | Size | Mean inventory | Std inventory | Mean demand | Mean waste |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Full set | 200 | 0.49 | 0.28 | 0.50 | 0.012 |
| Random sampling | 100 | 0.47 | 0.28 | 0.50 | 0.011 |
| KMeans centroids | 100 | 0.50 | 0.29 | 0.47 | 0.013 |

[Figure - T39_coverage.png]
*Figure 2a. Spatial distribution of the background in the inventory-demand plane. The full set of 200 states is shown in light tone, with the 100 random samples and the 100 KMeans centroids distinguished by different markers. Both 100-state sets cover the domain evenly, supporting the use of the random baseline.*

[Figure - T39_pairwise_dist.png]
*Figure 2b. Distribution of pairwise Euclidean distances within each 100-state subset. The two distributions nearly overlap, indicating equivalent diversity between the two extraction strategies.*

#### 3.3.2 Background Sensitivity

The stability of SHAP is evaluated through a systematic experimental grid covering three dimensions of variation. The first dimension is the origin of the background, contrasting a synthetic background drawn from a uniform distribution with a trajectory background extracted directly from the first steps of the test time series, normalized by product capacity and combined with initial inventory and waste rate to reflect the actual operational trajectory. The second dimension is the background size at three levels, with 100 as the baseline. The third dimension is the sampling strategy, comprising random sampling and KMeans-based centroids. The grid is implemented independently for both agents on the trained models and for both the macro three-dimensional case and the micro 660-dimensional case. For the macro case, KernelSHAP is adopted for its exactness in low dimensions, while for the micro case, Partition Explainer is chosen for its scalability through hierarchical clustering. All evaluations are performed on the same fixed set of test states from the medium scenario to ensure a fair comparison.

In the macro case, SHAP exhibits high stability with respect to size and sampling strategy within the same origin, but shows a systematic shift when the origin changes from synthetic to trajectory. Within synthetic backgrounds of any size, the rank correlation remains at the maximum, indicating no size effect. In contrast, any synthetic versus trajectory comparison yields a stable correlation below the maximum across all sizes, corresponding to a single rank swap among the three features. Feature coverage at a fixed threshold further reveals threshold sensitivity, with synthetic backgrounds yielding higher coverage than trajectory backgrounds despite high rank preservation.

Table 2b. Sensitivity of macro-level SHAP with respect to background origin and size.

| Background origin | Size | Rank correlation vs. baseline |
| :--- | :---: | :---: |
| Synthetic, random | 50, 100, 200 | 1.00 at all sizes |
| Synthetic, KMeans | 50, 100, 200 | 1.00 at all sizes |
| Trajectory, random | 50, 100, 200 | 0.87 at all sizes |
| Trajectory, KMeans | 50, 100, 200 | 0.87 at all sizes |

[Figure - T40_macro_sensitivity.png]
*Figure 3a. Rank correlation of macro-level SHAP versus background size. Synthetic backgrounds remain at the maximum across all sizes, while trajectory backgrounds remain stably below, demonstrating size insensitivity but a systematic origin shift.*

[Figure - T40_fcs_sensitivity.png]
*Figure 3b. Feature coverage at the macro level across strategies and sizes. Synthetic backgrounds yield higher coverage than trajectory backgrounds at the same threshold, illustrating sensitivity of this metric to SHAP magnitude.*

In the micro case with 660 features, the results present a contrasting pattern. Although the overall magnitude remains highly similar, the ranking of features is unstable with respect to any variation of the background. The average rank correlation remains near zero across all origins, sizes and sampling strategies, despite high cosine similarity and minimal mean absolute differences. The full micro grid with 50 test states requires over four hours of computation, reflecting the substantial cost of high-dimensional explanation.

Table 2c. Sensitivity of micro-level SHAP with respect to background origin and size.

| Background origin | Mean rank correlation | Magnitude similarity |
| :--- | :---: | :---: |
| Synthetic, random | approximately 0 | 0.94 - 0.99 |
| Synthetic, KMeans | approximately 0 | 0.94 - 0.95 |
| Trajectory, random | approximately 0.05 | 0.94 - 0.95 |
| Trajectory, KMeans | approximately 0.04 | 0.94 - 0.95 |

[Figure - T40_micro_sensitivity.png]
*Figure 4. Rank correlation of micro-level SHAP with 660 features versus background size. All curves fluctuate around zero across all sizes and strategies, indicating ranking instability in high dimensions despite preserved magnitude.*

#### 3.3.3 Validity of Perturbed States

The validity of perturbed states generated during SHAP computation is verified against the physical constraints of the inventory problem, including bounds on inventory, demand and waste and the functional relationship between waste and inventory within a tolerance. The verification is performed on several thousand perturbed states per scenario for the macro case and on a thousand median-masked states for the micro case, covering three operational scenarios of increasing difficulty.

Trajectory backgrounds achieve the maximum validity in all scenarios, while synthetic backgrounds achieve high but slightly lower validity that decreases under the most demanding conditions, with the lowest rate observed in the hard scenario. The micro-level median masking achieves full validity. The mean validity at the macro level remains above 96%, providing evidence that perturbed states are generally operationally feasible, while the hard synthetic case serves as a cautionary note.

Table 2d. Domain validity of perturbed states.

| Scenario | Dimensionality | Background | Total states | Validity rate |
| :--- | :---: | :---: | :---: | :---: |
| Easy | 3 | Synthetic | 5000 | 95.5% |
| Easy | 3 | Trajectory | 5000 | 100% |
| Medium | 3 | Synthetic | 5000 | 98.8% |
| Medium | 3 | Trajectory | 5000 | 100% |
| Hard | 3 | Synthetic | 5000 | 84.5% |
| Hard | 3 | Trajectory | 5000 | 100% |
| Medium | 660 | Synthetic, median | 1000 | 100% |

[Figure - T41_validity_rate.png]
*Figure 5a. Domain validity rate of perturbed states. Trajectory backgrounds achieve the maximum in all three scenarios, while synthetic backgrounds achieve high validity in the easy and medium scenarios but a notably lower rate in the hard scenario, indicating higher feasibility of trajectory-based perturbations.*

[Figure - T41_waste_violation.png]
*Figure 5b. Distribution of the waste residual for perturbed states in the medium scenario. The majority of the mass lies within the admissible interval, with only the tails exceeding the tolerance, illustrating the clipping mechanism and the source of violations.*

#### 3.3.4 Consistent SHAP Implementation

Two complementary SHAP explainers are used consistently at different granularities, ensuring both reusability and scalability. The macro system-level analysis employs KernelSHAP on aggregated states, which is exact with a limited number of coalitions and requires only seconds per evaluation. The micro SKU-level analysis employs Partition Explainer on raw high-dimensional states with a partition mask and hierarchical clustering, which scales near-linearly and avoids the exponential explosion, requiring several minutes per evaluation. Both analyses use 100 representative background states drawn from 200 synthetic states with a fixed seed, complemented by trajectory ablations. The two levels are complementary: the macro level indicates which feature group drives the decision and the micro level indicates which specific product within that group contributes most.

Table 2e. Standardized configuration of the two SHAP methods.

| Criterion | Macro level | Micro level |
| :--- | :--- | :--- |
| Objective | Identify the feature group driving the decision | Identify the specific product driving the decision |
| Input | Aggregated three-dimensional state | Raw 660-dimensional state |
| Method | Exact Shapley values with limited coalitions | Hierarchical partitioning with near-linear scalability |
| Background | 100 representatives from 200 synthetic states | 100 representatives from 660-dimensional sets |
| Computation time | Seconds per evaluation | Minutes per evaluation |

[Figure - T42_top20_micro.png]
*Figure 6. Twenty most influential micro-level features in the medium scenario for the DQN agent. Bars are color-coded by feature group, demonstrating SKU-level drivers beyond the macro-level group.*

4. Evaluation

4. 1. Evaluation Schema reward, service level, holding cost, waste cost, ordering

cost, and stockout rate for Base - stock, DQN, and

The evaluation schema in this study is designed to

A2C_mod.

simultaneously assess the decision - making effectiveness of DRL agents and the quality of XRL explanations 4.2. Operational Performance Comparison with (RDX, MSX, SHAP) in the many products inventory

Classical Baseline

management problems. The evaluation process is not only based on traditional operational performance metrics but To evaluate the operational competitiveness of the also considers the transparenc y and interpretability of learned DRL policies, this study compares DQN and agent behavior. The evaluation framework is composed of A2C_mod with the classical Base - Stock policy under the three core elements: the dataset, the training – testing setup, same testing conditions. The comparison is conducted for and the scheme used to assess explanation methods.

the overall test set and three operational scenarios, namely

4. 1.1. Dataset EASY, MEDIUM, and HARD. The results are

summarized in Table 4.

This study directly adopts the DRL agents DQN and A2C_mod that have been trained in the multi - product The results show tha t the Base - Stock policy provides a inventory management environment. These agents are strong classical reference in the proposed inventory trained in the publicly available retail dataset Instacart environment. In terms of total reward, Base - Stock Market Basket Analysis, as presented in Table 3 achieves the highest score in the overall setting

4. 1.2. Experimental Design (+ 0. 8269), outperforming DQN (+ 0. 7197) and

A2C_mod + 0. 6939. This trend is consistent across the

The experimen tal design evaluates, in parallel, the

EASY, MEDIUM, and HARD scenarios, where Base -

performance and explainability of the two agents, DQN

Stock obtains total rewards of + 0. 8978, + 0. 8312, and

and A2C_mod, within the same inventory management

- 0. 7518, respectively. These results indicate that a well -

environment. Agents are trained in a fixed dataset and then generalize in different scenarios. During evaluation, agent calibrated classical replenishment policy remains

competitive under the propo sed reward structure.

Attribute Value

However, the comparison also reveals clear operational

Data source Instacart Market Basket Analysis

trade - offs among the methods. Base - Stock achieves lower holding and waste costs, but it produces lower service Number of customers 60,000 levels and higher stockout rates than the DRL agents. In

Number of products 220

the overall setting, Base - Stock reaches a service level of

- 0. 9363 and a stockout rate of + 0. 0637. In contrast, Simulation period 349 days

DQN and A2C_mod achieve almost perfect service levels

Total number of decision cycles 1,396

- 0. 9999 and + 0. 9997) and substantially lower stockout

Training cycles 900 (∼ 64%)

rates + 0. 0001 and + 0. 0003). This suggests that the DR L agents learn policies that prioritize demand fulfillment and Testing cycles 496 (∼ 36%) stockout prevention, although this comes at the cost of higher holding and waste - related penalties. Across the three scenarios, all methods experience a decline in total reward as the environme nt becomes more difficult. The reduction is particularly clear for the DRL agents in the HARD scenario, where DQN decreases to + 0. 5803 and A2C_mod decreases to + 0. 5462. By comparison, Base - Stock remains more stable, decreasing from + 0. 8978 in EASY to + 0. 7518 in HARD. This suggests that the classical policy is more conservative and stable, while the DRL agents are more sensitive to changes in demand pressure, inventory level, and waste risk.

4. 3. Reward Difference Explanation
4. 3.1. Evaluation scenarios and experimental setup:

Fig. 2. Reward Difference Explanation (RDX) analysis

To evaluate the decision - making behavior of agents in results for DQN and A2C_mod.

different operational contexts, the study designs three

In the EASY scenario, the reward structures of the two

scenarios with increasing levels of complexity and risk. models are almost identical, with the service component Table 5 summarizes the detailed configu rations of each playing the dominant role. Operational costs have only a scenario and their corresponding evaluation objectives. minor impact, indicating that DQN and A2C_mod exhibit The three evaluation scenarios are designed to test the similar decision - making behavior i n a stable environment.

stability and discriminative ability of the XAI pipeline In the MEDIUM scenario, differences begin to emerge as under different operational conditions. They are derived A2C_mod shows a greater sensitivity to hold and ordering from the empir ical characteristics of the Instacart - based costs, while DQN continues to maintain a service prioritized inventory data. The MEDIUM scenario uses the original strategy. This reflects differences in how the two agents

b alance service quality and operational costs as uncertainty

demand distribution as the calibration baseline. The

increases. In the HARD scenario, A2C_mod clearly adjusts

EASY scenario reduces demand intensity and spoilage

its strategy by substantially reducing operational costs, thus

risk to represent more stable products with lower

achieving a positive total reward. In contrast, DQN

ope rational pressure, while the HARD scenario increases

continues to prioritize service and incurs high costs,

both demand intensity and spoilage risk to simulate high -

resulting in a negative total reward. These results indicate

pressure perishable - product conditions. This design

that A2C_mod adapts better in high - risk environments. In

allows the same trained agents to be evaluated under

general, the RDX results show that the two agents behave in

controlled changes in demand and waste le vels, thereby

a similar way in simple scenarios, but A2C_mod

testing whether RDX, MSX, and SHAP explanations

demonstrates greater flexibility as the environmental

remain meaningful when the operating environment complexity increases. RDX analysis helps clarify the becomes more difficult. decision - making mechanisms and business trade - offs of

each model. The comparison with the Base - Stock policy

4. 3. 2. Experimental Results and Analysis:

provides an operational reference for int erpreting the RDX

Figure 2 presents the RDX results for the two agents, results. When a DRL agent achieves better service - level DQN and A2C_mod, in the three evaluation scenarios performance or lower stockout rate than the classical EASY, MEDIUM and HARD by decomposing the total

baseline, the RDX decomposition helps identify which

reward into the components of service, storage, waste and

reward components support this improvement. Conversely,

order.

when the DRL agent obtains a lower total reward than Base - Stock, RDX indicates whether this result is caused by higher

Table 3. Summary of the experimental dataset.

holding, waste, or ordering - related penalties. Therefore, decision making be havior with limited contextual RDX not only explains the differences between DQN and adaptability.

A2C_mod, but also clarifies how the learned DRL policies differ from a conventional replenishment strategy.

Figure 3 compares the distribution of Q values for all actions between the two agents, A2C_mod and DQN, in the three scenarios EASY, MEDIUM and HARD. For A2C_mod, since the algorithm is based on a policy value mechanism, the Q values are transformed into an equivalent form to ensure direct comparability with DQN.

In the EASY scenario, A2C_mod relatively clearly distinguishes between the selected action and the remaining actions, while DQN exhibits high and uniform Q values between actions, indicating that the differentiation of actions is not pronounced in a simple environment. In the MEDIUM and HARD scenarios, Fig. 3. Comparison of Q - value distributions of A2C_mod A2C_mod shows increasingly clear separation, with the and DQN across different levels of environmental

difficulty.

selected action having a markedly higher 𝑄 value, while DQN continues to maintain relatively uniform 𝑄 values. Figure 4 presents the RDX results by comparing the This reflects a stable but less flexible decision - making selected action with alternative actions in individual

reward components, including service, holding, waste, and

strategy as environmental complexity increases. These

order. The values in the chart represent the reward

results indicate that A2C_mod more clearly differentiates

differences between the selected action and other options,

between effective and less effective actions in challenging

thus indicating the relative advantages or disadvantages of

environments, whereas DQN exhibits threshold like

the current c hoice for each component.

Table 4. Performance comparison with the classical Base - Stock baseline.

Testing scenario Evaluation metric Base - Stock DQN A2C_mod

Total Reward ↑ +0.8269 +0.7197 ± 0.0396 +0.6939 ± 0.1463

Service Level ↑ +0.9363 +0.9999 ± 0.0001 +0.9997 ± 0.0007

Holding Cost ↓ +0.0000 +0.1110 ± 0.0630 +0.0935 ± 0.1495

All scenarios

Waste Cost ↓ +0.0077 +0.0672 ± 0.0008 +0.0672 ± 0.0008

Ordering Cost ↓ +0.1017 +0.1020 ± 0.0258 +0.1478 ± 0.0301

Stockout Rate ↓ +0.0637 +0.0001 ± 0.0001 +0.0003 ± 0.0007

Total Reward ↑ +0.8978 +0.8304 ± 0.0533 +0.8194 ± 0.1546

Service Level ↑ +0.9356 +1.0000 +1.0000 ± 0.0001

Holding Cost ↓ +0.0000 +0.1145 ± 0.0718 +0.1010 ± 0.1579

EASY scenario

(𝑥 = 30%, 𝑠𝑎𝑙𝑒𝑠 = 20%, 𝑤𝑎𝑠𝑡𝑒 = 1. 0%)

Waste Cost ↓ +0.0003 +0.0098 ± 0.0001 +0.0098 ± 0.0002

Ordering Cost ↓ +0.0375 +0.0453 ± 0.0258 +0.0698 ± 0.0424

Stockout Rate ↓ +0.0644 +0.0000 +0.0000 ± 0.0001

Total Reward ↑ +0.8312 +0.7484 ± 0.0498 +0.7160 ± 0.1557

Service Level ↑ +0.9365 +0.9999 ± 0.0001 +0.9996 ± 0.0008

MEDIUM scenario (x=60%,sales=50%,waste=5.0%)

Holding Cost ↓ +0.0000 +0.1111 ± 0.0665 +0.0964 ± 0.1553

Waste Cost ↓ +0.0039 +0.0484 ± 0.0007 +0.0475 ± 0.0010

Testing scenario Evaluation metric Base - Stock DQN A2C_mod

Ordering Cost ↓ +0.1014 +0.0921 ± 0.0183 +0.1396 ± 0.0359

Stockout Rate ↓ +0.0635 +0.0001 ± 0.0001 +0.0004 ± 0.0008

Total Reward ↑ +0.7518 +0.5803 ± 0.0197 +0.5462 ± 0.1303

Service Level ↑ +0.9369 +0.9997 ± 0.0002 +0.9994 ± 0.0011

Holding Cost ↓ +0.0000 +0.1076 ± 0.0534 +0.0831 ± 0.1354

HARD scenario

(𝑥 = 90%, 𝑠𝑎𝑙𝑒𝑠 = 80%, 𝑤𝑎𝑠𝑡𝑒 = 15. 0%)

Waste Cost ↓ +0.0188 +0.1432 ± 0.0018 +0.1363 ± 0.0040

Ordering Cost ↓ +0.1663 +0.1686 ± 0.0392 +0.2339 ± 0.0180

Stockout Rate ↓ +0.0631 +0.0003 ± 0.0002 +0.0006 ± 0.0011

Table 5. Configuration of experimental scenarios for RDX.

Attribute Value Characteristics Easy Scenario Inventory 30%, Demand 20%, Spoilage rate 1% This is an ideal situation to order aggressively to meet demand without significant

risk of excessive inventory.

Medium Scenario Inventory 60%, Demand 50%, Spoilage rate 5% This is a balanced situation that requires the agent to carefully trade - off between

ordering in anticipation of demand and the risk of holding inventory.

Hard Scenario Inventory 90%, Demand 80%, Spoilage rate 15% This is a critical situation. Although demand is high, inventory is already near

capacity, and the spoilage rate is high. The agent faces a major conflict: ordering more increases holding and disposal costs, while not ordering leads to missed sales.

conditions.

4. 4. Minimal Sufficie nt Explanation

MSX is constructed based on RDX to identify the minimal set of core reward components that is still sufficient to justify the agent’s decision. While RDX provides a detailed analysis of differences between actions between individual reward components, MSX focuses on clarifying the indispensable factors in the decision - making rationale. In this study, MSX is applied to both DQN and A2C_mod in the EASY, MEDIUM and HARD scenarios to analyze how the structure of the agents’ reasoning changes as environmental co mplexity increases.

Fig. 4. The heatmap shows the reward component differences in RDX.

Positive difference values indicate that the selected action yields higher benefits than alternative actions for the corresponding component, while negative values indicate that there are still more advantageous options. The results show that in the EASY and MEDIUM scenarios, both models still exhibit some competing actions that are more favorable in terms of cost, reflecting a decision space that is not yet c learly differentiated. In contrast, in the HARD scenario, A2C_mod shows a clear advantage, with the Fig. 5. MSX evaluation based on the influence level of selected action showing positive differences between most each reward component.

components, while DQN still has alternative actions that

The results shown in Figure 5 indicate that, in all three

are more favorable in terms of holding an d ordering costs.

scenarios, sequentially removing each individual reward

Overall, the RDX plots provide an intuitive view of the

component does not change the agent’s deci sion for either

rationality of the selected action compared to other

model. This observation suggests that the decision does

options, while clarifying how the models balance the

not rely on a single factor but is instead formed through a

reward components under different environmental

stable combination of multiple reward components. Even The total reward after adjustment becomes:

in the HARD scenario, where risk levels and objective conflicts are h igher, no component, when removed, can (λ, 𝑗) (𝑘) (𝑗)

𝑟 𝑡 = ∑ 𝑤 𝑘 𝑟 𝑡 + (λ 𝑤 𝑗) 𝑟 𝑡

reverse the decision. Therefore, there exists no single

𝑘 ≠ 𝑗

reward component that is minimal and sufficient to explain the decision, thus confirming the multi factor Since RDX reflects the contribution of each component nature of the decision - making process in the inventory to the Q - value, when the weight changes, the contribution of management problem. component 𝑗 changes proportionally with 𝜆:

However, when MSX is examined from the (𝑗)

(𝑗)

Δ 𝑄 = λ ⋅ Δ 𝑄

perspective of the minimal subset of reward components, 𝑤𝑒𝑖𝑔 ℎ 𝑡 the models exhibit markedly different reasoning

This procedure is performed sequentially for each

structures, as shown in Figure 8. For A2C_mod, in the

component an d for all three scenarios EASY, MEDIUM, and

EASY and MEDIUM scenarios, the MSX set includes

HARD presented in Table 6.

only the service component, indicating that demand fulfillment serves as the core factor in low and medium risk environments. When moving to the HARD scenario, the MSX set shifts to the holding component, reflecting an adjustment of the model’s core reasoning factor under increased inventory cost pressure. In contrast, for DQN, the MSX set in all three scenarios consistently includes only the holding component, indicating a stable minimal explanation structure that is largely independent of environmental context. These results clarify the difference in decision making flexibility between the two models, with A2C_mod demonstrating better context de pendent adaptability than DQN.

4. 4 Sensiti vity Analysis

Fig 6. Sensitivity analysis of the average RDX with respect

To evaluate the degree of dependence of reward - based

to the coefficient λ in the EASY, MEDIUM, and HARD

explanations (RDX and MSX) on weight design, the study

scenarios for DQN and A2C_mod.

conducts a sensitivity analysis on the weighting coefficients of each reward component. Below, the sensitivity analysis

Figure 6 illustrates the variation of the average RDX

framework use d to examine the stability of explanations

value with respect to the adjustment coefficient λ across

when changing the priority levels among operational

three scenarios (EASY, MEDIUM, HARD) for the two

objectives is presented.

architectures DQN and A2C_mod. Overall, RDX increases

The total reward at time t is determined by the following

with λ, indicating that the explanation structure depends on

formula: the amplification level of the reward weights. In all

𝑠𝑟𝑣 ℎ 𝑜𝑙𝑑 𝑤𝑎𝑠𝑡𝑒

𝑟 𝑡 = 𝑤 𝑠𝑟𝑣 𝑟 𝑡 + 𝑤 ℎ 𝑜𝑙𝑑 𝑟 𝑡 + 𝑤 𝑤𝑎𝑠𝑡𝑒 𝑟 scenarios, the ordering cost component (w - order) exhibits 𝑡

𝑜𝑟𝑑𝑒𝑟

- 𝑤 𝑜𝑟𝑑𝑒𝑟 𝑟 the highest sensitivity, with a steep slope as λ increases, 𝑡

Baseline setup: while the holding cost (w - hold) increases at a lower rate. In

contrast, the service (w - ser) and waste (w - waste)

𝑤 𝑠𝑟𝑣 = 𝑤 ℎ 𝑜𝑙𝑑 = 𝑤 𝑤𝑎𝑠𝑡𝑒 = 𝑤 𝑜𝑟𝑑𝑒𝑟 = 1. 0 components maintain small RDX values with minimal

fluctuation, reflecting relative stability underweight

The sensitivity analysis is conducted using the One - At - a - changes. A comparison between the two architectures Time (OAT) method, in which only one weight is adjusted at shows that DQN exhibits significantly larger variation each step while the remaining wei ghts are kept unchanged. amplitudes, particularly in the ME DIUM and HARD Specifically, for a component 𝑤 𝑗 ∈ scenarios, where RDX increases sharply as λ increases.

Meanwhile, A2C_mod maintains lower and more stable

𝑤 𝑠𝑟𝑣, 𝑤 ℎ 𝑜𝑙𝑑, 𝑤 𝑤𝑎𝑠𝑡𝑒, 𝑤 𝑜𝑟𝑑𝑒𝑟, the weight is adjusted as

RDX values, indicating that its explanations are less

follows:

sensitive to reward weight adjustments. As environmental

(λ) complexity increases, the variation level of RDX also 𝑤 = λ ⋅ 𝑤 𝑗 𝑗 increases, most noticeably for DQN. These results suggest

that the RDX explanation structure is primarily influenced

W here:

by the ordering cost component, and that the stability of explanations depends considerably on the reinforceme nt

λ ∈ 0. 5, 1. 0, 1. 5, 2. 0

learning architecture employed. In addition to analyzing the variation magnitude of RDX values, the study further

examines changes in the MSX structure as λ varies. This maintains consistency w hen the priority levels among analysis aims to assess whether the minimal explanation set operational objectives are adjusted.

Table 6. Sensitivity analysis setup by scenario and architecture.

Scenario Fixed settings Algorithm Varied reward weight(s) Weight factors tested (λ)

A2C_mod

EASY Inv 30%, Dem 20%, Spoil 1%

DQN

A2C_mod

MEDIUM Inv 60%, Dem 50%, Spoil 5% (𝑤 𝑠𝑟𝑣, 𝑤 ℎ 𝑜𝑙𝑑, 𝑤 𝑤𝑎𝑠𝑡𝑒, 𝑤 𝑜𝑟𝑑𝑒𝑟) {0.5, 1.0, 1.5, 2.0}

DQN

Inv 90%, Dem 80%, Spoil A2C_mod

HARD

15%

DQN

decreases from EASY to HARD, while w_hold increases correspondingly. However, within each individual scenario, the MSX structu re remains relatively stable with respect to λ.

In contrast, A2C_mod shows a clearer context dependence:

in the EASY scenario, the MSX structure varies with λ, whereas in the HARD scenario, the component frequencies remain nearly unchanged, reflecting high stability under weight adjustments. The MSX results complement the RDX analysis by demonstrating that explanation magnitude and explanation structure are two distinct aspects. While RDX measures the intensity of contribution changes, MSX reflects the stab ility of the core reasoning set, thereby clarifying behavioral differences between DQN and A2C_mod under different environmental conditions.

4. 5. SHapley Additive exPlanations

To address aggregation bias while maintaining strategic interpretability, this study adopts a two-level SHAP framework. Table 7a defines the interpretation scope.

Table 7a. SHAP Interpretation Scope Matrix

| SHAP Level | Input Features | Purpose | Interpretation Scope |
| :--- | :--- | :--- | :--- |
| Aggregated SHAP | 3 averaged features: Inventory (avg 220 SKUs), Demand/Sales (avg), Waste (avg) | Strategic overview for managers | System-level drivers |
| Top-k Micro SHAP | 660 raw features: inventory_SKU0-219, sales_SKU0-219, waste_feat_SKU0-219 | Identify specific SKU and system drivers | SKU-level and system-level drivers |

The aggregated 3-feature view provides a strategic cognitive framework for quickly identifying overall behavioral trends, while the micro-level analysis on the 660-dimensional raw feature space addresses heterogeneity. This structure maintains compatibility with prior FCS/ablation analyses while the micro-level results are treated as the primary SHAP findings.

*Note on comparability (Task 14): SHAP for DQN is computed on softmax(Q) (ckpt-60) and for A2C_mod on policy π(a*|s) (ckpt-64) - two quantities on different scales (Q unbounded vs π constrained [0,1] and interdependent via softmax). Hence, direct magnitude and stability comparisons between agents in Table 7b/7c are descriptive and qualitative, not quantitative. Sensitivity analysis on common-target logits (pre-softmax, 10/20/50 states, PartitionExplainer, Supplementary Task 13) shows Jaccard only 0.25-0.33 homogeneous across n_states and between logits vs softmax/π (e.g., DQN EASY 10 Top-5 175 0.00178 vs A2C 93 0.460, 0/5 overlap), supporting the need for qualitative comparison.*

4. 5.1. Problem setup:

The original state space consists of 664 SKU - level features (660 product-level + 4 system-level: Ut, Ct, Vt, Tt). However, applying SHAP directly to such a high - dimensional space with strong correlations among variables may result in dispersed and difficult - to - interpret explanation values. Therefore, the study adopts a purposive dimensionality reduction stra tegy, aggregating the features

Fig 7. Frequency of occurrence of reward components in

into three representative system - level variables: Inventory

the MSX set with respect to the coefficient λ across the

(I), Demand (D), and Waste (W). This aggregation aligns

EASY, MEDIUM, and HARD scenarios for DQN and

with the reward function structure and helps improve the

A2C_mod.

signal - to - noise ratio, while providing clearer manag erial implications at the system level However, this approach

Figure 7 presents the frequency of occurrence of reward

may reduce the ability to capture heterogeneity across

components in the MSX set with respect t o the coefficient λ

SKUs and may overlook cross - product interactions.

across three scenarios and two architectures. The results show

Therefore, the explanation results should be interpreted at

that the ordering cost (w_order) dominates in most

the system level, acknowledgi ng the trade - off between

configurations, with frequencies ranging from approximately

interpretability and granularity.

50% to nearly 100%, particularly high in the EASY scenario of DQN. The holding cost (w_hold) plays a secondary role, becoming more pronounced in the MEDIUM and HARD scenarios of DQN (around 30 – 40%), while the service (w_ser) and waste (w_waste) components appear at relatively low frequencies. From an architectural pe rspective, DQN exhibits a redistribution of components as environmental complexity increases: the proportion of w_order gradually

Fig. 8. Qualitative – quantitative summary chart of the

MSX results.

4. 5.2. Experimental Results and Analysis:

For A2C_mod (Figure 9), the SHAP values of the

Fig. 10. Local SHAP: A2C_mod vs DQN.

three features Inventory, Demand, and Waste are all concentrated very close to zero with low dispersion. This The local SHAP results (Figure 10) reveal clear indicates that the contribution of each feature to the model differences in state dependent responses between output is small and relatively balanced, reflecting a A2C_mod and DQN. Local SHAP analysis shows distinct decis ion - making mechanism that is not strongly differences in how A2C_mod and DQN react to system dominated by any single factor at the global level. In states. In the High Inventory and Low Inventory scenario s, contrast, DQN exhibits a more pronounced separation in DQN is strongly influenced by the Waste component, the SHAP distributions, particularly for Inventory and while A2C_mod exhibits very small SHAP values close to Demand, with larger dispersion and a clear influe nce on zero, indicating that its decisions are not dominated by any output. Meanwhile, Waste shows a very small single factor. In the High Demand and Low Demand contribution concentrated around zero. These results scenarios, DQN continues to show pron ounced sensitivity indicate that DQN’s global decision making is primarily to Demand and Waste, reflecting a decision - making driven by inventory and demand. behavior that reacts to market fluctuations. In contrast,

A2C_mod maintains low and dispersed SHAP

Comparing the two models, the global SHAP analysis

contributions, indicating a more stable decision - making

shows that A 2C_mod has a more distributed and stable

mechanism. Notably, in the Criti cal State scenario, DQN

contribution structure, while DQN relies heavily on a few

exhibits SHAP values with large magnitudes, while

dominant features, highlighting differences in the overall

A2C_mod still maintains values close to zero,

decision - making mechanisms between the two

demonstrating its ability to sustain stable decisions even

approaches.

under extreme conditions. Overall, the local SHAP analysis indicates that A2C _mod is more state stable, whereas DQN relies heavily on certain features, especially in high - risk scenarios.

4. 5.3. Top - k Micro - Level SHAP Analysis (Primary SHAP Result)

While Section 4.5.2 provides an aggregated overview, this section presents the primary SHAP analysis on the 660-dimensional product-level state space. Table 7b compares macro-averaged vs micro-level attributions (50 states/scenario, PartitionExplainer, background 100). At the macro level, Sales/Demand dominates for both agents, but micro-level reveals extreme concentration on 5-8 specific SKUs rather than uniform distribution across 220 SKUs, evidencing aggregation bias.

Table 7b. Comparison of Macro-averaged vs Micro-level SHAP (Mean|SHAP|; values in Supplementary)

| Agent | Scenario | Macro Dominant | Micro Top-5 (SKU ID) |
| :--- | :--- | :--- | :--- |
| DQN | EASY | sales | SKU64, 163, 100, 46, 155 |
| DQN | MEDIUM | sales | SKU64, 163, 46, 155, 118 |
| DQN | HARD | sales | SKU64, 163, 46, 155, 118 |
| A2C_mod | EASY | sales | SKU163, 155, 64, 46, 118 |
| A2C_mod | MEDIUM | sales | SKU163, 155, 64, 46, 118 |
| A2C_mod | HARD | sales | SKU64, 155, 163, 46, 118 |

*Mean|SHAP| values: DQN 0.00281-0.00253, A2C_mod 0.00044-0.00027; 3-4x tied baseline 0.00079. Figure 11 (Top-20, 660 features) is primary; Figure 9 (3 features) is overview. *Comparison qualitative due to different targets (softmax Q vs π); see Task 13 common-target analysis in Supplementary. Supplementary ablation comparing the partition-based explainer with a kernel-based alternative on a 10-state subset shows markedly higher granularity for the kernel approach and negligible top-20 overlap, supporting the limitation note that partition clustering may underestimate granularity while sales-group dominance remains.*

4. 5.4. Consistency of Top-k Features across EASY/MEDIUM/HARD

For Top-20, DQN shows Jaccard 1.00 and Spearman 1.00 across all scenario pairs (stable, 8 SKUs), while A2C_mod shows Jaccard 0.29-0.33 and Spearman 0.20 (context-dependent shift, core Top-5 remains). Full k=10/20/50 results (18 rows) in Supplementary task10-9/outputTask10-11_overlap.csv. *Comparison qualitative as noted in Section 3.3.3.*

4. 5.5. Why Dominant Products Dominate Top-20 - Case Analysis

Top-demand SKUs are SKU57 (22.6), SKU81 (17.2), SKU108 (15.5) - not in Top-8 SHAP (r=0.04). Dominant SKUs are low-volume high-CV (1.1-1.46) and small capacity (7-28), e.g., SKU64 CV 1.46 cap 7, SKU163 CV 1.27 cap 12 - volatile, stockout-sensitive. SKU215 is exception - high-volume (8.68) with large capacity 81. Full 36 SKUs in Supplementary task10-9/outputTask10-12_case_analysis.csv. SHAP is associative.

Figure 9 (3 features) is overview; Figure 11 (660 features) is primary. I_Micro(f_j)=1/(N·A) Σ|φ_{j,a}^{(n)}|.

4.5.6. Common Target Sensitivity (Task 13 - Logits, 10/20/50 states)

To test comparability, SHAP was computed on pre-softmax logits as common target (DQN q_values linear ckpt-60 and A2C_mod layer4 before softmax ckpt-64) with sensitivity across n_states=10/20/50 (100 background, PartitionExplainer, `output Training`). We chose n_states=10/20/50 to test sampling robustness: 10 states is a fast feasibility test (~25 min for 5 configs), while 20 and 50 increase statistical power but cost 2-5 hours for 90 explainers. If Jaccard between n=10 vs 50 remained high (>0.8), 10-state would be representative; our results show Jaccard only 0.25-0.33 homogeneous across scenarios (e.g., DQN EASY 10-20 0.250/RBO 0.767, HARD 10-20 0.333/RBO 0.769; A2C_mod EASY 10-20 0.250/RBO 0.633, HARD 10-20 0.250/RBO 0.537; full 18 rows in task11-9/outputTask11_n_states_sensitivity.csv), indicating Top-20 is sensitive to sample size and 10-state alone is not representative, hence reporting all three levels. Logits Top-8 remains Sales-dominant (SKU175,90,164,119,157,93,108,71; e.g., DQN EASY 10-state 0.00178 vs A2C 93 0.460) but ranking differs from softmax(Q)/π baseline (0/5 overlap), supporting qualitative comparison in Table 7b/7c. Full 360-row Top-20 logits in task11-9/outputTask11_common_target.csv.

4.5.7. Actor vs Critic Sensitivity (Task 15 - 10-state EASY/MEDIUM/HARD)

Sensitivity on A2C_mod with three outputs (π ckpt-64, logits, V(s)) on 10-state EASY/MEDIUM/HARD shows Jaccard Top-20 0.25-0.33 between π vs logits and π vs V(s) across all scenarios (e.g., DQN EASY softmax vs logits Jaccard 0.25/RBO 0.736, A2C EASY pi vs logits 0.25/RBO 0.420; A2C MEDIUM pi vs logits 0.333/RBO 0.509; EASY A2C pi Top-5 [175,90,164,119,93] vs logits [93,119,108,175,71] overlap 3/8, Jaccard 0.25; EASY pi vs V(s) Top-5 [175,90,164,119,93] vs V(s) [175,71,90,164,119] Jaccard 0.25/RBO 0.647; full 9 rows in task11-9/outputTask11_sensitivity.csv with homogeneous 0.25-0.33, indicating ranking differs across targets despite same Sales group, supporting qualitative comparison.

4.5.8. Reward Component Range and Distribution

![Sensitivity of average reward to weighting coefficients](Feedback 7-9/task13-9/task1/outputTask20_sensitivity.png)
*Figure: One-at-a-time sensitivity of average reward to the four weighting coefficients for the two learning agents and the base-stock policy.*

The analysis presented here uses the original transaction records covering 1,000 training periods and 504 testing periods for 220 products. Demand is normalized by product capacity, where capacity is defined as a multiple of average demand observed during training. Reward components are computed per product and per period from the normalized state using the same transition and reward definition employed during training, and descriptive statistics as well as distributional distances between training and testing periods are compared before and after normalization.

Before normalization, demand is right-skewed with a long tail, while after normalization the scale is an order of magnitude smaller and concentrated in a narrow interval. Capacity is heterogeneous but consistently an order of magnitude larger than average demand.

Table 4. Descriptive range of demand and capacity.

| Variable | Before normalization (raw units) | After normalization (demand / capacity) |
| :--- | :--- | :--- |
| Demand, training | mean 2.11, SD 4.62, max 162 | mean 0.10, SD 0.14, max 2.25 |
| Demand, testing | mean 0.73, SD 1.75, max 31 | mean 0.034, SD 0.069, max 1.25 |
| Capacity | mean 20.3, SD 28.2, range 4–208 | — |

Normalization reduces the mean and dispersion by more than an order of magnitude and brings the maximum from over a hundred units to a few units, which stabilizes the reward scale. The ratio of capacity to mean demand is approximately ten, confirming that capacity is defined as a sizable multiple of average demand.

When evaluated under different policies, a no-replenishment policy leads to frequent stockout, near-zero inventory and negligible waste, yielding a near-zero average reward. In contrast, a base-stock policy calibrated on training demand substantially reduces stockout, increases inventory by an order of magnitude and raises waste and balance terms, yielding a markedly higher average reward. The aggressive value-based agent incurs a noticeable overstock cost, whereas the base-stock and actor-critic policies incur negligible overstock.

Table 5. Distribution of reward components under different policies (mean over products and periods).

| Component | No replenishment (training / testing) | Base-stock policy (training / testing) | Value-based agent (testing) |
| :--- | :--- | :--- | :--- |
| Stockout rate | 0.99 / 0.97 | 0.15 / 0.022 | 0.007 |
| Overstock | 0.0 / 0.0 | 0.0 / 0.0 | 0.46 |
| Waste (q) | 7.0×10⁻⁵ / 2.2×10⁻⁴ | 0.0038 / 0.0053 | 0.024 |
| Balance spread | 0.007 / 0.026 | 0.23 / 0.21 | 0.14 |
| Average inventory | 0.003 / 0.009 | 0.15 / 0.21 | — |
| Average reward | 8×10⁻⁶ / –0.002 | 0.61 / 0.76 | 0.36 |

The table shows that a replenishment policy is necessary to move from a degenerate regime with almost certain stockout to a regime with controlled stockout and non-negligible inventory. Overstock remains negligible under the base-stock policy but becomes sizable under the aggressive value-based policy, demonstrating that overstock is policy-dependent rather than data-dependent. The increase in waste and balance spread under replenishment is proportional to inventory, consistent with a waste rate of a few percent per period.

Taken together, the results demonstrate three points. First, normalization is effective: the distributional distance between training and testing periods decreases from 1.37 for raw demand to 0.066 for normalized demand, a reduction of about 95%, indicating that capacity-based normalization substantially mitigates covariate shift without refitting on the test data and therefore avoids leakage. Second, the reported ranges justify the coefficient choices: waste lies in [0, 0.025] and balance spread in [0, 0.89], both within the unit interval after normalization, so no additional scaling is required. Third, the contrast between policies validates the reward design: without replenishment no inventory is retained, while with replenishment a stable operating point with high reward is attained, and the remaining differences in overstock and balance explain the complementary sensitivities observed in the weight-sensitivity analysis.

4.6. Comparative Evaluation
4.6.1 Design

To evaluate the role and contribution level of each explanation mechanism within the proposed XRL framework, the study conducts an ablation-oriented analysis by comparing three different explanation configurations. The objective is to determine whether each individual method is sufficient to provide comprehensive interpretability, or whether their combination is necessary to achieve holistic transparency. Specifically, the three configurations considered are: (i) Reward-only (RDX+MSX), (ii) Feature-based only (SHAP), and (iii) Hybrid framework.

4.6.2 Metrics

To quantitatively evaluate the explanation mechanisms within the proposed XRL framework, the study employs a set of metrics to measure objective coverage, feature utilization, cross-domain consistency, and the stability of the explanation structure under weight adjustments.

Objective Coverage Score (OCS): OCS measures the coverage level of business objec tives in reward - based explanations (RDX/MSX).

Table 7. Scenaros for local SHAP analysis.

Scenario System State Evaluation Purpose

Tests the ability to “stop

Inv = 0.9,

High Inventory ordering” when inventory is

Demand = 0.5

high.

Inv = 0.1, Tests the response to “urgent

Low Inventory replenishment” when inventory

Demand = 0.5 is low.

Inv = 0.5, Tests are sensitive to market

High Demand

Demand = 0.9 opportunity.

Fig. 11. Top - 20 SHAP feature importance over the 660 - dimensional product - level state space for DQN and Inv = 0.5,

Tests the ability to reduce costs

A2C_mo d across EASY, MEDIUM, and HARD Low Demand

Demand = 0.1 when demand is weak

scenarios.

Extreme situation: inventory is

Inv = 0.05,

4. 6. Comparative Evaluation Critical State nearly depleted while demand is Demand = 0.8

high.

Stability evaluates the robustness of the MSX

𝑀 structure when the weight coefficient λ changes.

(𝑘)

𝑂𝐶𝑆 = ∑ 𝐼 (| Δ 𝑄 | ≥ ε)

𝑀

∑

𝑘 = 1 𝑠 ∈ 𝑆 𝐼 (𝑀𝑆 𝑋 𝑡𝑒𝑠𝑡 𝑠, λ = 1. 0 = 𝑀𝑆 𝑋 𝑠, λ = 𝑖)

𝑆𝑡𝑎𝑏𝑖𝑙𝑖𝑡𝑦 = × 100

𝑁 𝑡𝑒𝑠𝑡

Where:

This metric reflects the proportion of states in

- 𝑴 is the number of business objectives (Service, which the MSX set remains unchanged when the weights

Holding, Waste, Order), are adjusted within a reasonable range.

(𝒌)

- 𝜟 𝑸 is the contribution of objective 𝒌,
- 𝜺 is the threshold for significant contribution Dominance Ratio (DR): DR mea sures the relative
- 𝑰 is the indicator function. contribution of each reward component.

(𝑘) | Δ 𝑄 |

Feature Coverage Score (FCS): FCS measures the 𝐷 𝑅 𝑘 =

𝑀 ∑ | (𝑖)

proportion of state features that significantly influence the Q - 𝑖 = 1

Δ 𝑄 |

value in SHAP - based explanations.

𝑁%MSX - change 1

𝐹𝐶𝑆 = ∑ 𝐼 (| ϕ 𝑖 | ≥ ε)%MSX - change=100%−Stability

𝑁 𝑖 = 1

This metric reflects the degree of change in the MSX

Where: structure when the weights are adjusted.

- 𝑁 is the number of input features,
4. 6.3 RDX/MSX - only
- 𝜙 𝑖 is the SHAP value of feature 𝑖.

The results in Figure 1 2 reveal the existence of a

Cross - domain Alignment Score (CAS): CAS structural trade - off mechanism between the Objective

measures the consistency between the business objective Coverage Score (OCS) and the size of the minimal domain (RDX/MSX) and the state feature domain (SHAP). explanation set (MSX - size) as the parameter λ varies from

0. 5 to 2.0. Specif ically, as λ increases, the threshold for

| 𝑑𝑜𝑚𝑖𝑛𝑎𝑛𝑡 𝑓𝑒𝑎𝑡𝑢𝑟𝑒𝑠 ∩ 𝑑𝑜𝑚𝑖𝑛𝑎𝑛𝑡 𝑜𝑏𝑗𝑒𝑐𝑡𝑖𝑣𝑒𝑠 | determining a “significant” contribution in OCS becomes 𝐶𝐴𝑆 = more stringent, leading to a tendency for the number of

𝐾

objectives exceeding the threshold to decrease or remain low.

Where: At the same time, the MSX mechanism requ ires a higher level

of Q - gap coverage, forcing the algorithm to expand the

- The set of dominant features includes features whose objective set to ensure complete explanation. Consequently,

SH AP values exceed the threshold, MSX - size increases with λ, while OCS does not increase

- The set of dominant objectives includes objectives correspondingly, reflecting an inverse relationship between

with high RDX contributions, the tw o metrics

- 𝐾 is the number of important factors considered.

Fig 1 2. Dependency relationship with respect to λ between OCS and MSX - size for DQN and A2C_mod.

The difference between the two architectures is

clearly reflected in how they respond to the λ constraint. For

DQN, both OCS and MSX - size vary in a continuous and A2C_mod achieve Stability close to 100%, confirming that gradual manner, indicating that Q - values are relatively the MSX set under the baseline configuration is i nternally evenly distributed across multiple objectives. This structure consistent. However, when λ increases beyond 1.0, the two reflects a stable multi - objective decision - making mechanism, architectures exhibit clearly divergent behaviors. For DQN, in which multiple objectives jointl y contribute to shaping the Stability decreases in a continuous and gradual manner.

action value. In contrast, A2C_mod exhibits more When λ increases to 1.5, Stability drops to approximately pronounced nonlinear behavior. At low λ levels, MSX - size 52 – 53%, and fu rther declines to around 44 – 46% at λ = 2.0 remains small, implying that decisions are primarily across all three scenarios. This relatively smooth downward dominated by a few prominent objectives. However, when λ trend indicates that the MSX structure is adjusted in an exceeds a mo derate threshold, MSX - size abruptly increases incremental and inheritable manner: dominant objectives to its maximum level, indicating that the model must are retained, while additional objective s are incorporated to incorporate additional objectives to satisfy the Q - gap satisfy the higher coverage threshold. In contrast, coverage requirement. This suggests that the value structure A2C_mod demonstrates an abrupt decline. Once λ exceeds of A2C_mod is more concentrated and more sensitive to 1.0, Stability sharply drops to approximately 27 – 29% and explanation constraints. These findings indicate that the remains at this low level as λ continues to increase. This RDX/MSX mechanism not only reflects the number of pattern is consistently observed across all three components involved in an explanation but also reveals the environmental scenarios, indicating that changes in the intrinsic value distribution structure of each architecture. MSX structure occur in a stepwise rather than gradual DQN demonstrate s greater stability and a more distributed manner. Notably, the shape of the Stability curves is nearly multi - objective allocation, whereas A2C_mod tends to identical across EASY, MEDIUM, and HARD for ea ch concentrate strongly on dominant objectives and only model. This suggests that the decline in Stability is expands the explanation set when constraints become stricter. primarily driven by the XAI constraint (λ), rather than

environmental complexity. In other words, the stability of

Figure 1 3 illustrates the variation of the St ability the explanation set reflects the architectural characteristics

metric with respect to the constraint coefficient λ across the of value learning mor e than the nature of the task itself.

three scenarios EASY, MEDIUM, and HARD. Stability Overall, DQN maintains significantly higher Stability than measures the extent to which the MSX set structure is A2C_mod as λ increases, with nearly double the value preserved when λ changes relative to the reference when λ ≥ 1.5. These results indicate that the value - based configuration λ = 1.0, thereby reflecting the intrinsic architecture of DQN produces explanation sets th at are consistency of the explanation mechanism under more inheritable and robust under strict explanation hyperparameter perturbations. constraints, whereas A2C_mod is more sensitive to

variations in the XAI hyperparameter.

The results show that at λ = 1.0, both DQN and

Fig 1 3. Robustness of the MSX set with respect to the threshold coefficient λ across the EASY, MEDIUM, and HARD scenarios for DQN and A2C_mod

4. 6.3. SHAP - only different agent architectures.

To evaluate the effectiveness of the SHAP - only Figure 1 2 presents the variation of the Feature configuration, the study establishes experiments under Contribution Score (FCS) with respect to the parameter λ multiple different settings. Table 8 summarizes the for the two architectures, DQN and A2C_mod, across three complete experimental setup used in this analysis. The environmental scenarios (EASY, MEDIUM, HARD). The FCS results corresponding to the 24 experimental results are computed over 496 test states an d averaged configurations are summarized in Table 8, showing the across 5 independent runs. Overall, both agents maintain variation in feature coverage with respect to λ and across very high FCS values (>0.90) across all configurations,

indicating that the three input features (inventory, sales, environ mental difficulty does not reduce feature coverage.

and waste) make significant contributions to the decision - Overall, the results confirm that both agents rely on the making process. As λ increases (corresponding to a higher entire feature space; however, DQN is more robust to threshold ε according to the relation ε = 0.01λ), FCS variations in the parameter λ, and λ = 1.0 can be considered decreases monotonically for both models, reflecting a reasonable balance between sen sitivity and specificity in increasingly stringent criteria for identifying “significant” SHAP - based explanation analysis.

features. However, the magnitude of decline is greater for A2C_mod than for DQN, resulting in a widening gap Table 8. Experimental Parameter Configuration Design between the two models as λ increases.

Component Setting

This difference indicates that the SHAP value

Agents DQN, A2C_mod

distribution of A2C_mod tends to be more concentrated around the threshold ε, making features more likely to be

Scenarios EASY, MEDIUM, HARD

excluded as λ increases, whereas DQN maintains contributions that consistently exceed the threshold. At the Number of λ levels 0.5, 1.0, 1.5, 2.0 same time, DQN exhibits almost perfect reproducibility

Threshold ε (FCS) 0.5%, 1.0%, 1.5%, 2.0%

(standard deviation approximately 0), while A2C_mod shows small variance (<0.3%), confi rming the high stability Total configurations 2 agents × 3 scenarios × 4 λ = 24 of both models but with different levels of sensitivity to configurations threshold changes. Notably, FCS in the HARD scenario is

Number of runs 5 runs / configuration

not lower than in MEDIUM; in some configurations, it is even comparable or higher, indicating that increased

Table 9. Comparison of FCS (Mean ± Std) across λ and ε levels under different environmental scenarios.

Agent Scenario λ ε FCS (Mean ± Std)

0. 5 0.005 0.9906 ± 0.0000
1. 0 0.010 0.9825 ± 0.0000

EASY

1. 5 0.015 0.9731 ± 0.0000
2. 0 0.020 0.9610 ± 0.0000
0. 5 0.005 0.9825 ± 0.0000
1. 0 0.010 0.9684 ± 0.0000

DQN MEDIUM

1. 5 0.015 0.9563 ± 0.0000
2. 0 0.020 0.9483 ± 0.0000
0. 5 0.005 0.9872 ± 0.0000
1. 0 0.010 0.9792 ± 0.0000

HARD

1. 5 0.015 0.9731 ± 0.0000
2. 0 0.020 0.9704 ± 0.0000
0. 5 0.005 0.9805 ± 0.0004
1. 0 0.010 0.9636 ± 0.0010

EASY

1. 5 0.015 0.9366 ± 0.0007

A2C_mod 2.0 0.020 0.9199 ± 0.0008

0. 5 0.005 0.9777 ± 0.0017

MEDIUM 1.0 0.010 0.9598 ± 0.0003

1. 5 0.015 0.9429 ± 0.0010

Agent Scenario λ ε FCS (Mean ± Std)

2. 0 0.020 0.9223 ± 0.0017
0. 5 0.005 0.9886 ± 0.0013
1. 0 0.010 0.9753 ± 0.0005

HARD

1. 5 0.015 0.9647 ± 0.0007
2. 0 0.020 0.9185 ± 0.0014

all configurations), indicating that the explanation method

The results in Figure 1 6, showing the distribution of remains statistically stable. Notably, no abrupt increase in

the Feature Contribution Score (FCS), evaluate the statistical variance is observed when transitioning from the EASY to stability of the SHAP - only method under the influence of HARD scenario for either agent, suggesting that agent architecture, environmental difficulty, and environmenta l difficulty does not compromise measurement classification threshold. The results indicate that DQN reproducibility. The influence of the parameter λ is also achieves nearly perfect reproducibility across all 12 clearly observed. While DQN maintains invariant stability configurations, with boxplots collapsing into thin horizontal across λ values, A2C_mod shows a slight increase in lines and no observable dispersion across runs. This suggests dispersion as λ increases, particularly at λ = 2.0. This trend that FCS is not affected by randomness in the SHAP may be explained by the presence of features with SHAP estimation process and refl ects the high determinism of the values close to the classification threshold, leading to minor trained Q - network. In contrast, A2C_mod exhibits small but fluctuations in FCS across runs. However, the magnitude of finite dispersion, reflected in narrow interquartile ranges and variation remains very limited and does not alter the overa ll very low variance. However, this level of variation remains trend of the results.

negligible compared to the mean FCS valu es (above 0.91 in

Fig 1 4. Effect of the parameter λ on the FCS of DQN and A2C_mod across the three environmental scenarios.

Fig 1 5. Distribution of the Feature Contribution Score (FCS) of DQN and A2C_mod across different λ levels in the three environmental scenarios.

4. 6.4. Combined frame alignment between feature - level explanations (SHAP) and

objective - level explanations (RDX) does not always exist.

The results in Figure 1 5 provide a comprehensive view Alignment becomes more evident in the HARD environment of the agent’s decision - making structure at both the objective and at sensitive λ thresholds, especially for A2C_mod, level and the feature level. The OCS results indicate that suggesting that under high - pressure conditions, the linkage environmental difficulty directly affects the number of structure between features and objectives becomes more objectives dominating the policy. In the EASY sc enario, consistent. A comparison between the two architectures both agents maintain a stable single - objective structure (OCS shows that A2C_mod maintains both a multi - objecti ve = 0.25) across the entire λ range, reflecting a focused strategy structure and more stable alignment as λ increases, whereas with no significant competition among reward components. DQN is more sensitive to threshold changes and tends to lose In contrast, in the MEDIUM and HARD scenarios, its multi - objective structure at high λ values. Overall, these particularly at l ow λ values, a multi - objective structure (OCS findings emphasize that multi - objective optimization does = 0.50) emerges, indicating the presence of trade - offs among not necessa rily imply direct compatibility between the two objectives as the level of competition increases. However, explanation layers. Therefore, jointly analyzing OCS and when compared with CAS, most configurations still yield a CAS is necessary to uncover potential decoupling between value of 0, implying that althou gh the agent may feature - level reasoning and objective - level reasoning in simultaneously optimize multiple objectives, direct multi - objective reinforcement learnin g systems.

Fig 1 6. Sensitivity of Objective Coverage Score (OCS) to Threshold Parameter λ across Scenarios for DQN and A2C_mod

Fig 1 7. Cross - domain Alignment Score (CAS) across Threshold Levels (λ), Agents, and Environmental Scenarios.

Figure 1 7 presents the Cross - domain Alignment Score baseline (𝐴𝐹 𝑅 𝑅𝑎𝑛𝑑𝑜𝑚), which masks the same number of (CAS) between feature - level explanations (SHAP) and reward componen ts randomly. The faithfulness ratio is objective - level explanations (RDX) across different defined as:

threshold values λ in the three scenarios EASY, MEDIUM,

𝐴𝐹 𝑅

and HARD for the two agents DQN and A2C_mod. The 𝑀𝑆𝑋

𝐹𝑅 =

results show that most configurations yield CAS = 0, 𝐴𝐹 𝑅 𝑅𝑎𝑛𝑑𝑜𝑚 reflecting the lack of direct compatibility between the two

where 𝐹𝑅 > 1 indicates that MSX - selected components

explanation layers. This implies that the objec tives activated

are more influential than randomly selected components.

at the RDX level are not necessarily directly mapped from features with high SHAP contributions. In the EASY scenario, CAS remains 0 across the entire λ range for both agents, indicating that although the policy may optimize a dominant obje ctive, the feature - level structure does not exhibit clear overlap with objective - level reasoning. In the MEDIUM scenario, alignment appears only at low λ values:

DQN achieves CAS = 0.25 at λ = 0.5, while A2C_mod maintains CAS = 0.25 at λ = 0.5 and 1.0 befo re dropping to 0 Fig 18. Action flip rate comparison between MSX - as λ increases. This suggests that when the evaluation guided mas king and random masking across EASY, threshold is sufficiently sensitive, overlap between the two MEDIUM, and HARD scenarios under different explanation sets exists; however, as λ becomes more 𝝀 thresholds.

stringent, alignment declines because only the strongest contributing c omponents are retained. Notably, in the HARD scenario, A2C_mod maintains CAS = 0.25 across the entire λ range, whereas DQN achieves alignment only at λ ≤ 1.0 and loses alignment at higher values. These results indicate that under high - pressure environments, the Actor – Critic architecture demonstrates a more robust ability to maintain linkage between feature - level and objective - level reasoning.

Overall, the figure confirms that compatibility between the two explanation layers is not an inherent property of th e Figure 19. RDX/MSX faithfulness analysis based on learned policy, but rather depends on environmental faithfulness ratio and Q - value gap drop under MSX - complexity, agent architecture, and the strictness level of the guided masking and random masking.

λ threshold.

The results show that MSX - guided masking becomes more faithful when the threshold is sufficiently strict. In

4.7 Faithfulness Evaluation of Explanations

To complement the descriptive metrics in Section 4.6, this study evaluates the faithfulness of the generated explanations through perturbation-based tests. The underlying principle is that if the components or features identified as important are truly influential, their removal should produce larger changes in the model output than the removal of less relevant or randomly selected elements. This evaluation is applied to both reward-level explanations (MSX) and feature-level explanations (SHAP) and is designed to provide causal evidence that the explanations reflect functional dependence rather than merely descriptive association. All tests are conducted on the trained agents without retraining, using the same data partition and evaluation protocol.

4.7.1 Perturbation Protocol

The protocol is designed to isolate the causal effect of the importance ranking while preserving the remainder of the state distribution. Each state consists of 660 dimensions formed by three consecutive blocks representing inventory, demand and waste, where waste is modelled as a linear function of inventory with controlled noise and bounded within the admissible interval. Three operational scenarios of increasing difficulty are examined by scaling demand intensity and waste rate, corresponding to easy, medium and hard conditions. Within each scenario, fifty test states are extracted from the normalized test sequence in combination with the initial inventory level, yielding one hundred and fifty states per agent and three hundred observations across the two agents. This size balances statistical power with the cost of repeated evaluation, as each random condition involves thirty repetitions.

Feature importance is determined by the mean absolute SHAP value obtained from the preceding micro-level analysis, which ranks all 660 features by their average influence across the test set. Three masking strategies are compared on the identical set of states and with the identical replacement mechanism. The Most Relevant First strategy removes features in descending order of SHAP importance, the Least Relevant First strategy follows the reverse order, and a random strategy removes the same number of features at random and serves as a calibrated baseline. The perturbation level is varied from one to ten features, corresponding to approximately 0.15 to 1.52 percent of the full space, a range chosen to be small enough to preserve the overall data manifold yet large enough to observe a measurable response. The replacement value is fixed at the median of one hundred background states generated from a controlled uniform distribution, a choice motivated by the need to erase information without introducing out-of-distribution values while maintaining operational feasibility. For the random strategy, each level is repeated thirty times and the result is summarized by the mean across repetitions, with the action-switching outcome determined by majority voting. A fixed random seed is used throughout to ensure full reproducibility.

For the value-based agent, the output change is quantified as the relative decrease in the Q-value of the originally optimal action, which normalises for differences in absolute scale across states. For the actor-critic agent, the change is measured as the absolute decrease in the policy probability of the optimal action, reflecting a direct loss of confidence. The action switching rate is defined as the proportion of cases in which the optimal action index changes after perturbation, thereby capturing discrete decision shifts in addition to continuous output changes. All measures are averaged across the product dimension within a state to reflect portfolio-level behaviour.

[Figure - task15-9_perturbation_with_CI.png]
*Figure 20. Perturbation curves as a function of the number of masked features for the two agents across the three scenarios. The Most Relevant First condition consistently yields the largest output change, followed by the random condition, while the Least Relevant First condition produces the smallest change. Shaded bands represent dispersion across states, with precise confidence intervals reported in the accompanying tables.*

4.7.2 SHAP Faithfulness: Area Under Perturbation Curve and Confidence Intervals

To summarise the cumulative effect of the perturbation process, the area under the perturbation curve is computed by trapezoidal integration across the ten successive masking levels, and an analogous area is computed for the action switching curve. This provides a compact index of faithfulness in which a larger area indicates a larger average degradation or a higher switching rate over the entire process. Uncertainty for the mean change and for the area is estimated by non-parametric bootstrap with one thousand resamples over the set of states, yielding 95% intervals, while the switching rate is equipped with Wilson intervals appropriate for binomial proportions near the boundary. This combination allows simultaneous assessment of continuous and discrete decision uncertainty.

The empirical pattern follows the expected faithfulness ordering in all conditions. The Most Relevant First condition produces a positive and monotonically increasing change with the number of masked features, the random condition remains near zero, and the Least Relevant First condition yields a negative or near-zero change, indicating that removing the least important features does not harm and may slightly improve the evaluated value. The magnitude, however, differs markedly between architectures, with the actor-critic agent exhibiting changes approximately twice as large as those of the value-based agent. The action switching rate remains at zero across all eighteen conditions even at the maximum masking level, with Wilson intervals ranging from zero to approximately seven percent, indicating that the discrete decision is highly robust within the examined perturbation range.

Table 8 summarises the cumulative and representative point estimates. To keep the layout compact for the narrow paper format, the full set of 18 conditions is split into two complementary views. The first focuses on the integrated effect and the second on the change at the maximum masking level, while switching and interval details are summarised in the text rather than in the table.

Table 8a. Area under perturbation curve by strategy (95% interval width <0.002).

| Agent | Scenario | Most Relevant First | Random | Least Relevant First |
| :--- | :--- | :---: | :---: | :---: |
| Value-based | Easy | 0.014 | -0.001 | -0.010 |
| Value-based | Medium | 0.011 | -0.001 | -0.010 |
| Value-based | Hard | 0.010 | -0.001 | -0.011 |
| Actor-critic | Easy | 0.024 | 0.007 | -0.020 |
| Actor-critic | Medium | 0.024 | 0.006 | -0.0003 |
| Actor-critic | Hard | 0.025 | 0.005 | 0.0002 |

Table 8b. Output change at maximum masking (10 features, ~1.5% of space).

| Agent | Scenario | Most Relevant First | Random | Least Relevant First |
| :--- | :--- | :---: | :---: | :---: |
| Value-based | Easy | 0.0023 | -0.00016 | -0.0011 |
| Value-based | Medium | 0.0020 | -0.00011 | -0.0012 |
| Value-based | Hard | 0.0017 | -0.00021 | -0.0013 |
| Actor-critic | Easy | 0.0040 | 0.0013 | -0.0022 |
| Actor-critic | Medium | 0.0039 | 0.0012 | -0.00003 |
| Actor-critic | Hard | 0.0036 | 0.0010 | 0.00032 |

The switching rate is zero in all conditions with Wilson intervals from zero to approximately seven percent, and 95% bootstrap intervals for the area and for the changes are narrow and do not overlap between Most Relevant First and random in any scenario, reinforcing the discriminability of the strategies.

[Figure - task15-9_asr_with_CI.png]
*Figure 21. Action switching rate as a function of masking level. Across all six panels, the three strategies remain at zero at every level, demonstrating that the discrete decision of both agents is robust to removal of up to ten features within the examined range.*

4.7.3 Statistical Comparisons: Ordered Masking and Minimal Sufficient Guidance

The superiority of the ordered conditions is tested at the level of individual states to distinguish systematic effects from differences in means alone. For each agent and scenario, the paired vector of differences between two strategies across the fifty states is examined with a non-parametric paired test that does not require normality, complemented by a paired parametric test as a reference. Effect size is quantified on the difference vector and correction for multiple comparisons is performed across the full family of tests. The procedure is applied both to the change at the maximum masking level and to the integrated area, ensuring that the conclusion does not depend on a single point on the curve.

All comparisons of Most Relevant First against the random baseline and against the Least Relevant First condition reach significance after correction, with large to very large effect sizes. The integrated comparison yields the same conclusion, indicating that the advantage holds over the entire perturbation process and not merely at an isolated level. The only comparison with a moderate effect corresponds to the random versus Least Relevant First contrast for the actor-critic agent in the hard scenario, where the Least Relevant First change is near zero rather than clearly negative as in the other conditions.

Beyond the ordered masking, a complementary test examines whether masking guided by a minimal sufficient set produces a larger change than masking a randomly selected set of the same size, thereby connecting feature-level and reward-level faithfulness. With a set size of five, the guided condition consistently outperforms the random condition across both agents and all scenarios with highly significant corrected values and large effects. With a size of two, the advantage remains significant but with a smaller effect, reflecting that removing too few features is insufficient to generate a substantial operational difference. Table 9 summarises the minimal-set comparison; the full family of ordered comparisons is reported in the supplementary material.

Table 9. Minimal sufficient guidance versus random masking of identical size (selected conditions; full set in supplement).

| Condition | Guided | Random | Difference | Corrected p | Effect size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Value-based, Easy, k=5 | 0.0014 | -0.00010 | 0.0016 | <0.001 | 3.8 |
| Value-based, Medium, k=5 | 0.0010 | -0.00013 | 0.0011 | <0.001 | 3.1 |
| Actor-critic, Easy, k=5 | 0.0027 | 0.00076 | 0.0019 | <0.001 | 2.8 |
| Actor-critic, Hard, k=2 | 0.0024 | 0.00022 | 0.0022 | <0.001 | 3.9 |

4.7.4 Threshold for Meaningful Change

The definition of a meaningful change is placed in relation to the granularity of the action space and to the natural noise of the system. The action space is discretised into fourteen replenishment levels, with the smallest increment between consecutive levels equivalent to one half of one percent of storage capacity. On this basis, a relative change of one percent in the Q-value and an absolute change of one percent in policy probability are adopted as thresholds that are sufficiently large to exceed noise due to waste and sampling, while remaining small enough to retain operational relevance, as this level corresponds to approximately two minimal action steps. In addition to this continuous threshold, a change in the optimal action index is regarded as a discrete threshold of direct decision relevance.

Applying the one-percent relative threshold reveals that no state exceeds the level in any of the eighteen conditions, including the Most Relevant First condition that produces the largest change. The average change at the maximum masking level ranges between approximately 0.16 and 0.40 percent, well below the adopted level. The discrete action-switching threshold yields the same result. This finding does not indicate a lack of faithfulness in the ranking, as the ordering between strategies remains highly significant, but suggests that the threshold is relatively high compared with the observed scale of change within the examined range and that a lower level or an extension to larger masking levels would be required to observe threshold exceedance. The threshold is therefore reported as a sensitivity parameter rather than a single fixed value.

Table 10. Proportion exceeding the one-percent threshold at maximum masking (Most Relevant First only).

| Agent | Scenario | Mean | SD | Proportion |
| :--- | :--- | :---: | :---: | :---: |
| Value-based | Easy | 0.0023 | 0.00044 | 0.00 |
| Value-based | Hard | 0.0017 | 0.00081 | 0.00 |
| Actor-critic | Easy | 0.0040 | 0.00009 | 0.00 |
| Actor-critic | Hard | 0.0036 | 0.00059 | 0.00 |

[Figure - task15-9_threshold_hist.png]
*Figure 22. Distribution of the output change at maximum masking for the Most Relevant First condition across the six panels. The dashed line indicates the one-percent threshold. The entire mass of each distribution lies to the left of the line, illustrating why the exceedance proportion is zero and suggesting calibration of the threshold to the empirical scale.*

4.7.5 Consistency Across States

Consistency is assessed through the distribution of the change across the fifty states within each condition, rather than relying on the mean alone. Descriptive measures including the mean, standard deviation, coefficient of variation and interquartile range are computed at the maximum masking level, and the distributions are visualised to allow assessment of heterogeneity. Systematic differences across scenarios are tested with a non-parametric test for independent groups, clarifying whether faithfulness depends on operational difficulty.

The actor-critic agent exhibits low dispersion with coefficients of variation between approximately 0.02 and 0.17, indicating a relatively uniform response across states. The value-based agent shows increasing heterogeneity with difficulty, with the coefficient rising from approximately 0.19 in the easy scenario to 0.48 in the hard scenario, reflecting growing non-uniformity as operational pressure intensifies. The test across scenarios indicates significant differences for both agents, demonstrating that faithfulness is not entirely invariant to operating conditions but exhibits a modest decline in magnitude and an increase in dispersion when moving from easy to hard. The ordering between the three strategies, however, is preserved in every scenario, indicating that the faithfulness of the ranking is stable even though the absolute level fluctuates.

Table 11. Consistency at maximum masking (Most Relevant First only).

| Agent | Scenario | CV | IQR | Median |
| :--- | :--- | :---: | :---: | :---: |
| Value-based | Easy | 0.19 | 0.00075 | 0.0023 |
| Value-based | Hard | 0.48 | 0.00130 | 0.0017 |
| Actor-critic | Easy | 0.02 | 0.00008 | 0.0040 |
| Actor-critic | Hard | 0.17 | 0.00057 | 0.0036 |

[Figure - task15-9_per_state_violin.png]
*Figure 23. Per-state distribution of the output change at maximum masking for the three strategies across the six panels. Each violin represents the full distribution of fifty observations together with its median and mean. The violins for the actor-critic Most Relevant First condition are narrow and elevated, indicating high consistency, while those for the value-based agent widen progressively from easy to hard, reflecting the increase in dispersion quantified in the accompanying table.*

Together, the perturbation, statistical, threshold and consistency analyses provide convergent evidence that the feature ranking is functionally faithful in terms of ordering, that the advantage over random and reverse baselines is statistically robust and practically sizable when integrated over the curve, that the minimal sufficient guidance adds complementary evidence across explanation spaces, and that the ranking is stable while its absolute magnitude and dispersion depend modestly on operational difficulty. These results directly address the five faithfulness requirements and establish that the proposed explanations are not merely descriptive but are grounded in measurable model behaviour, particularly under the more demanding inventory conditions.

### 4.8 Scalability to Product-Group Level (Task 5)

Scalability was evaluated by training A2C-mod and DQN independently on the three groups under identical settings of 600 episodes and 900 steps per episode with 14 discrete replenishment actions. All six models converged. For A2C, the mean reward over the final 100 episodes was 0.1807 for Fast, 0.0477 for Medium and 0.1514 for Slow. For DQN, the corresponding values were 0.7709 for Fast, 0.6508 for Medium and 0.5740 for Slow. DQN consistently outperformed A2C in every group, with differences of -0.59, -0.60 and -0.42, and showed improvements of 723% to 5,449% from the first 50 episodes, compared to only 1.48% for A2C Fast and a 79% decline for A2C Medium, indicating that the Medium group requires separate tuning. Results demonstrate that the framework is not hard-coded to 220 SKUs and scales to product-group granularity.

[Figure 3]
The combined learning curves compare A2C and DQN per group over 600 episodes. DQN curves are consistently above A2C curves after the early phase. The gap is largest for the Medium group and smallest for the Slow group, while all DQN groups show a steady increase and all A2C groups remain flat or slightly declining. This indicates a systematic advantage of DQN at reduced scale.

[Figure 4]
The bar chart compares the mean reward over the final 100 episodes for A2C and DQN in each group. In every group the DQN bar is three to four times higher than the corresponding A2C bar. The ranking Fast > Medium > Slow is preserved for both algorithms, but the absolute level is substantially higher for DQN.

[Figure 5]
The bar chart compares the mean stockout rate over the final 100 episodes. A2C shows markedly higher stockout rates, particularly for the Fast group, while DQN stockout rates are near zero in all groups. This indicates that DQN better prevents stockouts at the product-group level.

[Figure 6]
The bar chart compares the mean waste rate over the final 100 episodes. DQN waste rates are slightly higher than those of A2C in all groups, reflecting a trade-off between aggressive replenishment to avoid stockouts and increased waste.

**Table 1 - Task 5: A2C on 3 SKU groups (last100)**

| Group | Episodes | Reward_last100 | Stockout | Waste |
| :--- | :---: | :---: | :---: | :---: |
| Fast | 460 | 0.1807 | 0.1514 | 0.0103 |
| Medium | 600 | 0.0477 | 0.0450 | 0.0195 |
| Slow | 600 | 0.1514 | 0.0583 | 0.0169 |

**Table 2 - Task 5: DQN on 3 SKU groups (last100)**

| Group | Episodes | Reward_last100 | Stockout | Waste | Quantile |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fast | 600 | 0.7709 | 0.00008 | 0.0216 | 0.2023 |
| Medium | 600 | 0.6508 | 0.00042 | 0.0217 | 0.3207 |
| Slow | 600 | 0.5740 | 0.00238 | 0.0219 | 0.3929 |

**Table 3 - Task 5: A2C vs DQN per group**

| Group | A2C | DQN | Δ (A2C−DQN) | Winner |
| :--- | :---: | :---: | :---: | :---: |
| Fast | 0.1807 | 0.7709 | -0.5902 | DQN |
| Medium | 0.0477 | 0.6508 | -0.6031 | DQN |
| Slow | 0.1514 | 0.5740 | -0.4226 | DQN |

Detailed results are provided in the supplementary material.

### 4.9 Sensitivity to Action Space Resolution

The preceding analyses assume a fixed discretization of the replenishment decision into fourteen levels. To assess whether the conclusions depend on this specific granularity, a systematic sensitivity study was conducted across three resolutions representing coarse, baseline and fine discretizations. The coarse condition employs seven levels ranging from no replenishment to full replenishment with sparse intermediate steps, the baseline retains the fourteen levels used throughout the study, and the fine condition employs twenty-eight levels with dense interpolation between conservative and aggressive actions. The same inventory dynamics, reward structure and state representation were preserved across all conditions, and the two agent families were trained independently under each resolution. For the actor-critic family the hidden dimension was retained at its individually tuned optimum, while for the value-based family the corresponding optimum was retained, since equalizing capacity would have degraded performance and obscured the effect of resolution itself. All evaluations were performed deterministically on the held-out test sequence with a fixed seed, using a greedy policy without exploration, so that observed differences can be attributed to resolution rather than stochastic variation.

Operational performance was measured through a deterministic rollout over the full test horizon, recording average reward together with its constituent cost terms. The evaluation reproduces the exact environment transition used during training, thereby preserving consistency between learning and testing. From an explainability perspective, the same three complementary lenses employed earlier were applied in a standardized manner. At the feature level, Shapley-based attributions were computed with a reduced background and a fixed number of coalitions to keep computation tractable while retaining the kernel-based formulation, and the resulting feature coverage was summarized. At the objective level, the difference between the best and second-best actions was decomposed into service, holding, waste and balance components through a one-step lookahead, from which objective coverage, the size of the minimal sufficient set and its stability under threshold variation were derived. These measures jointly capture whether explanations remain sparse and stable when granularity changes.

The performance results reveal a non-monotonic relationship between resolution and effectiveness, and the pattern differs markedly between the two families. The value-based agent maintains high reward across all resolutions and benefits slightly from finer granularity, whereas the actor-critic agent attains its best performance at the baseline resolution and degrades substantially under both the coarser and finer discretizations due to a sharp increase in overstock. The finding indicates that a moderate resolution provides the most favorable trade-off for the policy-based method, while the value-based method is more tolerant to extremes when the levels are appropriately spaced. The accompanying cost terms confirm that the degradation of the actor-critic agent at the extremes is operationally meaningful and not an artifact of reward scaling.

**Table 4a. Operational performance on the held-out test set across resolutions.**

| Agent | Resolution | Reward | Overstock | Waste |
| :--- | :---: | :---: | :---: | :---: |
| Actor-critic | 7 | -0.13 | 0.96 | 0.024 |
| Actor-critic | 14 | 0.38 | 0.00 | 0.013 |
| Actor-critic | 28 | -0.12 | 0.84 | 0.024 |
| Value-based | 7 | 0.80 | 0.00 | 0.024 |
| Value-based | 14 | 0.60 | 0.00 | 0.019 |
| Value-based | 28 | 0.82 | 0.01 | 0.024 |

*The table reports the mean over the full test horizon under a deterministic greedy policy. Higher reward and lower costs are preferable. The value-based agent is reported with its tuned capacity and the actor-critic agent with its own, as equalizing capacity would have disadvantaged the former.*

[Figure - task4_reward_vs_actions.png]
*Figure 12a. Average reward on the test set as a function of resolution. The value-based agent remains in the high-reward regime across all three resolutions with a modest advantage for the fine discretization, while the actor-critic agent exhibits a pronounced peak at the baseline resolution and declines at both extremes.*

Explanation robustness, in contrast, does not follow the performance trend and shows no monotonic dependence on granularity. Feature-level sparsity remains high in all conditions, with only a small fraction of features exceeding the significance threshold, and the average magnitude of attributions differs by less than an order of magnitude between families. Objective coverage and the size of the minimal sufficient set vary less systematically, although the fine resolution for the actor-critic agent requires on average more objectives to reach the same explanatory threshold. Stability under threshold perturbation is highest for the coarse resolution of the actor-critic agent and remains within a moderate band for the value-based agent across all resolutions. Taken together, the results indicate that while performance is sensitive to resolution in a family-dependent manner, the sparsity and relative stability of explanations are largely preserved, with the notable exception of increased explanatory complexity under very fine discretization for the policy-based method.

**Table 4b. Robustness of explanations across resolutions.**

| Agent | Resolution | Feature coverage | Objective coverage | Minimal set size | Stability |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Actor-critic | 7 | 0.00 | 0.63 | 1.00 | 90.7 |
| Actor-critic | 14 | 0.00 | 0.28 | 1.01 | 50.0 |
| Actor-critic | 28 | 0.00 | 0.59 | 2.68 | 59.8 |
| Value-based | 7 | 0.02 | 0.42 | 1.18 | 58.8 |
| Value-based | 14 | 0.00 | 0.29 | 1.51 | 52.0 |
| Value-based | 28 | 0.02 | 0.39 | 1.70 | 60.1 |

*Feature coverage is the fraction of features whose absolute attribution exceeds the significance threshold; objective coverage is the analogous fraction for reward components; minimal set size is the average cardinality of the smallest sufficient explanation; stability is the Jaccard-based preservation under threshold variation. Values are averaged over the test horizon under the same deterministic evaluation.*

[Figure - task4_fcs_vs_actions.png]
*Figure 12b. Feature coverage as a function of resolution. Both families remain in the sparse regime, indicating that only a small subset of features is deemed significant irrespective of granularity.*

[Figure - task4_stability_vs_actions.png]
*Figure 12c. Stability of the minimal sufficient set as a function of resolution. The value-based agent shows a relatively flat profile, while the actor-critic agent varies more markedly, with the coarse resolution being the most stable.*

Overall, the sensitivity study demonstrates that operational effectiveness and explanatory characteristics respond differently to resolution. The fourteen-level discretization represents a balanced choice for the actor-critic family, whereas the value-based family tolerates both coarser and finer discretizations when level spacing is appropriate. The preservation of sparsity and moderate stability across resolutions supports the generalizability of the earlier explanatory conclusions, while the increase in minimal set size under very fine granularity for the policy-based agent highlights a practical limit where finer control comes at the cost of more complex justifications.

5. Conclusion

This study proposes a unified Explainable Reinforcement Learning (XRL) framework for large-scale inventory management, aiming to systematically analyze the decision-making mechanisms of two representative DRL architectures, DQN and A2C_mod. By integrating Reward Difference Explanation (RDX), Minimal Sufficient Explanation (MSX), and SHAP, the framework enables multi-level interpretation, including trade-offs in the reward space, compact justification sets derived from MSX, and the contribution levels of state features. Experimental results reveal clear structural differences between the two architectures. A2C_mod demonstrates the ability to reallocate the importance of objectives as environmental complexity increases, indicating flexible policy adaptation.

In contrast, DQN exhibits more stable but less adaptive decision patterns, with a tendency to rely on a narrower set of dominant objectives. SHAP analysis at both global and local levels further confirms that A2C_mod distributes attentio n more broadly across state features, whereas DQN primarily focuses on inventory and demand signals. These findings suggest that explanation stability and decision adaptability are architecture - dependent properties rather than merely reflections of perform ance differences

Although the proposed framework provides structured

Fig 20. Perturbation curves for SHAP faithfulness insights into reasoning in both reward space and feature

space, several limitations point to directions for future

verification under MoRF, Random, and LeRF masking.

research. First, explanation robustness should be evaluated

The results generally follow the expected faithfulness across multiple random seeds using dedicated stability pattern: metrics to distinguish intrinsic architectural characteristics

from variability caused by stochastic training processes.

𝐷𝑟𝑜 𝑝 𝑀𝑜𝑅𝐹 > 𝐷𝑟𝑜 𝑝 𝑅𝑎𝑛𝑑𝑜𝑚 > 𝐷𝑟𝑜 𝑝 𝐿𝑒𝑅𝐹 Second, future studies should extend SHAP analysis beyond For DQN, masking the most relevant SHAP features aggregated features to more f ine - grained product - level and produces larger output changes than masking random or least system - level representations to mitigate information loss due

to dimensionality reduction. Third, systematic validation by

relevant features. This effect beco mes clearer in the

domain experts is needed to assess the alignment between the

MEDIUM and HARD scenarios, where the selected action is

identified trade - offs and feature allocati ons and real - world

more likely to change after MoRF masking. This suggests

managerial reasoning. In addition to methodological

that the top - ranked SHAP features are functionally related to

extensions, applying the framework to other DRL algorithms

DQN’s Q - value estimation and action selection. For such as PPO or SAC would enable assessment of the A2C_mod, the selected action remains more stable, with generalizability of the explanation mechanisms across limited action switching across scenarios. However, the policy - gradient and ac tor - critic variants. Furthermore,

incorporating richer operational constraints such as budget

selected - action probability still decreases more strongly

limits, delivery lead times, and supply chain risks would

under MoRF than under Random or LeRF masking. This

enhance the practical applicability of the model.

indicates that SHAP - identified features affect t he confidence of the policy, even when the final selected action does not

CRediT author statement

change. Overall, the perturbation - based results provide

Dac Hoang Nguyen: Conceptualization of the review

empirical evidence that the proposed explanations are not

article, Literature search, Writing the manuscript as the

merely descriptive. RDX/MSX identifies reward

main author and corresponding author; Egawa Masao,

components whose remov al can change the selected action, Viet Pham Quoc Le: Reviewing, Data synthesis, Critical while SHAP identifies state features whose perturbation feedback; Nhu Tai Do: Editing, Refining the discussion

and c onclusions.

COMPETING INTERESTS 2026. [Online]. Available:

https://dl.acm.org/doi/book/10.5555/3312046

The authors declare that there is no conflict of interest

[13] V. Mnih et al., “Human - level control through deep

regarding the publication of this article.

r einforcement learning,” Nature, vol. 518, no. 7540, pp.

REFERENCES 529 – 533, Feb. 2015, doi: 10.1038/nature14236.

[14] A. Barredo Arrieta et al., “Explainable Artificial

[1] T. M. Whitin, “Inventory Contro l Research: A Intelligence (XAI): Concepts, taxonomies, Survey,” Management Science, vol. 1, no. 1, pp. 32 – opportunities and challenges toward responsible AI,” 40, 1954. Inf ormation Fusion, vol. 58, pp. 82 – 115, Jun. 2020, [2] G. Liu, W. Deng, X. Xie, L. Huang, and H. Tang, doi: 10.1016/j.inffus.2019.12.012.

“Human - Level Control Through Directly Trained [15] “Explainable post hoc portfolio management financial Deep Spiking Q - Networks,” IEEE Transactions on policy of a Deep Reinforcement Learning agent | PLOS Cybernetics, vol. 53, no. 11, pp. 71 87 – 7198, Oct. 2023, One.” Accessed: Mar. 05, 2026. [Online]. Available:

doi: 10.1109/TCYB.2022.3198259. https://j ournals.plos.org/plosone/article?id=10.1371/jo [3] V. Mnih et al., “Asynchronous Methods for Deep urnal.pone.0315528 Reinforcement Learning,” in Proceedings of The 33rd [16] A. Kuznietsov, B. Gyevnar, C. Wang, S. Peters, and S.

International Conference on Machine Learning, V. Albrecht, “Explainable AI for Safe and Trustworthy PMLR, Jun. 2016, pp. 1928 – 1937. Accessed: Mar. 05, Autonomous Driving: A Systematic Review,” IEEE

2026. [Online]. Available: Transactions on Intelligent Tra nsportation Systems,

https://proceedings.mlr.press/v48/mniha16.html vol. 25, no. 12, pp. 19342 – 19364, Oct. 2024, doi:

[4] H. Meisheri, V. Baniwal, N. N. Sultana, H. Khadilkar, 10.1109/TITS.2024.3474469.

and B. Ravindran, “Using Reinforcement Learning for [17] “Explainable AI for Robot Failures | Proceedings of the a Large Variable - Dimensional Inventory Management 2021 ACM/IEEE International Conference on Human - Problem”. Robot Interaction.” Accessed: Mar. 05, 2026. [Onl ine].

[5] “Explainable Reinforcement Learning: A Survey | Available:

Springer Nature Link.” Accessed: Mar. 05, 2026. https://dl.acm.org/doi/10.1145/3434073.3444657 [Online]. Available: [18] E. Puiutta and E. M. S. P. Veith, “Explainable https://link.springer.com/chapter/10.1007/978 - 3 - 030 - Reinforcement Learning: A Survey,” in Machine 57321 - 8_5 Learning and Knowledge Extraction, A. Holzinger, P.

[6] Z. Juozapaitis, A. Koul, A. Fern, M. Erwig, and F. Kieseberg, A. M. Tjoa, and E. Weippl, Eds., Cham:

Doshi - Velez, “Explainable Reinforc ement Learning Springer International Publishing, 2020, pp. 77 – 95.

via Reward Decomposition”. doi: 10.1007/978 - 3 - 030 - 57321 - 8_5.

[7] “A unified approach to interpreting model predictions | [19] “Understanding the Actions of an Agent Utilizing an Proceedings of the 31st International Conference on Actor - Critic Algorithm in Deep Reinforcement Neural Information Processing Systems.” Accessed: Learning | Request PDF,” ResearchGate. Accessed:

Mar. 05, 2026. [Online]. Available: Mar. 05, 20 26. [Online]. Available:

https://dl.acm.o rg/doi/10.5555/3295222.3295230 https://www.researchgate.net/publication/393014390_ [8] D. Beechey, T. M. S. Smith, and Ö. Şimşek, Understanding_the_Actions_of_an_Agent_Utilizing_ “Explaining Reinforcement Learning with Shapley an_Actor - Values,” in Proceedings of the 40th International Critic_Algorithm_in_Deep_Reinforcement_Learning Conference on Machine Learning, PMLR, Jul. 2023, [20] R. N. Boute, J. Gijsbrechts, W. van Jaarsveld, and N.

pp. 2003 – 2014. Accessed: Mar. 05, 2026. [Online]. Vanvuchelen, “De ep reinforcement learning for Available: inventory control: A roadmap,” European Journal of https://proceedings.mlr.press/v202/beechey23a.html Operational Research, vol. 298, no. 2, pp. 401 – 412, [9] M. Sieke, “Foundations of Inventory Management,” in Apr. 2022, doi: 10.1016/j.ejor.2021.07.016.

Supply Chain Contract Management: A Performance [21] “[PDF] Attention, Learn to Solve Routing Problems! | Analysis of Efficient Supply Chain Contracts, M. Semantic Scholar.” Ac cessed: Mar. 05, 2026. [Online].

Sieke, Ed., Wiesbaden: Springer Fachmedien, 2008, Available:

pp. 9 – 36. doi: 10.1007/978 - 3 - 658 - 24382 - 1_2. https://www.semanticscholar.org/paper/Attention%2C [10] E. A. Silver, D. F. Pyke, and D. J. Thomas, Inventory - Learn - to - Solve - Routing - Problems! - Kool - and production management in supply chains, Fourth Hoof/ce4f001c1d8ddb9a95cf54e14240ef02c44bd329 edition, First issued in paperback. Boca Raton, FL London New York, NY: C RC Press, Taylor & Francis Group, 2021.

[11] J. Si, A. G. Barto, W. B. Powell, and D. Wunsch, “The Linear Programming Approach to Approximate Dynamic Programming,” in Handbook of Learning and Approximate Dynamic Programming, IEEE, 2004, pp.

153 – 178. doi: 1 0.1109/9780470544785.ch6.

[12] “Reinforcement Learning: An Introduction | Guide books | ACM Digital Library.” Accessed: Mar. 05, 2026. [Online]. Available: https://dl.acm.org/doi/book/10.5555/986238

[22] M. Khouja, “The single-period (news-vendor) problem: literature review and suggestions for future research,” Omega, vol. 27, no. 5, pp. 537–553, 1999.

[23] F. W. Harris, “How many parts to make at once,” Factory, The Magazine of Management, vol. 10, no. 2, pp. 135–136, 1913.

[24] A. Federgruen and P. Zipkin, “An inventory model with limited production capacity and uncertain demands,” Naval Research Logistics, 1980s.

[25] S. Mannor and J. N. Tsitsiklis, “Mean-variance optimization in Markov decision processes,” in Proc. ICML, 2011.