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

3. 3. Feature - based Explanation with SHAP

to 100 representative samples for SHAP estimation. During

masking, KernelSHAP marginalizes missing features using behavior is analyzed using XRL methods: RDX and MSX the backg round distribution rather than replacing them with to explain decisions in the reward space, and SHAP to

analyze the influence of state features, creating a direct

zero, since zero may correspond to meaningful operational

linkage between behavior and explanation. This procedure

conditions such as no inventory or no demand. Because

enables a direct connection between the agent’s decision -

inventory, demand, waste, and capacity - related variables

making behavior and the corresponding explanations

may be correlated, SHAP values are i nterpreted as throughout the evaluation process.

associative feature attributions rather than causal effects.

4. 1.3. Classical Inventory Baseline

The use of a structured background distribution helps reduce unrealistic out - of - distribution perturbations, but it does not To provide an operational reference for evaluating the

DRL agents, this study introduces a classical bas e - stock

fully remove the influence of feature dependence.

policy as a non - learning inventory - control baseline. This

Therefo re, SHAP is used as a feature - space interpretation

baseline is used to examine whether the learned DRL

tool and is complemented by perturbation - based faithfulness

policies are competitive with a standard inventory

evaluation. The main implementation settings of SHAP are replenishment rule. For each product 𝑖, the target summarized in Table 2. inventory level is defined as:

Table 2. SHAP implementation configuration.

𝑆 𝑖 = 𝑑 ˉ 𝑖 + 𝑘 𝜎 𝑖

Component Configuration where 𝑑

ˉ 𝑖 and 𝜎 𝑖 denote the mean and standard deviation of

historical demand for product 𝑖, respectively. The

SHAP variant KernelSHAP using shap.KernelExplainer

parameter 𝑘 is a safety factor that controls the trade - off

Explained output for

Selected - action Q - value 𝑄 (𝑠, 𝑎 ∗) between stockout risk and inventory cost. At each time

DQN Explained output for step 𝑡, the repl enishment quantity is computed as:

Selected - action policy output 𝜋 (𝑎 ∗ ∣ 𝑠)

A2C_mod

∗

Background distribution Representative normalized inventory states 𝑢

𝑖 = max (0, 𝑆 𝑖 − 𝑥 𝑖 (𝑡))

200 generated states, subsampled to 100 where 𝑥 𝑖 (𝑡) is the current inventory level of product 𝑖. The

Background size

representative samples resulting replenishment quantity is clipped to the Marginalization using the background

Masking strategy normalized capacity range and mapped to the nearest

distribution

available discrete repleni shment action. In this study, 𝑘 is

Aggregated features Inventory, Demand, Waste

selected from:

Top - k SKU - level/system - level SHAP

Additional analysis

analysis

𝑘 ∈ {0. 5, 1. 0, 1. 5, 2. 0, 2. 5}

Interpreted as associative attribution, not

Correlated features

causal effect

The base - stock policy is evaluated under the same testing cycles, capacity constraints, and evaluation metrics as

4. Evaluation

DQN and A2C_mod. The comparison reports total

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

*Note on comparability (Task 14): SHAP for DQN is computed on softmax(Q) and for A2C_mod on policy π(a*|s) - two quantities on different scales (Q unbounded vs π constrained [0,1] and interdependent via softmax). Hence, direct magnitude and stability comparisons between agents in Table 7b/7c are descriptive and qualitative, not quantitative. Sensitivity analysis on common-target logits (pre-softmax, 10/20/50 states, Supplementary Task 13) shows Jaccard only 0.25-0.29 between n_states and between logits vs softmax/π, supporting the need for qualitative comparison.*

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

*Mean|SHAP| values: DQN 0.00281-0.00253, A2C_mod 0.00044-0.00027; 3-4x tied baseline 0.00079. Figure 11 (Top-20, 660 features) is primary; Figure 9 (3 features) is overview. *Comparison qualitative due to different targets (softmax Q vs π); see Task 13 common-target analysis in Supplementary.*

4. 5.4. Consistency of Top-k Features across EASY/MEDIUM/HARD

For Top-20, DQN shows Jaccard 1.00 and Spearman 1.00 across all scenario pairs (stable, 8 SKUs), while A2C_mod shows Jaccard 0.29-0.33 and Spearman 0.20 (context-dependent shift, core Top-5 remains). Full k=10/20/50 results (18 rows) in Supplementary task10-9/outputTask10-11_overlap.csv. *Comparison qualitative as noted in Section 3.3.3.*

4. 5.5. Why Dominant Products Dominate Top-20 - Case Analysis

Top-demand SKUs are SKU57 (22.6), SKU81 (17.2), SKU108 (15.5) - not in Top-8 SHAP (r=0.04). Dominant SKUs are low-volume high-CV (1.1-1.46) and small capacity (7-28), e.g., SKU64 CV 1.46 cap 7, SKU163 CV 1.27 cap 12 - volatile, stockout-sensitive. SKU215 is exception - high-volume (8.68) with large capacity 81. Full 36 SKUs in Supplementary task10-9/outputTask10-12_case_analysis.csv. SHAP is associative.

Figure 9 (3 features) is overview; Figure 11 (660 features) is primary. I_Micro(f_j)=1/(N·A) Σ|φ_{j,a}^{(n)}|.

4.5.6. Common Target Sensitivity (Task 13 - Logits, 10/20/50 states)

To test comparability, SHAP was computed on pre-softmax logits as common target (DQN q_values linear and A2C_mod layer4 before softmax) with sensitivity across n_states=10/20/50 (100 background, PartitionExplainer). We chose n_states=10/20/50 to test sampling robustness: 10 states is a fast feasibility test (~25 min for 5 configs), while 20 and 50 increase statistical power but cost 2-5 hours for 90 explainers. If Jaccard between n=10 vs 50 remained high (>0.8), 10-state would be representative; our results show Jaccard only 0.25-0.29 for DQN and 0.25-0.667 for A2C_mod (e.g., DQN EASY 10-20 0.250, A2C_mod EASY 10-20 0.667; full 18 rows in task11-9/outputTask11_n_states_sensitivity.csv), indicating Top-20 is sensitive to sample size and 10-state alone is not representative, hence reporting all three levels. Logits Top-5 differs from softmax(Q)/π Top-5 (DQN Jaccard 0.667, A2C_mod 0.25 for EASY 10-state), supporting qualitative comparison in Table 7b/7c. Full 360-row Top-20 logits in task11-9/outputTask11_common_target.csv.

4.5.7. Actor vs Critic Sensitivity (Task 15 - 10-state EASY/MEDIUM/HARD)

Sensitivity on A2C_mod with three outputs (π, logits, V(s)) on 10-state EASY/MEDIUM/HARD shows Jaccard Top-20 only 0.25-0.29 between π vs logits and π vs V(s) across all scenarios (e.g., EASY A2C pi Top-5 [175,90,164,119,93] vs logits [93,119,90,71,108] overlap 2/5, Jaccard 0.25; EASY pi vs V(s) Top-5 [175,90,164,119,93] vs V(s) [175,71,90,164,119] Jaccard 0.25), indicating ranking differs across targets despite same Sales group, supporting qualitative comparison. Full 9 rows in task11-9/outputTask11_sensitivity.csv.

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

4. 7. Faithfulness Evaluation of Explanations

particular, when 𝜆 ≥ 1. 5, the MEDIUM and HARD

To complement the descriptive comparison metrics in

scenarios show substantially higher action flip rates under

Section 4.6, this study further evaluates the faithfulness of

MSX - guided masking than under random maski ng. In the

the generated explanations through perturbation - based tests.

HARD scenario, masking MSX - identified reward

The goal is to examine whether the reward components or

components changes the selected action in nearly all

state features identified as important by RDX/MSX and

evaluated states, whereas random masking produces a much

SHAP are functionally related to the agents’ ou tputs. If

lower action flip rate. This indicates that MSX identifies

masking explanation - identified components or features

reward components that are func tionally important to the

causes larger changes than masking random or less relevant

agent’s decision.

ones, the explanation can be considered more faithful to the model behavior. In contrast, the EASY scenario shows almost no action

changes under either MSX - guided or random masking. This

4. 7.1 Faithfulness Evaluation for RDX/MSX

is expected because the reward components are less

For RDX/MSX, the evaluation is conducted by masking conflicting in simple environments, and many action s have the reward components selected by MSX and observing similar reward values. Overall, the RDX/MSX faithfulness whether the agent’s selected action or value estimate results suggest that the explanation is more informative in changes. Given the decomposed value function: scenarios with stronger reward conflicts, especially in

MEDIUM and HARD settings.

𝑄 𝑡𝑜𝑡𝑎𝑙 = 𝑄 𝑠𝑒𝑟𝑣𝑖𝑐𝑒 + 𝑄 ℎ 𝑜𝑙𝑑𝑖𝑛𝑔 + 𝑄 𝑤𝑎𝑠𝑡𝑒 + 𝑄 𝑜𝑟 𝑑𝑒𝑟

∗ 4.7.2 Faithfulness Evaluation for SHAP

the MSX - identified component set 𝑆 is removed, and the remaining value is recomputed. The action flip rate of MSX For SHAP, f aithfulness is evaluated by perturbing the top - masking (𝐴𝐹 𝑅 𝑀𝑆𝑋) is then compared with a random masking ranked features identified by SHAP and measuring the

resulting changes in the model outputs. Three masking reduces Q - values or policy probabilities. Therefore, the strategies are compared: MoRF, Random, and LeRF. MoRF proposed XRL framework is supported by faithfulness masks the most relevant SHAP features first, LeR F masks evidence, particularly under more difficult inventory the least relevant features first, and Random serves as a sc enarios.

baseline. For DQN, the output change is measured by the

5. Conclusion

relative decrease in the selected - action Q - value:

∗ ∗ 𝑄 (𝑠, 𝑎) − 𝑄 This study proposes a unified Explainable 𝑚𝑎𝑠𝑘𝑒𝑑 (𝑠, 𝑎)

Δ 𝑄 (𝑘) =

∗ Reinforcement Learning (XRL) framework for large - scale

∣ 𝑄 (𝑠, 𝑎) ∣ + 𝜖

inventory management, aiming to systematically analyze the

For A2C_mod, the output change is measured by the

decision - making mechanisms of two representative DRL

decrease in the selected - action probability:

architectures, DQN and A2C_ mod. By integrating Reward Difference Explanation (RDX), Minimal Sufficient

∗ ∗

Δ 𝜋 (𝑘) = 𝜋 (𝑎 ∣ 𝑠) − 𝜋 𝑚𝑎𝑠𝑘𝑒𝑑 (𝑎 ∣ 𝑠) Explanation (MSX), and SHAP, the framework enables The action switching rate (𝐴𝑆𝑅) is also used to measure multi - level interpretation, including trade - offs in the reward how often the final selected action changes after fea ture space, compact justification sets derived from MSX, and the

contri bution levels of state features. Experimental results

masking. Figure 20 presents the perturbation curves for

reveal clear structural differences between the two

SHAP faithfulness verification under MoRF, Random, and

architectures. A2C_mod demonstrates the ability to

LeRF masking strategies.

reallocate the importance of objectives as environmental complexity increases, indicating flexible policy adaptation.

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

[12] “Reinforcement Learning: An Introduction | Guide books | ACM Digital Library.” Accessed: Mar. 05,