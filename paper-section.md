# 用大语言模型替代 ABM 规则的交互式智能体（LAIDSim 样例模型）

English title: LLM-Driven Agents in ABM: From Hard Rules to Dialogue, Reflection, and Decision

本小节介绍一个将大语言模型（LLM）嵌入到基于主体的模型（ABM）中的样例系统（LAIDSim，LLM-Aided Influence and Decision Simulation）。我们以“疫苗接种”情境为例，用 LLM 代替传统 ABM 中基于规则的行为函数，让智能体通过“对话—反思—决策”的流程进行交互与演化。

---

### 案例引导：从设定到决策的单体轨迹（Agent 13）

为便于读者快速把握机制，我们以一次代表性运行中的 Agent 13 为例，给出“设定 → 网络属性 → 互动 → 自我反思 → 决策”的完整链条（数据来源：`simulation_20251021_152111/`）。

- 设定（State Init）
  - 档案：profile="A person with average characteristics.", age=37
  - 初始信念（LLM/VADER）：约 0.965 / 0.965（支持接种）
  - 更新强度：α = 0.5（来自 `config.py`）

- 网络属性（Social Graph）
  - 拓扑：小世界（`NETWORK_TYPE = "small_world"`，`CONNECTION_PROB = 0.15`，N=20）
  - 边权：每条边随机赋权 ∈ [0.1, 1.0]，用于加权聚合对话影响（见 `tools.create_network`）

- 互动与对话（Tick 0 的一个片段）
  - 与邻居 Agent 16 展开 2×2 轮自然语言对话（对话文本见报告 Top-1）。
  - 对话后，Agent 13 进行开放式反思并自评（JSON）：
    - 反思摘录：“I'm still uncertain about vaccination, but I've gained a better understanding of the scientific consensus and the importance of credible sources.”
    - 自评得分：从 ≈ 0.965 回落至 0.000（保守化到“不确定”）

- 决策（Vaccination）
  - 规则：若未接种，则以 `max(0, belief)` 作为接种概率进行采样。
  - 该轮因自评为 0.0，接种概率为 0；随后的多邻居互动与加权更新推动信念上扬（运行结果显示该主体的最终信念=1.0，并完成接种）。

这个案例体现了本模型的关键思想：用“语言互动 → 反思自评 → 网络加权聚合”替代硬编码规则。自评可能在早期出现“保守化回撤”（向 0 靠近），随后在多次互动与网络结构的共同作用下逐步收敛到更稳定的行动倾向（如接种）。

## 1. 模型总体工作流（Workflow）

- 输入（Profiles & Config）
  - 通过 LLM 并行生成 n 个智能体的人物档案（背景简述、年龄等），必要时使用默认回退档案（`tools.generate_profiles`）。
  - 加载仿真参数与 API 设置（智能体数量、步数、网络类型与密度、LLM 端点等）。

- 初始化（Model Init）
  - 创建 `VaxSimulationModel` 实例：构造 `mesa` 调度器与社会网络（小世界/无标度/随机，`tools.create_network`）。
  - 为每个智能体创建 `VaxAgent`，设定初始信念 `belief`（LLM通道）与 `belief_vader`（情感通道），并保存历史轨迹。
  - 以 5% 概率初始化“高信念支持者”（belief=1.0，未立即接种），用于检验传播与同伴影响。

- 交互（Per Step, Async）
  - 每个步长中，每个智能体仅与其网络邻居进行多轮自然语言对话（约 4–5 次 API 调用/邻居）。
  - 对话内容经清洗以去除 Qwen 的 `<think>` 隐式推理标签；不设置 `max_tokens`，先完整接收再清洗与解析，提升有效对话率。
  - 对话结束后通过“开放式反思+JSON 自评”提取该智能体当前自我立场分数（`belief_score ∈ [-1,1]`）。

- 更新（Belief Update & Vaccination）
  - 仅基于“有效对话”（自评 JSON 有效）计算加权平均信念变化：权重来自社会网络边的影响力（0.1–1.0）。
  - 以系数 α（`agent_alpha`）控制更新强度；并行维护两条信念轨（LLM 自评 vs. VADER 情感）。
  - 接种决策：若未接种，则以 `max(0, belief)` 作为接种概率进行 Bernoulli 采样；接种后两条信念即时置为 1.0。

- 记录与导出（Logging & Export）
  - 模型层指标用 `mesa.DataCollector` 记录；主体层数据用手工日志保证时序正确。
  - 每步结束导出至 `data/output/simulation_*/`，并生成可视化与分析报告。

---

## 2. 关键设定（Model Settings）

- 仿真规模与结构
  - 智能体数：`N_AGENTS`（默认 20，可配置）
  - 步长：`MAX_STEPS`（默认 5，可配置）
  - 社会网络：`NETWORK_TYPE`（small_world/scale_free/random），连接概率 `CONNECTION_PROB`
  - 网络边权：区间 [0.1, 1.0]，表示影响力强弱

- 智能体状态（VaxAgent）
  - `belief`：基于 LLM 自评的立场分数（-1 反对 ~ +1 支持）
  - `belief_vader`：基于 VADER 情感分析的分数
  - `is_vaccinated`：是否已接种，`tick_vaccinated` 记录时间
  - 历史：`belief_history` / `belief_vader_history`

- LLM 调用策略
  - 模型：`Qwen/Qwen3-8B`（本地端点，OpenAI 兼容接口）
  - 不限制 `max_tokens`，用正则 3–4 步清洗 `<think>` 标签（完整对、半截、残留单标记）
  - 对话结束以“自然语言+JSON 自评”获取 `summary_sentence` 与 `belief_score`

- 信念更新（行为规则被 LLM 替代的方式）
  - 传统 ABM：硬编码规则 f(state, neighbor) → new_state
  - 本模型：对话 → 反思 → 自评（LLM 输出）→ 加权聚合（网络权重）→ 更新
  - 更新公式（示意）：`belief_{t+1} = clip(belief_t + α · weighted_mean(Δbelief from valid dialogues))`

---

## 3. 代码组织（Code Architecture in `src/`）

- `agent.py`
  - `VaxAgent`：
    - 与邻居进行多轮自然语言对话（异步，顺序清洗响应）
    - 通过开放式反思 + JSON 自评提取 `belief_score`
    - 维护 LLM 与 VADER 两条并行信念轨
    - 接种决策与时序化的 `step/advance` 更新

- `model.py`
  - `VaxSimulationModel`：
    - 初始化：生成档案、创建智能体、构建网络
    - `async_actions()` 并发对话；使用进度条跟踪 API 调用
    - 数据收集：模型层用 `DataCollector`，主体层手工日志 `agent_data_log`
    - 导出：所有对话、主体档案、模型指标、主体时序数据

- `tools.py`
  - 文本处理与 JSON 提取：鲁棒的 `<think>` 清洗与多策略解析
  - VADER 情感分析（自动下载词典）
  - 网络构建：小世界/无标度/随机（带随机边权）
  - 异步生成主体档案（失败回退）

- `analysis.py`
  - 高级可视化：
    - 综合趋势图（LLM/VADER + 接种率，双轴对齐、无置信区间）
    - 比较折线与分布矩阵（在 `main.py` 中调用）
    - 影响力散点图（差异 vs. 变化，带回归与统计）
  - 报告：最具影响力对话文本导出

- `main.py`
  - 入口：加载配置、运行模型、导出数据、调用可视化与分析

- `config.py`
  - 存放 API 与仿真参数（如 `N_AGENTS`、`MAX_STEPS`、`CONNECTION_PROB`、`NETWORK_TYPE`、`AGENT_ALPHA` 等）

---

## 4. 输出产物（Outputs in `data/output/simulation_*/`）

- `model_data.csv`（模型层指标，逐步时间序列）
  - `Vaccination_Rate`：接种率（0–1）
  - `Average_Belief_LLM`：平均 LLM 自评信念
  - `Average_Belief_VADER`：平均 VADER 情感信念
  - `Belief_Std_Dev_LLM` / `Belief_Std_Dev_VADER`：两条轨迹的离散度

- `agent_data.csv`（主体层指标，逐步 × 主体）
  - `Step, AgentID, Belief_LLM, Belief_VADER, Is_Vaccinated`

- `all_dialogues.json`（全量对话日志，有效性标记）
  - 结构：`{tick, interlocutors, initial_beliefs, exchanges[], elicited_summary, elicited_self_score, elicited_sentiment_score, is_valid}`

- `agent_profiles.json`（主体静态档案 + 信念轨迹）
  - 结构：`{agent_id, profile, age, belief_history[], belief_vader_history[]}`

- `visualizations/`（可视化 PNG）
  - `comprehensive_trends.png`：综合趋势（LLM/VADER + 接种率，双轴对齐、Y∈[-0.5,1]）
  - `comparative_belief_trends.png`：LLM vs VADER 平均轨迹对比
  - `belief_distribution_matrix.png`：起始与终止分布（2×2）
  - `influence_scatter_plot.png`：影响力散点与回归

- `most_impactful_dialogues_report.txt`（文本报告）
  - Top-N 对话详单、统计摘要（均值、方差、极值）与原文片段

---

## 5. 决策替代与方法学贡献（What the LLM Replaces）

- 从“手写规则”到“语言交互”：
  - 传统规则：`belief_{t+1} = f(belief_t, neighbor_state, noise, …)`
  - LLM 机制：`(dialogue → reflection → self-score) × neighbors → weighted aggregation → update`
- 优势：
  - 行为的可解释性（对话文本 + 反思摘要）
  - 跨情境可迁移（更换提示而非重写规则）
  - 支持多通道比较（LLM 自评 vs. 情感通道）
- 风险与对策：
  - LLM 冗余推理标签（<think>）→ 多步正则清洗
  - 截断风险 → 移除 `max_tokens`，先收全再处理
  - 噪声自评 → 仅采用“有效对话”参与更新，且用网络权重平滑

---

## 6. 运行与复现（Reproducibility）

- 快速运行
```bash
cd src
python main.py
```

- 输出位置
```
src/data/output/simulation_YYYYMMDD_HHMMSS/
```

- 依赖（关键）
  - `mesa`, `openai`（兼容端点）, `aiohttp`, `networkx`, `nltk`（VADER）, `matplotlib`, `seaborn`, `pandas`, `numpy`, `tqdm`

---

## 7. 小结

本样例展示了如何在 ABM 中以 LLM 替代硬编码的行为规则，用“对话—反思—自评—聚合”的链条驱动主体状态演化。模型不仅提供可解释的过程数据（对话与反思文本），还支持并行评价通道（LLM vs. VADER），并与社会网络结构相结合，适合用于研究信息影响、态度更新与集体行为的机制。

---

## 8. 结果展示与解释（Results and Interpretation）

以下结果来自一次代表性运行：`src/data/output/simulation_20251021_152111/visualizations/` 与同目录下的 `most_impactful_dialogues_report.txt`。

### 8.1 综合趋势图（comprehensive_trends.png）

- 文件：`comprehensive_trends.png`
- 设定：左轴为平均信念（LLM 自评与 VADER 情感，范围 -0.5–1.0），右轴为接种率（与左轴对齐起点）。
- 观测要点（该次运行的读数）：
  - LLM 平均信念：约 0.146 → 0.642（随步长上行）
  - VADER 平均信念：约 0.146 → 0.774（整体高于 LLM，抬升更快）
  - 接种率：0.0% → 75.0%（单调上升，与信念上扬一致）
- 解释：
  1) 两条信念轨迹整体上升，说明“对话—反思—自评”的机制在该网络/设定下，推动了更积极的疫苗态度；
  2) VADER 曲线通常领先 LLM，一种可能的机制是情感表达的正向性在对话文本中更早显化，而 LLM 自评更保守；
  3) 接种率的爬升与信念上扬同向，符合“belief → action（prob=max(0, belief)）”的决策设定；
  4) 我们移除了置信区间、收窄了 Y 轴至 [-0.5, 1.0] 并对齐双轴起点，使趋势更聚焦、可比。

方法学启示：若需要更稳健或更慢的上扬，可按需：降低 `agent_alpha`、提高“有效对话”的准入门槛、或在更新中加入“惯性/记忆（如动量或上限步长）”。
图表设定:
双Y轴设计：左轴代表平均信念得分（Average Belief Score），范围为 -0.4 至 1.0，用于展示“LLM 自我评估信念”（蓝色实线）与“VADER 情感信念”（紫色实线）。右轴代表接种率（Vaccination Rate），范围与左轴对齐，用于展示“疫苗接种率”（绿色虚线）的进程。
X轴：表示仿真步长（Simulation Step），从 0 到 5。
核心观测 (Key Observations):
统一的初始状态: 在第 0 步，系统初始化时，LLM 信念与 VADER 信念的平均值完全相同（约 0.15），代表群体初始持中性偏积极的态度。此时，接种率为 0，符合仿真开始前的设定。
信念的同步增长与显著分化: 随着仿真推进，所有三条曲线均呈现强劲的上升趋势。一个关键现象是，代表对话文本情感的 VADER 信念曲线（紫色）从第 1 步起就持续高于代表深度反思的 LLM 信念曲线（蓝色），且增长斜率更陡。这表明，在交互中，智能体语言表达的情感正向性（emotional positivity）比其经过反思后形成的认知认同（cognitive agreement）发展得更快、更强。
行动紧随信念: 接种率曲线（绿色）的增长轨迹与两条信念曲线高度相关，从 0% 稳步攀升至 75%。这有力地验证了模型的核心假设：信念是行动的前导（接种概率 = max(0, belief)），群体态度的积极转变直接驱动了集体行为的发生。
后期趋于饱和: 在第 4 步到第 5 步之间，VADER 信念曲线趋于平缓，几乎变为水平，表明对话中的情感表达可能已达到一个饱和点。与此同时，LLM 信念仍在缓慢上升，而接种率的增速也略有放缓。这暗示了当表层的情感共识形成后，深层认知转变的边际效应可能会减弱。
解释与推论 (Interpretation and Inferences):
正向反馈循环的形成: 该图清晰地描绘了一个正向社会影响的反馈循环：初始的积极对话促进了信念的提升，提升的信念又促使更多智能体接种疫苗，而已接种或持高信念的智能体可能在后续对话中发表更坚定的支持言论，从而进一步强化了整个群体的积极趋势。
“情感”与“认知”的双通道差异: VADER 与 LLM 信念的分离，揭示了两种不同的态度测量维度。VADER 更像一个快速、表层的**“情感温度计”，捕捉对话中的即时情绪；而 LLM 的“反思-自评”机制则代表了一个更慢、更审慎的“认知评估器”**。这种差异是传统 ABM 中难以捕捉的，体现了 LLM 在模拟复杂人类决策过程中的潜力。
机制的有效性验证: 整个趋势验证了“对话 → 反思 → 信念更新 → 行为决策”这一由 LLM 驱动的核心链条的有效性。它成功地将微观的语言交互，转化为宏观、可预测的群体动态演化。
方法学启示:
本次运行展示的信念增长速度和最终接种率并非固定不变。通过调整关键参数，如信念更新强度（agent_alpha）、网络密度与类型（CONNECTION_PROB, NETWORK_TYPE）或引入对立观点智能体，可以模拟出信念增长更缓慢、出现意见分歧甚至极化的不同社会情境。这为研究信息传播与干预策略提供了灵活的实验平台。

### 8.2 影响力散点图（influence_scatter_plot.png）

- 文件：`influence_scatter_plot.png`
这张图深入揭示了单次对话中，个体间信念差异如何转化为观点变化，是理解模型微观影响机制的关键窗口。
图表定义:
横轴 (X-axis): 信念差异 (Belief Disparity)，计算方式为邻居的初始信念 - 智能体自身的初始信念。正值意味着智能体与一个更支持疫苗的邻居对话；负值则意味着邻居的态度更消极。该轴衡量了社会比较的压力或潜在影响力。
纵轴 (Y-axis): 对话后的信念变化 (Belief Change After Dialogue)，代表智能体在单次对话后，通过“反思-自评”产生的信念得分变化量。正值表示态度变得更积极，负值表示更消极。
数据点: 每个点代表一次有效的对话交互（共 344 次）。点的颜色与纵轴对应，从深红（大幅负向变化）到亮绿（大幅正向变化），提供了变化的视觉强度。
回归分析: 图中包含一条线性回归线（蓝色）及其置信区间，直观展示了“信念差异”与“信念变化”之间的总体趋势。皮尔逊相关系数 (Pearson r = 0.540) 量化了这种中等强度的正相关关系。
核心观测 (Key Observations):
显著的正相关趋势: 回归线清晰地显示，信念差异越大（邻居越积极），智能体自身的信念也越倾向于向积极方向变化。r = 0.540 的值表明，这种关系不仅是可见的，而且在统计上是中等强度的。这符合社会影响理论中的同化效应（assimilation effect）：个体会倾向于向其互动对象的观点靠拢。
影响力的非对称性:
正向影响 (右上象限): 当智能体与比自己更积极的邻居（X > 0）互动时，绝大多数数据点都落在 Y > 0 的区域，即发生了积极的信念变化。这说明模型中的智能体很容易被更积极的观点“向上拉”。
负向影响与抵抗 (左侧象限): 当邻居比自己更消极时（X < 0），情况更为复杂。虽然存在一些负向变化（左下象限），但也有大量点分布在 Y ≈ 0 的水平线附近，甚至有少量正向变化。这表明智能体对负面影响表现出了一定的抵抗性 (resistance)，或对话未能动摇其原有立场。
极端负向变化的特殊性: 在左下角存在几个显著的离群点，代表了信念的大幅下降（Belief Change 接近 -1.0）。深入分析这些案例（参考 most_impactful_dialogues_report.txt）会发现，它们通常发生在一个初始信念极高的智能体（如 belief ≈ 1.0）与一个中立或略微消极的邻居对话之后。这并非简单的“被说服”，而更可能是 LLM 在反思环节中，面对不同观点时，从“绝对确信”退回至一个更审慎、不确定的状态，这体现了 LLM 自我评估中的保守性或中立化倾向。
解释与推论 (Interpretation and Inferences):
对话是信念趋同的主要驱动力: 该图从微观层面证实了 8.1 节宏观趋势的来源。群体平均信念的上升，正是由一次次主要发生在右上象限的、微小但持续的积极影响累积而成的。
模型内嵌的“积极偏见”: 当前的模型设定和提示（Prompt）可能内在地鼓励了一种开放和积极的交流氛围。智能体似乎更愿意接受和采纳支持性的论点，而对否定性的观点则表现出筛选和抵抗。这或许反映了现实世界中，在非对抗性情境下，人们倾向于避免冲突并寻求共识的倾向。
LLM 的认知复杂性: 散点图的非完美线性关系和极端离群点的存在，恰恰说明了 LLM 替代硬编码规则的价值。简单的规则（如信念变化 = α * 信念差异）会产生一条完美的直线，而 LLM 通过模拟对话和反思，引入了语境依赖性、非线性反应和内在认知偏见（如对极端立场的审慎回调），使得影响过程更加丰富和真实。
研究意义与方法学贡献:
从“是什么”到“为什么”: 此图超越了“群体信念在上升”的简单描述，深入解释了“为什么以及如何”上升。它将宏观动态（集体行为）与微观互动（个体对话）联系起来，为社会现象的机制探索提供了有力证据。
量化“语言的说服力”: 本研究通过“对话-反思-自评”的流程，将非结构化的自然语言交互，成功转化为可量化的信念变化数据，并与经典的社会网络变量（信念差异）进行关联分析。这为计算社会科学中研究语言影响力的课题提供了一个创新的方法论框架。
指导模型迭代与理论探索: 图中揭示的“积极偏见”和“极端回调”等现象，为下一步的模型优化提供了明确方向。例如，可以通过修改 Agent 的性格设定（Profile）或反思提示（Prompt），引入更具“批判性”或“固执”的智能体，来探索不同沟通风格对群体极化的影响。这使得模型不仅是一个模拟工具，更是一个理论探索的虚拟实验室。

### 8.3 最具影响力对话（most_impactful_dialogues_report.txt）

- 统计摘要（该次运行）：
  - 总对话数：344；有效对话：344；无效：0
  - 平均改变：+0.0470；标准差：0.1569
  - 最大正向：+0.3366；最大负向：-0.9645

- Top-3 个案（均为强负向变化）：
  1) Dialogue #1（Tick 0）：Agent 13（初始 0.965）与 Agent 16（0.020）对话后，自评回到 0.000，变化 -0.965；反思文本显示“仍不确定，但理解更多科学共识”，符合“从强支持回归不确定”的保守化机制；
  2) Dialogue #2（Tick 0）：Agent 5（0.952）与 Agent 3（-0.099）后回到 0.000（-0.952）；
  3) Dialogue #3（Tick 2）：Agent 16（0.105）与 Agent 14（-0.515）后降至 -0.500（-0.605）。

解读与建议：
  - 这些强负向样本集中在早期步长，说明在信念尚未收敛前，对话能触发较大幅度的保守化回撤；
  - 如果研究目的不希望出现“瞬间归零”，可：
    - 在 belief 更新时加入“最大步长限制”（如 |Δbelief| ≤ 0.3/步）；
    - 引入“信念惯性/遗忘率”，对大幅度变动进行平滑；
    - 在开放式反思提示中增加“如果仅是不确定，请在原有立场附近进行小幅调整”的引导；
    - 根据边权（信任度）或关系强度，对个别邻居的影响设上限。

### 8.4 其他可视化

- `belief_distribution_matrix.png`：起始与终止分布（2×2），可观察到分布向右移与峰度变化，用于补充整体趋向。
信念分布演化对比 (belief_distribution_matrix.png)
这张 2x2 的矩阵图提供了关于群体信念结构演化的关键快照，它通过直方图展示了在仿真开始（Step 0）和结束（Step 5）时，由“LLM 自我评估”和“VADER 情感分析”两种方法测得的信念分布情况。
图表定义:
矩阵布局:
上行: 展示了仿真第 0 步的初始信念分布。
下行: 展示了仿真第 5 步的最终信念分布。
左列: 基于 LLM“反思-自评”的信念分数。
右列: 基于 VADER 文本情感分析的信念分数。
坐标轴:
横轴 (X-axis): 信念分数（Belief Score），范围从 -1.0（强烈反对）到 1.0（强烈支持）。
纵轴 (Y-axis): 智能体数量（Count）。
核心观测 (Key Observations):
一致的起点: 在第 0 步（上行图），LLM 和 VADER 的信念分布完全相同。分布形态较为扁平且分散，集中在 [-0.75, 0.25] 区间内，显示出群体初始态度的多样性和不确定性，整体略微偏向中立。
向右的压倒性迁移: 对比第 0 步和第 5 步（上行 vs. 下行），两种测量方法都记录到了一个从中间地带向最右端（belief = 1.0）的大规模迁移。这清晰地展示了群体共识的形成过程。
最终形成的“共识峰”: 在第 5 步（下行图），两个分布图的最右侧都出现了一个巨大的柱状峰，代表了大量智能体（约 15 名，占总数 75%）的信念值精确地等于 1.0。这直接反映了模型的核心规则：智能体一旦决定接种疫苗，其信念值就会被设定为 1.0。因此，这个“共识峰”在很大程度上就是“已接种人群”的画像。
未接种者的分布差异: 两种方法最核心的差异体现在对剩余未接种智能体的刻画上：
LLM 自我评估 (左下图): 在最终状态下，仍有少数智能体的信念分布在负值区域（如 -0.75 附近），表明通过深度反思，一部分个体即使在积极的社交环境中，依然维持或形成了审慎乃至反对的立场。
VADER 情感分析 (右下图): 在最终状态下，几乎没有任何智能体的情感得分为负。剩余未接种者的情感得分主要集中在 [0, 0.25] 这个略微积极的区间。这表明，即使一些智能体在认知层面尚未完全接受疫苗，但他们在对话中表露出的情感已经普遍变得积极或至少是中性的。
解释与推论 (Interpretation and Inferences):
共识的形成机制: 分布的右移和“共识峰”的隆起，直观地展示了社会影响下的**信念级联（belief cascade）**效应。积极的互动（如 8.2 所示）不断累积，推动个体越过决策阈值，采取行动（接种），而行动又进一步强化了其信念，并固化了群体共识。
“情感先行，认知滞后”: VADER 与 LLM 在最终分布上的差异，生动地揭示了“情感”与“认知”的脱钩与不同步。在社会交往中，人们更容易在语言情绪上达成一致（避免冲突、表达友好），这使得 VADER 分布迅速地“净化”了负面情绪。然而，深度的认知信念（LLM 自评）的转变则更为复杂和缓慢，允许了更多异质性和顽固性的存在。
模型的现实主义: 这种双通道的差异性，恰恰是该模型超越传统规则模型的地方。它不仅模拟了“人们最终做了什么”，还揭示了他们“口头怎么说”和“内心怎么想”之间可能存在的微妙差别，这为理解宣传、说服和公共舆论的复杂性提供了更丰富的视角。
方法学启示:
同时采用基于词典的情感分析（VADER）和基于大型语言模型的深度反思（LLM Self-Score）来测量智能体状态，能够提供一种立体的、多维度的观测。这种方法论上的设计，使得模型能够捕捉到那些在单一指标下会被掩盖的社会心理现象，例如群体表面情绪的快速趋同与深层信念的缓慢演化。

---

小结：综合趋势显示“向支持—接种上升”的系统性演化；散点图与最具影响力对话揭示了微观层面的强影响案例与保守化机制。若研究目标需要更稳健的更新过程，可从提示工程与更新算子两端进行“温和化”设计（限制步长、加入惯性、增强信任权重约束）。

###框架图

## Agent 交互流程图（含 Prompt、更新与决策）

```mermaid
flowchart TD
  A[开始：Agent i 在 Tick t 准备与邻居交互] --> B[获取邻居列表 N(i)]
  B --> C{对每个邻居 j 依次处理}
  C --> D1[初次对话（由邻居 j 发起）\nSystem: "简短自然回应(2-3句)"\nUser: Person B 背景=profile(j), 态度=get_attitude(belief_j)\n任务: 开启简短对话]
  D1 --> D2[LLM 调用 → 三步清洗 <think>\n保存 exchange(B→A)]
  D2 --> E1{对话循环 k=1..3}
  E1 -->|k 为奇数，A 说话| E2A[Prompt(A 响应)\nSystem: "简短自然回应(2-3句)"\nUser: A.profile(i)，当前观念=get_attitude(tick_belief_i)\n任务: 自然回应 2-3 句]
  E1 -->|k 为偶数，B 说话| E2B[Prompt(B 续谈)\nSystem: "简短自然回应(2-3句)"\nUser: B.profile(j)，当前观念=get_attitude(belief_j)\n任务: 自然续谈 2-3 句]
  E2A --> E3[LLM 调用 → 清洗 <think>\n保存 exchange(A→B 或 B→A)]
  E2B --> E3
  E3 --> E1
  E1 -->|完成| F1[开放式反思 + JSON 自评\nSystem: "仅返回有效 JSON"\nUser: 先写2-3句反思 → 再给 JSON{summary_sentence, belief_score∈[-1,1]}\n附：显示对话前 score=tick_belief_i]
  F1 --> F2[LLM 调用 → 增强清洗 <think>\nextract_json_from_response()]
  F2 --> G{JSON 有效且含 summary_sentence 与 belief_score?}
  G -->|否| H1[标记对话无效 is_valid=False\n该对话不参与更新]
  G -->|是| H2[标记有效 is_valid=True；计算单次增量\nΔ_llm = belief_score - tick_belief_i\nΔ_vader = vader(summary) - tick_belief_vader_i\n权重 w = edge_weight(i,j)]
  H1 --> I
  H2 --> I
  I --> C
  C -->|全部邻居处理完| J{存在有效对话?}
  J -->|否| K[pending_belief = belief_i\npending_belief_vader = belief_vader_i]
  J -->|是| L[按权重加权平均\npending_belief = clip(tick_belief_i + α·avg_w(Δ_llm))\npending_belief_vader = clip(tick_belief_vader_i + α·avg_w(Δ_vader))]
  L --> S[进入该 Tick 的行为决策 (step)]
  K --> S

  subgraph 当步行为决策 step()
    S --> N{若尚未接种:\n p = max(0, belief_i)\n 抽样 u~U(0,1)}
    N -->|u < p| O[接种成功:\n is_vaccinated=True\n tick_vaccinated=t\n belief=1.0\n belief_vader=1.0]
    N -->|否则| P[保持未接种]
  end

  O --> M
  P --> M

  M[状态推进 advance():\n若存在 pending_* → 覆盖当前信念并写历史\n(已接种者信念保持为 1.0)] --> Q[结束：Agent i 完成 Tick t]
```

### 节点要点与顺序说明
- 对话阶段（多轮）：严格使用简短自然语言风格的 System 指令；每次调用后都对 <think> 做“三步清洗”（完整对、半截、残留标签）。
- 反思与自评：先自然语言反思，再给 JSON（只返回 JSON 的 System 约束），提取 summary_sentence 与 belief_score。
- 有效性：只有含 summary_sentence 和 belief_score 的对话计入更新；其它跳过。
- 更新规则：按网络边权 w 对“单次对话增量”求加权平均，再用 α 缩放并裁剪到 [-1,1]，写入 pending_belief 与 pending_belief_vader。
- 决策（step）：基于当前 belief（未应用 pending）抽样接种；接种即刻把两条信念置为 1.0。
- 推进（advance）：将 pending_* 写回并记录历史。已接种者在本 tick 应保持信念=1.0（实现上需确保 pending_* 为 1.0 或在 advance 前后不被覆盖）。

## 关键 Prompt 片段（可直接复用）

- 对话 System（通用，2-3 句）
  - You are a helpful assistant. Provide brief, natural responses (2-3 sentences).

- 初次对话（由邻居 B 发起）
  - User: 
    - You are Person B having a conversation about vaccination.
    - Your background: {neighbor.profile}
    - Your current attitude: You {get_attitude_from_belief(neighbor.belief)}
    - Start a brief conversation with Person A about vaccination (2-3 sentences).

- 轮流对话（A 说话）
  - User:
    - You are Person A responding.
    - Your background: {agent.profile}
    - Your current view: You {get_attitude_from_belief(agent.tick_belief)}
    - Respond naturally to what Person B said (2-3 sentences).

- 轮流对话（B 说话）
  - User:
    - You are Person B continuing the conversation.
    - Your background: {neighbor.profile}
    - Your current view: You {get_attitude_from_belief(neighbor.belief)}
    - Continue the conversation naturally (2-3 sentences).

- 反思 + JSON 自评（强制 JSON）
  - System:
    - You are a helpful assistant that provides responses in valid JSON format only. Do not include any text outside the JSON object.
  - User（摘要）:
    - Reflect in 2-3 sentences on how this conversation affected your thinking.
    - Then output JSON:
      {
        "summary_sentence": "...",
        "belief_score": < -1.0 to 1.0 >
      }
    - Your view BEFORE the conversation: You {get_attitude_from_belief(agent.tick_belief)} (score: {agent.tick_belief:.2f})

## 补充提示
- 清洗策略：每次 LLM 返回后都执行 3 步正则清洗 <think>（完整对、半截起始、残留标签），再做 JSON 提取与有效性判断。
- 仅邻居交互：只与网络邻居对话，边权作为影响力用于加权聚合。
- 双轨记录：并行维护 LLM 自评与 VADER 两条信念轨，更新和历史分别记账。
- 决策先于推进：先根据“当前信念”作接种判定，再把 pending_* 写回，确保接种者信念最终保持 1.0。
