# Workplace Network Simulation - 使用说明

## 更新内容

本次更新将仿真系统改为使用真实的工作场所(workplace)数据：

### 数据来源

1. **人口数据**: `data/input/workplace_36013030400w1_extended_population.csv`
   - 包含工作场所所有成员的详细信息
   - 字段：年龄、性别、教育程度、职业、收入、家庭规模等
   - 筛选条件：`is_workplace_member == True`（30人）

2. **网络数据**: `data/input/workplace_36013030400w1_internal_network.csv`
   - 包含工作场所内部的社交网络连接
   - 格式：`source_reindex` 和 `target_reindex` 的配对关系
   - 总共69条边（连接）

### 主要改动

#### 1. tools.py 新增功能
- `load_workplace_profiles()`: 从CSV文件加载真实人口数据和网络
- 自动生成完整的profile描述（包含年龄、性别、职业、教育、收入等信息）
- 使用真实的工作场所网络结构

#### 2. model.py 改进
- 新增参数：`use_workplace_data`, `population_csv`, `network_csv`
- 支持两种模式：
  - **Workplace模式**: 使用CSV数据（30个真实agent）
  - **LLM模式**: 使用原有的LLM生成profile

#### 3. main.py 自动检测
- 自动检测CSV文件是否存在
- 如果存在：使用workplace数据（30个agent）
- 如果不存在：回退到LLM生成模式

### 网络特征

从测试结果可以看到：
- **节点数**: 30个agent
- **边数**: 69条连接
- **平均度**: 4.60（每个agent平均有4.6个连接）
- **度分布**: 最小2个连接，最大8个连接

### 运行方式

#### 测试数据加载（推荐先运行）
```bash
cd src
python test_workplace_data.py
```

#### 运行完整仿真
```bash
cd src
python main.py
```

### Profile示例

原始CSV数据会被自动转换为完整的profile描述，例如：

```
Agent 0:
  Profile: A 61-year-old female working in Service. Has Bachelor's Degree 
           level education. Has health insurance coverage. Lives with 1 
           family member(s).
  Age: 61
  Occupation: Service
  Education: Bachelor's Degree
  Income: $13,000
  Network connections: 2 neighbors
```

### 保持的功能

以下功能保持不变：
- ✅ 多轮对话机制
- ✅ LLM驱动的belief更新
- ✅ VADER情感分析对比
- ✅ 网络可视化
- ✅ 数据导出和分析

### 技术细节

1. **Reindex映射**: CSV中的`reindex`字段被映射到连续的agent_id (0-29)
2. **教育分类**: 根据education数值自动分类（16=高中，20=本科，21+=研究生）
3. **职业分类**: 根据occupation代码映射到7个职业类别
4. **网络权重**: 每条边被赋予0.5-1.0的随机权重

### 文件结构

```
src/
├── main.py              # 主入口（已更新：自动检测CSV）
├── model.py             # 模型类（已更新：支持workplace数据）
├── agent.py             # Agent类（保持不变）
├── tools.py             # 工具函数（新增：load_workplace_profiles）
├── config.py            # 配置文件（新增：USE_WORKPLACE_DATA注释）
├── analysis.py          # 分析模块（保持不变）
└── test_workplace_data.py  # 新增：数据加载测试

data/input/
├── workplace_36013030400w1_extended_population.csv
└── workplace_36013030400w1_internal_network.csv
```

### 常见问题

**Q: 如何切换回LLM生成模式？**
A: 重命名或移动CSV文件即可，系统会自动回退到LLM模式。

**Q: 可以调整agent数量吗？**
A: Workplace模式固定为30人（CSV数据）。LLM模式可在config.py调整N_AGENTS。

**Q: 网络结构可以修改吗？**
A: Workplace模式使用固定的真实网络。LLM模式可调整CONNECTION_PROB和NETWORK_TYPE。

**Q: 对话轮次可以调整吗？**
A: 可以，修改config.py中的MAX_STEPS参数（默认5轮）。
