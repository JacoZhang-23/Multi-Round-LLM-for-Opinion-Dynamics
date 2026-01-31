#!/usr/bin/env python3
"""
Analyze the extended network data
"""
import os
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
extended_network_csv = os.path.join(base_dir, "data", "input", "workplace_36013030400w1_extended_network.csv")
population_csv = os.path.join(base_dir, "data", "input", "workplace_36013030400w1_extended_population.csv")

print("=" * 80)
print("Analyzing Extended Network Data")
print("=" * 80)

# 读取扩展网络
df = pd.read_csv(extended_network_csv)
print(f'\n📊 Network Statistics:')
print(f'  Total edges: {len(df)}')
print(f'\n  Relation types:')
for rel, count in df['Relation'].value_counts().items():
    print(f'    {rel}: {count} edges')

# 获取所有唯一节点
all_nodes = set(df['source_reindex'].unique()) | set(df['target_reindex'].dropna().astype(int).unique())
print(f'\n  Total unique nodes in network: {len(all_nodes)}')

# 读取人口数据
pop_df = pd.read_csv(population_csv)
workplace_members = pop_df[pop_df['is_workplace_member'] == True]
print(f'\n👥 Population Statistics:')
print(f'  Total population rows: {len(pop_df)}')
print(f'  Workplace members: {len(workplace_members)}')

# 检查有多少workplace成员在网络中
workplace_reindexes = set(workplace_members['reindex'].values)
nodes_in_network = workplace_reindexes & all_nodes
print(f'\n🔗 Network Coverage:')
print(f'  Workplace members in network: {len(nodes_in_network)} / {len(workplace_members)}')
print(f'  Non-workplace nodes in network: {len(all_nodes - workplace_reindexes)}')

# 分析每种关系类型
print(f'\n📈 Detailed Analysis by Relation Type:')
for rel_type in ['wk', 'hh', 'sm']:
    rel_df = df[df['Relation'] == rel_type]
    if len(rel_df) > 0:
        rel_nodes = set(rel_df['source_reindex'].unique()) | set(rel_df['target_reindex'].dropna().astype(int).unique())
        workplace_in_rel = workplace_reindexes & rel_nodes
        print(f'  {rel_type.upper()} relations:')
        print(f'    Edges: {len(rel_df)}')
        print(f'    Unique nodes: {len(rel_nodes)}')
        print(f'    Workplace members involved: {len(workplace_in_rel)}')

print("\n" + "=" * 80)
print("✅ Analysis Complete")
print("=" * 80)
