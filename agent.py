# agent.py

"""
Defines the VaxAgent class for the simulation.
This is the final, definitive version with robust dialogue and elicitation logic.
"""

import mesa
import numpy as np
import random
import aiohttp
import re
from typing import List, Dict
from openai import AsyncOpenAI
import logging

from tools import get_attitude_from_belief, get_sentiment_score, extract_json_from_response

# 配置日志，禁用 OpenAI 的详细 INFO 日志
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Import API configuration from config
from config import API_URL, API_KEY, MODEL_NAME, BELIEF_DISTRIBUTION_TYPE, BELIEF_MEANS, BELIEF_STD

# Initialize OpenAI client with configuration from config.py
client = AsyncOpenAI(
    base_url=API_URL,
    api_key=API_KEY,
)


REFUSAL_PATTERNS = [
    "i can't", "i cannot", "i'm sorry", "i am sorry", "unable to", "i won't",
    "i will not", "can't assist", "cannot assist", "not able to", "i must decline",
    "refuse", "policy", "not appropriate", "can't help with that"
]


def is_refusal(text: str) -> bool:
    if not text:
        return True
    lowered = text.strip().lower()
    return any(p in lowered for p in REFUSAL_PATTERNS)


class VaxAgent(mesa.Agent):
    """An agent with parallel belief states for comparative analysis."""

    def __init__(self, unique_id: int, model, profile_data: Dict, alpha: float):
        # Initialize Mesa Agent with unique_id and model
        super().__init__(unique_id, model)
        # Set additional attributes
        self.profile = profile_data['profile']
        self.age = profile_data['age']
        self.alpha = alpha

        # 使用固定的初始belief（如果提供）
        if hasattr(model, 'fixed_initial_beliefs') and model.fixed_initial_beliefs is not None:
            initial_belief = model.fixed_initial_beliefs[unique_id]
        else:
            # 否则随机生成
            mu = BELIEF_MEANS.get(BELIEF_DISTRIBUTION_TYPE, 0.0)
            initial_belief = float(np.clip(np.random.normal(mu, BELIEF_STD), -1.0, 1.0))
        
        self.belief = initial_belief
        self.belief_vader = initial_belief

        self.is_vaccinated = False
        self.tick_vaccinated = -1

        self.dialogue_history = []
        self.belief_history = [self.belief]
        self.belief_vader_history = [self.belief_vader]
        self.tick_belief = self.belief
        
        # Scheduling and locking
        self.is_locked = False  # Lock status for exclusive dialog participation
        self.dialog_memory = {}  # Per-neighbor dialog memory: {neighbor_id: [messages]}

    def get_neighbors(self) -> List['VaxAgent']:
        """Get list of neighbors from the network graph."""
        neighbor_ids = list(self.model.network.neighbors(self.unique_id))
        return [self.model.schedule.agents[nid] for nid in neighbor_ids]

    async def conduct_dialogue_with_neighbor(self, session: aiohttp.ClientSession, neighbor: 'VaxAgent') -> Dict:
        """Conducts a multi-turn dialogue with robust reflection."""
        agent_a, agent_b = self, neighbor
        dialogue_record = {
            'tick': self.model.schedule.steps, 'interlocutors': [agent_a.unique_id, agent_b.unique_id],
            'initial_beliefs': {'self': agent_a.tick_belief, 'neighbor': agent_b.belief},
            'exchanges': [],
        }
        messages = []

        # ========== 第一次 API 调用：初始对话 ==========
        prompt = f"""You are Person B having a conversation about vaccination. 
Your background: {agent_b.profile}
Your current attitude: You {get_attitude_from_belief(agent_b.belief)}

Start a brief conversation with Person A about vaccination (2-3 sentences)."""
        
        api_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant in a fictional social simulation. "
                    "Provide brief, natural responses (2-3 sentences). "
                    "Avoid medical advice; speak in general terms. Do not refuse to answer."
                )
            },
            {"role": "user", "content": prompt}
        ]
        
        try:
            completion = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=api_messages,
                temperature=0.7
                # 不限制 max_tokens，完整接收响应后再处理
            )
            # 提取内容并立即清洗（三步增强清洗）
            content = completion.choices[0].message.content
            
            # 调试：打印原始内容长度
            if '<think>' in content.lower():
                print(f"🔍 Agent {self.unique_id}: Raw content contains <think> tag (length: {len(content)})")
            
            # 三步清洗法
            response = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            response = re.sub(r'<think>.*', '', response, flags=re.DOTALL).strip()  # 清除残留的不完整开始标签
            response = re.sub(r'</?think>', '', response).strip()
            
            # 调试：验证清洗后的结果
            if '<think>' in response.lower():
                print(f"❌ Agent {self.unique_id}: Cleaning FAILED! Still contains <think> (length: {len(response)})")
            else:
                if len(content) != len(response):
                    print(f"✅ Agent {self.unique_id}: Cleaned {len(content) - len(response)} chars")
            
            if is_refusal(response):
                response = "I can share general thoughts without giving medical advice. It's a complex topic, and people weigh safety, effectiveness, and trust differently. I'm open to discussing it in general terms."

            if hasattr(self.model, 'api_call_counter'):
                self.model.api_call_counter.update(1)
        except Exception as e:
            print(f"LLM API Exception (Agent {self.unique_id}): {e}")
            response = "I understand your perspective on this matter."
        
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": response})
        dialogue_record['exchanges'].append({'speaker_id': agent_b.unique_id, 'message': response})

        # ========== 对话循环：3轮交互 ==========
        for turn in range(1, 4):
            is_self_turn = turn % 2 != 0
            current_speaker = agent_a if is_self_turn else agent_b

            # 构建提示
            if is_self_turn:
                prompt = f"""You are Person A responding.
Your background: {current_speaker.profile}
Your current view: You {get_attitude_from_belief(current_speaker.tick_belief)}

Respond naturally to what Person B said (2-3 sentences)."""
            else:
                prompt = f"""You are Person B continuing the conversation.
Your background: {current_speaker.profile}
Your current view: You {get_attitude_from_belief(current_speaker.belief)}

Continue the conversation naturally (2-3 sentences)."""

            messages.append({"role": "user", "content": prompt})
            
            # API 调用
            api_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant in a fictional social simulation. "
                        "Provide brief, natural responses (2-3 sentences). "
                        "Avoid medical advice; speak in general terms. Do not refuse to answer."
                    )
                }
            ] + messages
            
            try:
                completion = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=api_messages,
                    temperature=0.7
                    # 不限制 max_tokens，完整接收响应后再处理
                )
                # 提取内容并立即清洗（三步增强清洗）
                content = completion.choices[0].message.content
                
                # 三步清洗法：处理完整和不完整的 <think> 标签
                response = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                response = re.sub(r'<think>.*', '', response, flags=re.DOTALL).strip()  # 清除残留的不完整开始标签
                response = re.sub(r'</?think>', '', response).strip()
                
                if is_refusal(response):
                    response = "I can share general thoughts without giving medical advice. It's a nuanced topic, and people balance different concerns. I'm open to discussing it respectfully."

                if hasattr(self.model, 'api_call_counter'):
                    self.model.api_call_counter.update(1)
            except Exception as e:
                print(f"LLM API Exception (Agent {self.unique_id}): {e}")
                response = "I understand your perspective on this matter."
            
            messages.append({"role": "assistant", "content": response})
            dialogue_record['exchanges'].append({'speaker_id': current_speaker.unique_id, 'message': response})

        # ========== 最后一次 API 调用：开放式自评（更自然的反思）==========
        elicitation_instruction = f"""After this conversation, please reflect on your current view about vaccination.

First, in 2-3 sentences, describe how this conversation affected your thinking (if at all).

Then, provide a JSON object with your updated view:
{{
  "summary_sentence": "One sentence describing your current view after the conversation",
  "belief_score": <a number between -1.0 and 1.0 representing your current stance>
}}

Guidelines for belief_score:
- Think about where you stand NOW on the spectrum from strongly against (-1.0) to strongly support (+1.0)
- Be honest about your genuine view, not what you think is "correct"
- Consider: +1.0=strongly support, +0.5=support, 0=uncertain, -0.5=against, -1.0=strongly against

Your view BEFORE the conversation: You {get_attitude_from_belief(agent_a.tick_belief)} (score: {agent_a.tick_belief:.2f})

Please share your reflection and provide the JSON:"""

        messages.append({"role": "user", "content": elicitation_instruction})
        
        # JSON 请求的 API 调用
        api_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant in a fictional social simulation. "
                    "Provide responses in valid JSON format only. Do not include any text outside the JSON object. "
                    "Avoid medical advice; speak in general terms. Do not refuse to answer."
                )
            }
        ] + messages
        
        try:
            completion = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=api_messages,
                temperature=0.7
                # 不限制 max_tokens，完整接收响应后再处理
            )
            # 提取内容并立即清洗
            content = completion.choices[0].message.content
            
            # 调试：JSON 请求的清洗验证（关键！）
            if '<think>' in content.lower():
                print(f"🔍 JSON Request - Agent {self.unique_id}: Raw content contains <think> (length: {len(content)})")
                print(f"   Preview: {content[:150]}...")
                # 检查是否有结束标签
                if '</think>' not in content.lower():
                    print(f"   ⚠️  WARNING: No closing </think> tag found!")
            
            # 增强的清洗：处理完整和不完整的 <think> 标签
            # 1. 先清除完整的 <think>...</think> 对
            final_response = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            # 2. 清除残留的开始标签：两种策略
            #    a) 如果后面有 JSON（{），只删除到 { 之前
            final_response = re.sub(r'<think>.*?(?=\{)', '', final_response, flags=re.DOTALL).strip()
            #    b) 如果没有 JSON，删除整个残留的 <think> 标签及其后所有内容
            final_response = re.sub(r'<think>.*$', '', final_response, flags=re.DOTALL).strip()
            # 3. 最后清除任何残留的单独 <think> 或 </think> 标签
            final_response = re.sub(r'</?think>', '', final_response).strip()
            
            # 调试：验证清洗后的结果
            if '<think>' in final_response.lower():
                print(f"❌ JSON Request - Agent {self.unique_id}: Cleaning FAILED!")
                print(f"   After cleaning: {final_response[:150]}...")
            else:
                if len(content) != len(final_response):
                    print(f"✅ JSON Request - Agent {self.unique_id}: Cleaned {len(content) - len(final_response)} chars")
            
            if is_refusal(final_response):
                final_response = (
                    "{\"summary_sentence\": \"I feel uncertain and prefer to think about this more in general terms.\", "
                    f"\"belief_score\": {agent_a.tick_belief:.2f}}}"
                )

            if hasattr(self.model, 'api_call_counter'):
                self.model.api_call_counter.update(1)
        except Exception as e:
            print(f"LLM API Exception (Agent {self.unique_id}): {e}")
            final_response = (
                "{\"summary_sentence\": \"I feel uncertain and prefer to think about this more in general terms.\", "
                f"\"belief_score\": {agent_a.tick_belief:.2f}}}"
            )

        # 解析 JSON
        elicited_data = extract_json_from_response(final_response)

        # Verification: retry once if JSON invalid or refusal detected
        if not elicited_data or 'summary_sentence' not in elicited_data or 'belief_score' not in elicited_data:
            verification_prompt = (
                "Return ONLY a JSON object with keys: summary_sentence (string) and belief_score (number between -1 and 1). "
                "No extra text. Keep it brief and neutral."
            )
            verification_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant in a fictional social simulation. "
                        "Provide responses in valid JSON format only. Do not include any text outside the JSON object."
                    )
                },
                {"role": "user", "content": verification_prompt}
            ]
            try:
                verification_completion = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=verification_messages,
                    temperature=0.2
                )
                verification_content = verification_completion.choices[0].message.content
                verification_clean = re.sub(r'<think>.*?</think>', '', verification_content, flags=re.DOTALL).strip()
                verification_clean = re.sub(r'<think>.*', '', verification_clean, flags=re.DOTALL).strip()
                verification_clean = re.sub(r'</?think>', '', verification_clean).strip()

                if is_refusal(verification_clean):
                    verification_clean = (
                        "{\"summary_sentence\": \"I feel uncertain and prefer to think about this more in general terms.\", "
                        f"\"belief_score\": {agent_a.tick_belief:.2f}}}"
                    )

                if hasattr(self.model, 'api_call_counter'):
                    self.model.api_call_counter.update(1)

                elicited_data = extract_json_from_response(verification_clean)
            except Exception as e:
                print(f"LLM Verification Exception (Agent {self.unique_id}): {e}")

        if elicited_data and 'summary_sentence' in elicited_data and 'belief_score' in elicited_data:
            dialogue_record.update({
                'elicited_summary': elicited_data['summary_sentence'],
                'elicited_self_score': np.clip(float(elicited_data['belief_score']), -1.0, 1.0),
                'elicited_sentiment_score': get_sentiment_score(elicited_data['summary_sentence']),
                'is_valid': True  # 标记为有效对话
            })
        else:
            # 没有有效的 summary，设置为无效对话
            dialogue_record.update({
                'elicited_summary': None,
                'elicited_self_score': None,
                'elicited_sentiment_score': None,
                'is_valid': False  # 标记为无效对话
            })
        return dialogue_record

    # ... The rest of agent.py (update_belief_from_dialogues, step, advance) remains unchanged ...
    async def update_belief_from_dialogues(self, session: aiohttp.ClientSession):
        """
        Conduct dialogues ONLY with network neighbors and calculate pending belief changes.
        Invalid dialogues (without proper summary) are excluded from belief updates.
        """
        # 清理旧的pending beliefs
        self.pending_belief = None
        self.pending_belief_vader = None
        
        # 已接种的agent不再更新belief（但仍可作为neighbor参与别人的对话）
        if self.is_vaccinated:
            self.pending_belief = self.belief  # 保持当前belief不变
            self.pending_belief_vader = self.belief_vader
            return

        # 保存当前状态
        self.tick_belief = self.belief
        tick_belief_vader = self.belief_vader
        
        # 只与网络中的邻居进行对话
        neighbors = self.get_neighbors()
        if not neighbors:
            self.pending_belief = self.belief
            self.pending_belief_vader = self.belief_vader
            return

        belief_changes_llm, belief_changes_vader, weights = [], [], []

        for neighbor in neighbors:
            dialogue_record = await self.conduct_dialogue_with_neighbor(session, neighbor)
            self.dialogue_history.append(dialogue_record)

            # 只处理有效的对话（有 summary 的对话）
            if dialogue_record.get('is_valid', False):
                final_elicited_belief_llm = dialogue_record['elicited_self_score']
                change_llm = final_elicited_belief_llm - self.tick_belief
                belief_changes_llm.append(change_llm)

                final_elicited_belief_vader = dialogue_record['elicited_sentiment_score']
                change_vader = final_elicited_belief_vader - tick_belief_vader
                belief_changes_vader.append(change_vader)

                # 从网络边获取权重
                edge_weight = self.model.network[self.unique_id][neighbor.unique_id].get('weight', 0.5)
                weights.append(edge_weight)

        # 如果有有效的对话，计算加权平均变化
        if weights and len(belief_changes_llm) > 0:
            weighted_mean_change_llm = np.average(belief_changes_llm, weights=weights)
            self.pending_belief = np.clip(self.tick_belief + self.alpha * weighted_mean_change_llm, -1.0, 1.0)

            weighted_mean_change_vader = np.average(belief_changes_vader, weights=weights)
            self.pending_belief_vader = np.clip(tick_belief_vader + self.alpha * weighted_mean_change_vader, -1.0, 1.0)
        else:
            # 没有有效对话，保持原信念不变
            self.pending_belief = self.belief
            self.pending_belief_vader = self.belief_vader

    def step(self):
        if not self.is_vaccinated:
            vaccination_prob = max(0, self.belief)
            if random.random() < vaccination_prob:
                self.is_vaccinated = True
                self.tick_vaccinated = self.model.schedule.steps
                # 接种后设置pending_belief为1.0，将在advance()中更新
                self.pending_belief = 1.0
                self.pending_belief_vader = 1.0

    def advance(self):
        # Update belief from pending values if they exist
        if hasattr(self, 'pending_belief') and self.pending_belief is not None:
            self.belief = self.pending_belief
        if hasattr(self, 'pending_belief_vader') and self.pending_belief_vader is not None:
            self.belief_vader = self.pending_belief_vader
        
        # Always record current belief to history
        self.belief_history.append(self.belief)
        self.belief_vader_history.append(self.belief_vader)