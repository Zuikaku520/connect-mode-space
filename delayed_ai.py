import numpy as np
import time
import requests
from typing import List, Dict
import matplotlib.pyplot as plt

# ================== 配置 ==================
LLM_BACKEND = "ollama"
OLLAMA_MODEL = "qwen2:7b"
STYLE_THRESHOLDS = {"high": 0.7, "medium": 0.3, "low": 0.0}

class DelayedMode:
    """增强版延迟模式：慢半拍、记仇、情绪持久"""
    def __init__(self, delay=2.0, tau=5.0, dt=0.2):
        self.delay = delay
        self.tau = tau
        self.dt = dt
        self.intensity = 0.0
        self.next_intensity = 0.0
        self.buffer = []          # (时间, 幅度, 事件文本)
        self.event_memory = []    # (时间, 文本, 幅度)
        self.name = "Delayed"
        self.history = []
    
    def inject(self, amplitude: float, event_text: str = ""):
        self.buffer.append((time.time(), amplitude, event_text))
    
    def update(self, steps=1):
        for _ in range(steps):
            now = time.time()
            total = 0.0
            remaining = []
            for t, amp, txt in self.buffer:
                if now - t >= self.delay:
                    total += amp
                    if txt and abs(amp) > 0.2:
                        self.event_memory.append((now, txt, amp))
                else:
                    remaining.append((t, amp, txt))
            self.buffer = remaining
            if total != 0:
                self.next_intensity += total
                self.next_intensity = np.clip(self.next_intensity, -1.0, 1.0)
            self.next_intensity *= np.exp(-self.dt / self.tau)
            self.intensity = self.next_intensity
            # 清理旧记忆
            self.event_memory = [(t, txt, a) for (t, txt, a) in self.event_memory if now - t < 180]
        self.history.append((time.time(), self.intensity))
        if len(self.history) > 100:
            self.history.pop(0)
    
    def get_style(self):
        if self.intensity > 0.7:
            return "情绪高涨，语句较长，使用感叹号"
        elif self.intensity > 0.3:
            return "情绪平稳积极"
        elif self.intensity > 0:
            return "情绪低落，话少"
        else:
            return "情绪冷淡或压抑，可能带讽刺或记仇"
    
    def get_active_memories(self, min_abs=0.2, max_age=180):
        now = time.time()
        active = []
        for t, txt, amp in self.event_memory:
            if now - t <= max_age and abs(amp) >= min_abs:
                sent = "负面" if amp < 0 else "正面"
                active.append(f"（{sent}，{int(now-t)}秒前）用户：{txt}")
        return active

def get_llm_response(prompt, system_msg):
    # 使用你原有的函数，确保有重试机制
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.8}
    }
    try:
        resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=30)
        data = resp.json()
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        elif "response" in data:
            return data["response"]
        else:
            return ""
    except:
        return ""

def main():
    mode = DelayedMode(delay=2.0, tau=5.0, dt=0.2)
    character_prompt = "你是星穹铁道角色银狼，性格傲娇，毒舌但内心关心人。"
    print("=== 增强版延迟模式（记仇、慢反应） ===")
    print("试试骂她‘笨蛋’，然后等几秒再说话，她会记仇很久。\n")
    
    while True:
        user_input = input("你: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        
        # 简单情感分析
        negative = ["笨蛋", "讨厌", "差劲", "笨", "蠢", "骂"]
        positive = ["喜欢", "爱", "棒", "对不起", "谢谢"]
        amp = 0.0
        event_text = ""
        for w in negative:
            if w in user_input:
                amp = -0.6
                event_text = f"骂我‘{w}’"
                break
        if amp == 0.0:
            for w in positive:
                if w in user_input:
                    amp = 0.4
                    event_text = f"夸我‘{w}’"
                    break
        
        if amp != 0:
            mode.inject(amp, event_text)
            print(f"[系统] 事件注入: {amp:+.1f} - {event_text} (延迟{int(mode.delay)}秒后生效)")
        
        # 推进时间：让延迟效果有机会出现
        mode.update(steps=10)  # 10*0.2=2秒，正好等于delay
        
        # 打印当前情绪强度
        print(f"[情绪强度: {mode.intensity:.2f}] 风格: {mode.get_style()}")
        
        # 获取记忆
        memories = mode.get_active_memories(min_abs=0.2, max_age=120)
        mem_str = "\n".join(f"- {m}" for m in memories) if memories else "无重要记忆"
        
        system_msg = f"""{character_prompt}
当前情绪强度: {mode.intensity:.2f} (负值表示不开心/记仇)
回复风格: {mode.get_style()}
近期记忆:
{mem_str}
请根据情绪强度和记忆，自然地回复。如果有负面记忆，可以表现出埋怨或冷淡。"""
        
        reply = get_llm_response(user_input, system_msg)
        if not reply:
            # 降级回复
            if mode.intensity < -0.3:
                reply = "……（沉默）哼。"
            elif mode.intensity < 0:
                reply = "懒得理你。"
            else:
                reply = "嗯。"
        print(f"银狼: {reply}\n")
        
        # 继续推进时间，模拟对话间隔（但不再绘图）
        mode.update(steps=5)
    
    # 退出后绘制完整情绪曲线
    if mode.history:
        times, intensities = zip(*mode.history)
        plt.figure(figsize=(10, 4))
        plt.plot([t - times[0] for t in times], intensities, 'b-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Emotional Intensity')
        plt.title('Emotional Intensity Over Time (Full Session)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.show()

if __name__ == "__main__":
    main()
    # 在 main() 函数末尾，退出循环后添加：