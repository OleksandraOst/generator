import sys
import json
import time
import matplotlib.pyplot as plt
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI

# ==========================================
# 1. LOGGER SETUP (Captures Terminal to File)
# ==========================================
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# This ensures every print() goes to both the screen and the file
sys.stdout = TeeLogger("benchmark_log.txt")

# ==========================================
# 2. DATA SCHEMAS (Pydantic)
# ==========================================
class BenchmarkItem(BaseModel):
    topic: str
    question: str
    difficulty_intent: int = Field(ge=1, le=10)

class Evaluation(BaseModel):
    score: float = Field(ge=0, le=1)
    reasoning: str

# ==========================================
# 3. CORE EVOLUTION LOGIC
# ==========================================
class EvolvingBenchmark:
    def __init__(self, api_key: str, base_url: str, model_name: str, alpha: float = 0.3):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name
        self.history: List[str] = []
        self.ema_score: float = 0.0
        self.alpha = alpha 

    def get_novel_question(self) -> BenchmarkItem:
        history_str = ", ".join(self.history[-15:])
        # Utility: Scale difficulty based on performance (EMA)
        target_diff = 9 if self.ema_score > 0.8 else 5
        
        prompt = (
            f"Generate a novel reasoning question (Difficulty: {target_diff}/10). "
            f"Avoid these topics: {history_str}. "
            "Return ONLY JSON with keys: 'topic', 'question', 'difficulty_intent'."
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "You are a JSON generator."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return BenchmarkItem.model_validate_json(response.choices[0].message.content)

    def run_iteration(self):
        # 1. Generate
        item = self.get_novel_question()
        self.history.append(item.topic)
        print(f"[?] Topic: {item.topic}")

        # 2. Answer
        answer_resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": item.question}]
        )
        answer = answer_resp.choices[0].message.content

        # 3. Judge
        eval_resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "Judge answer 0-1. Return JSON: {'score': float, 'reasoning': str}"},
                      {"role": "user", "content": f"Q: {item.question}\nA: {answer}"}],
            response_format={"type": "json_object"}
        )
        eval_result = Evaluation.model_validate_json(eval_resp.choices[0].message.content)
        
        # 4. Update EMA: S_t = alpha*Y_t + (1-alpha)*S_{t-1}
        if self.ema_score == 0:
            self.ema_score = eval_result.score
        else:
            self.ema_score = (self.alpha * eval_result.score) + (1 - self.alpha) * self.ema_score
            
        print(f"[*] Score: {eval_result.score} | EMA: {self.ema_score:.4f}")
        return eval_result.score

# ==========================================
# 4. EXECUTION LOOP
# ==========================================
SYSTEM_CONFIG = {
    "api_key": "YOUR_GEMINI_API_KEY",
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    "model_name": "gemini-2.5-flash"
}

bench = EvolvingBenchmark(**SYSTEM_CONFIG)
raw_scores, ema_trend = [], []

try:
    for i in range(10): 
        print(f"\n--- Iteration {i+1} ---")
        try:
            score = bench.run_iteration()
            raw_scores.append(score)
            ema_trend.append(bench.ema_score)
        except Exception as e:
            print(f"[!] Error: {e}")
            if "429" in str(e):
                print("[!] Rate limit reached. Sleeping 60s...")
                time.sleep(60)

        # MANDATORY: 12s sleep to keep within 15 requests/min limit
        time.sleep(12)

finally:
    if raw_scores:
        plt.figure(figsize=(10, 5))
        plt.scatter(range(1, len(raw_scores)+1), raw_scores, color='blue', alpha=0.3, label='Raw Scores')
        plt.plot(range(1, len(ema_trend)+1), ema_trend, color='red', linewidth=2, label='EMA Trend')
        plt.title('LLM Evolution Benchmark')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # SAVE BEFORE SHOW
        plt.savefig("benchmark_results_plot.png")
        print("\n[+] SUCCESS: Plot saved as 'benchmark_results_plot.png'")
        plt.show()