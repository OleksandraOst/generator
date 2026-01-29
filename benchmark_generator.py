import sys
import json
import time
import matplotlib.pyplot as plt
from typing import List, Optional
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

sys.stdout = TeeLogger("benchmark_log.txt")

# ==========================================
# 2. DATA SCHEMAS (Pydantic)
# ==========================================
class BenchmarkItem(BaseModel):
    topic: str
    question: str
    difficulty_intent: int = Field(ge=1, le=10)

class Evaluation(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0")
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
        self.alpha = alpha  # EMA smoothing factor

    def get_novel_question(self) -> BenchmarkItem:
        history_str = ", ".join(self.history[-15:])
        # Difficulty scaling: if EMA is high, push for harder questions
        target_diff = 10 if self.ema_score > 0.8 else 5
        
        prompt = (
            f"Generate a novel, complex reasoning question. Target difficulty: {target_diff}/10. "
            f"Avoid these previous topics: {history_str}. "
            "Return ONLY a JSON object with keys: 'topic', 'question', 'difficulty_intent'."
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a benchmark generator. You MUST output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return BenchmarkItem.model_validate_json(response.choices[0].message.content)

    def get_answer(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content

    def judge_answer(self, question: str, answer: str) -> Evaluation:
        prompt = (
            f"Evaluate the following response to the question. \n"
            f"Question: {question}\nAnswer: {answer}\n"
            "Score from 0.0 (wrong) to 1.0 (perfect). Return ONLY JSON with 'score' and 'reasoning'."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an objective judge. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return Evaluation.model_validate_json(response.choices[0].message.content)

    def update_ema(self, new_score: float):
        if self.ema_score == 0:
            self.ema_score = new_score
        else:
            self.ema_score = (self.alpha * new_score) + (1 - self.alpha) * self.ema_score

# ==========================================
# 4. EXECUTION LOOP & PLOTTING
# ==========================================
# Configuration for Google Gemini (OpenAI-compatible)
SYSTEM_CONFIG = {
    "api_key": "YOUR_GEMINI_API_KEY",
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    "model_name": "gemini-2.5-flash" 
}

bench = EvolvingBenchmark(**SYSTEM_CONFIG)
raw_scores = []
ema_trend = []

print("Starting Self-Evolving Benchmark Loop...")

try:
    for i in range(10): # Set number of iterations
        print(f"\n--- Iteration {i+1} ---")
        try:
            # Step 1: Generate
            item = bench.get_novel_question()
            bench.history.append(item.topic)
            print(f"[?] Topic: {item.topic}")
            print(f"[?] Question: {item.question[:100]}...")

            # Step 2: Answer
            answer = bench.get_answer(item.question)

            # Step 3: Judge
            eval_result = bench.judge_answer(item.question, answer)
            
            # Step 4: Update Stats
            bench.update_ema(eval_result.score)
            raw_scores.append(eval_result.score)
            ema_trend.append(bench.ema_score)

            print(f"[*] Score: {eval_result.score} | EMA: {bench.ema_score:.4f}")
            print(f"[*] Reasoning: {eval_result.reasoning[:120]}...")

        except Exception as e:
            print(f"[!] Iteration failed: {e}")
            if "429" in str(e):
                print("[!] Rate limit hit. Waiting 60s...")
                time.sleep(60)

        # Pause to respect Gemini Free Tier (15 Requests Per Minute)
        time.sleep(5)

finally:
    # Generate final visualization
    if raw_scores:
        plt.figure(figsize=(12, 6))
        plt.scatter(range(1, len(raw_scores)+1), raw_scores, color='blue', alpha=0.4, label='Raw Scores')
        plt.plot(range(1, len(ema_trend)+1), ema_trend, color='red', linewidth=2, label='EMA Trend')
        plt.title('LLM Self-Evolving Benchmark (Gemini 2.5 Flash)')
        plt.xlabel('Iteration')
        plt.ylabel('Score (0.0 - 1.0)')
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig("benchmark_results_plot.png")
        print("\n[+] Success: Final plot saved to 'benchmark_results_plot.png'")
        plt.show()

print("\nBenchmark Session Complete. Log saved to 'benchmark_log.txt'.")