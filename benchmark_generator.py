import sys
import json
import time
import matplotlib.pyplot as plt
from typing import List, Any
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI

# ==========================================
# 1. THE "TEE" LOGGER (Writes to Screen AND File)
# ==========================================
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout  # Save the original screen output
        self.log = open(filename, "a", encoding="utf-8") # Open/Create the text file
    
    def write(self, message):
        self.terminal.write(message)  # Write to screen
        self.log.write(message)       # Write to file
        self.log.flush()              # Force save immediately so you don't lose data
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# This command makes every 'print()' in the script go to BOTH locations
sys.stdout = TeeLogger("benchmark_log.txt")

# ==========================================
# 2. DATA SCHEMAS (Validation & Cleaning)
# ==========================================
class BenchmarkItem(BaseModel):
    topic: str
    question: str
    difficulty_intent: int = Field(ge=1, le=10)

    @field_validator('difficulty_intent', mode='before')
    @classmethod
    def parse_messy_int(cls, v: Any) -> int:
        """Cleans up cases where the AI says '5/10' instead of just 5."""
        if isinstance(v, str):
            v = v.split('/')[0].split(' ')[0].strip()
            return int(v)
        return v

class Evaluation(BaseModel):
    score: float = Field(ge=0, le=1)
    reasoning: str

# ==========================================
# 3. BENCHMARK ENGINE
# ==========================================
class EvolvingBenchmark:
    def __init__(self, api_key: str, base_url: str, model_name: str, alpha: float = 0.3):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name
        self.history: List[str] = [] # Keeps track of old topics
        self.ema_score: float = 0.0  # The moving average score
        self.alpha = alpha 

    def get_novel_question(self) -> BenchmarkItem:
        history_str = ", ".join(self.history[-15:])
        prompt = (
            f"Generate a novel reasoning question. DO NOT repeat these topics: {history_str}. "
            "Return ONLY JSON: {'topic': string, 'question': string, 'difficulty_intent': int}."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return BenchmarkItem.model_validate_json(response.choices[0].message.content)

# ==========================================
# 4. RUNNING THE LOOP
# ==========================================
SYSTEM_CONFIG = {
    "api_key": "YOUR_GEMINI_API_KEY",
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    "model_name": "gemini-2.5-flash"
}

bench = EvolvingBenchmark(**SYSTEM_CONFIG)
raw_scores, ema_trend = [], []

print("--- BENCHMARK STARTED ---")
try:
    for i in range(5): # Running 5 iterations for this test
        print(f"\n>>> ITERATION {i+1} <<<")
        try:
            # STEP 1: GENERATE QUESTION
            item = bench.get_novel_question()
            bench.history.append(item.topic)
            print(f"QUESTION TOPIC: {item.topic}")
            print(f"THE QUESTION: {item.question}") # <--- NOW PRINTING QUESTION TEXT

            # STEP 2: GET ANSWER
            ans_resp = bench.client.chat.completions.create(
                model=bench.model,
                messages=[{"role": "user", "content": item.question}]
            )
            answer = ans_resp.choices[0].message.content
            print(f"AI ANSWER: {answer}") # <--- NOW PRINTING ANSWER TEXT

            # STEP 3: EVALUATE
            eval_resp = bench.client.chat.completions.create(
                model=bench.model,
                messages=[{"role": "system", "content": "Score 0.0-1.0. Return JSON: {'score': float, 'reasoning': str}"},
                          {"role": "user", "content": f"Q: {item.question}\nA: {answer}"}],
                response_format={"type": "json_object"}
            )
            eval_res = Evaluation.model_validate_json(eval_resp.choices[0].message.content)
            
            # Update data for plotting
            bench.ema_score = (bench.alpha * eval_res.score) + (1 - bench.alpha) * bench.ema_score
            raw_scores.append(eval_res.score)
            ema_trend.append(bench.ema_score)

            print(f"RESULT: Score={eval_res.score} | EMA={bench.ema_score:.4f}")
            print(f"JUDGE REASONING: {eval_res.reasoning}")

        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(12) # Stay under the rate limit!

finally:
    # Save and Show Plot
    if raw_scores:
        plt.plot(raw_scores, 'bo', label='Raw')
        plt.plot(ema_trend, 'r-', label='EMA')
        plt.savefig("benchmark_results_plot.png")
        print("\n[+] Plot saved as benchmark_results_plot.png")
        plt.show()