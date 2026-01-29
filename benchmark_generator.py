import os
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI

# Schema for the Generated Question
class BenchmarkItem(BaseModel):
    topic: str
    question: str
    difficulty_intent: int = Field(description="Intended difficulty 1-10")

# Schema for the Judge's Evaluation
class Evaluation(BaseModel):
    score: float = Field(description="Score from 0.0 to 1.0")
    reasoning: str



class EvolvingBenchmark:
    def __init__(self, api_key: str, base_url: str, model_name: str, alpha: float = 0.3):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name
        self.history: List[str] = []
        self.ema_score: float = 0.0
        self.alpha = alpha  # Smoothing factor for EMA

    def get_novel_question(self) -> BenchmarkItem:
        history_context = "\n".join(self.history[-10:]) # Reference last 10 items
        prompt = f"Generate a novel, complex logic or coding question. \nRecent topics: {history_context}"
        
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "system", "content": "You are a benchmark creator."},
                      {"role": "user", "content": prompt}],
            response_format=BenchmarkItem,
        )
        return completion.choices[0].message.parsed

    def get_answer(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content

    def judge_answer(self, question: str, answer: str) -> Evaluation:
        prompt = f"Question: {question}\nAnswer: {answer}\nGrade this answer accurately."
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "system", "content": "You are an objective judge."},
                      {"role": "user", "content": prompt}],
            response_format=Evaluation,
        )
        return completion.choices[0].message.parsed

    def update_ema(self, new_score: float):
        # EMA Formula: S_t = α * Y_t + (1 - α) * S_{t-1}
        if self.ema_score == 0:
            self.ema_score = new_score
        else:
            self.ema_score = (self.alpha * new_score) + (1 - self.alpha) * self.ema_score

    def run_iteration(self):
        # 1. Generate
        item = self.get_novel_question()
        self.history.append(item.topic)
        print(f"\n[?] New Question: {item.question[:100]}...")

        # 2. Solve
        answer = self.get_answer(item.question)

        # 3. Judge
        eval_result = self.judge_answer(item.question, answer)
        
        # 4. Update
        self.update_ema(eval_result.score)
        print(f"[*] Score: {eval_result.score} | EMA: {self.ema_score:.4f}")

# Configuration
SYSTEM_CONFIG = {
    "api_key": "your-api-key",
    "base_url": "https://api.openai.com/v1", # Or your local provider
    "model_name": "gpt-4o-2024-08-06" # Must support Structured Outputs
}

bench = EvolvingBenchmark(**SYSTEM_CONFIG)

for i in range(5):
    print(f"\n--- Iteration {i+1} ---")
    bench.run_iteration()