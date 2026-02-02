import sys
import json
import time
import atexit
from typing import List, Any
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
import matplotlib.pyplot as plt

# ==========================================
# 1. SAFE TEE LOGGER
# ==========================================
class TeeLogger:
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

logger = TeeLogger("az_benchmark_log.txt")
sys.stdout = logger
atexit.register(logger.close)

# ==========================================
# 2. DATA SCHEMAS
# ==========================================
class BenchmarkItem(BaseModel):
    topic: str
    question: str
    difficulty_intent: int = Field(ge=1, le=10)

    @field_validator("difficulty_intent", mode="before")
    @classmethod
    def clean_int(cls, v: Any) -> int:
        if isinstance(v, str):
            return int(v.split("/")[0].strip())
        return v

class FailureMode(BaseModel):
    category: str
    description: str
class Evaluation(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    failure_modes: List[FailureMode]


# ==========================================
# 3. BENCHMARK ENGINE (FIXED)
# ==========================================
class AstraZenecaBenchmark:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        generator_model: str,
        solver_model: str,
        judge_model: str,
        alpha: float = 0.3,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.generator_model = generator_model
        self.solver_model = solver_model
        self.judge_model = judge_model

        self.alpha = alpha
        self.ema_score = None
        self.iteration = 0

        self.question_history: List[str] = []

    # --------------------------------------
    # Question Generator (Adaptive Difficulty)
    # --------------------------------------
    def generate_question(self, domain: str, difficulty: int) -> BenchmarkItem:
        recent_questions = "\n".join(self.question_history[-10:])

        if difficulty > 8:
            prompt = "\nCRITICAL: The question MUST contain a subtle false premise or a counter-intuitive edge case. It should be designed to TRICK the model."
        else: prompt = f"""
            You are a Senior Principal Scientist at AstraZeneca.

            Generate a *novel* reasoning-heavy question in {domain}.

            Difficulty level: {difficulty}/10

            Focus on ONE:
            - ADC linker or payload trade-offs
            - clinical inclusion/exclusion edge cases
            - biomarker stratification failures
            - regulatory or translational risks

            Avoid repeating prior reasoning patterns.
            Previous questions:
            {recent_questions}

            Return JSON only:
            {{
            "topic": "...",
            "question": "...",
            "difficulty_intent": {difficulty}
            }}
            """

        try:
                response = self.client.chat.completions.create(
                    model=self.generator_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                
                # Validate JSON structure
                item = BenchmarkItem.model_validate_json(
                    response.choices[0].message.content
                )
                self.question_history.append(item.question)
                return item

        except Exception as e:
                # Fallback if generation fails so the app doesn't crash
                print(f"Error generating question: {e}")
                return BenchmarkItem(
                    topic="Generation Error",
                    question="[SYSTEM ERROR: Could not generate question. Skipping...]",
                    difficulty_intent=difficulty
                )

    # --------------------------------------
    # Candidate Solver
    # --------------------------------------
    def solve(self, domain: str, question: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.solver_model,
                messages=[
                    {"role": "system", "content": f"You are an expert in {domain}."},
                    {"role": "user", "content": question},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error solving question: {e}")
            return "[SYSTEM ERROR: Solver failed to respond]"

    # --------------------------------------
    # External Judge 
    # --------------------------------------
    def judge(self, domain: str, question: str, answer: str) -> Evaluation:
        prompt = """
            You are an independent clinical trial reviewer.

            Evaluate the answer strictly for:
            - scientific correctness
            - clinical validity
            - missing critical considerations
            Be critical. If the answer is vague, generic, or misses any nuance, score it low. Do not give partial credit for 'mostly correct' answers 
            
            SCORING RULES:
            - Score 1.0: The answer is chemically/clinically perfect AND safe.
            - Score 0.5: The answer is mostly correct but misses a minor nuance or caveat.
            - Score 0.0: The answer contains ANY hallucination, factual error, or safety risk.

            
            Return JSON with the exact format:
            {
            "score": 0.0,
            "reasoning": "string",
            "failure_modes": [
                {
                "category": "string",
                "description": "string"
                }
            ]
            }

            """

        try:
                    response = self.client.chat.completions.create(
                        model=self.judge_model,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": f"Question:\n{question}\n\nAnswer:\n{answer}"},
                        ],
                        response_format={"type": "json_object"},
                    )

                    return Evaluation.model_validate_json(
                        response.choices[0].message.content
                    )

        except Exception as e:
                    print(f"Error judging response: {e}")
                    # Return a 'Neutral' or 'Failure' evaluation to keep the loop alive
                    return Evaluation(
                        score=0.0, 
                        reasoning=f"System Error during evaluation: {str(e)}", 
                        failure_modes=[FailureMode(category="System", description="API/JSON Error")]
                    )

    # --------------------------------------
    # Self-Evolving Loop
    # --------------------------------------
    def run_iteration(self, domain: str):
        self.iteration += 1

        # Adaptive difficulty
        if self.ema_score is None:
            difficulty = 5
        elif self.ema_score > 0.75:
            difficulty = min(10, 6 + self.iteration // 2)
        else:
            difficulty = max(3, 5 - self.iteration // 2)

        item = self.generate_question(domain, difficulty)
        answer = self.solve(domain, item.question)
        evaluation = self.judge(domain, item.question, answer)

        # EMA (fixed)
        if self.ema_score is None:
            self.ema_score = evaluation.score
        else:
            self.ema_score = (
                self.alpha * evaluation.score
                + (1 - self.alpha) * self.ema_score
            )

        return {
            "iteration": self.iteration,
            "difficulty": difficulty,
            "topic": item.topic,
            "question": item.question,
            "answer": answer,
            "score": evaluation.score,
            "ema": self.ema_score,
            "failure_modes": evaluation.failure_modes,
            "reasoning": evaluation.reasoning,
        }

