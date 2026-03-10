Self-evolving benchmark Generator (🧬 Clinical reasoning Assistant)

- Supports an OpenAI API–compatible endpoint (to be used for the question generation, the question answering, and the evaluation)
- Every question is novel
- Exponential moving average score

The LLM has 3 different parts: generator, solver, and judge.
The Generator creates a unique question at the calculated difficulty.
The Solver (the model you are testing) attempts to answer it.
The Judge compares the answer to the question's intent and awards a score (0.0 to 1.0).
The score is fed back into the EMA, which will determine the difficulty for the *next* cycle.


The difficulty integer (1–10) directly changes the instructions given to the Generator Model.

* Low Difficulty (1–4): The Generator is instructed to ask straightforward questions (e.g., standard definitions, well-known clinical guidelines).
* High Difficulty (8–10): The Generator is forced to create "novel reasoning" tasks. It looks for edge cases, conflicting constraints, or scenarios where standard rules fail.

<img width="1436" height="746" alt="Screenshot 2026-03-10 at 11 24 14" src="https://github.com/user-attachments/assets/28e3d07c-28ed-4136-8ba1-f14708e4c426" />

<img width="1390" height="728" alt="Screenshot 2026-03-10 at 11 24 54" src="https://github.com/user-attachments/assets/97e99ea9-d64e-43f9-bc2e-3169d119b678" />


