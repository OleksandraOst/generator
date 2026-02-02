Self-evolving benchmark Generator (🧬 AstraZeneca: Clinical reasoning Assistant)

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

<img width="1316" height="701" alt="Screenshot 2026-02-01 at 21 37 29" src="https://github.com/user-attachments/assets/9288f9a8-c9b0-43f7-8df3-920c44dac9f9" />

<img width="1394" height="714" alt="Screenshot 2026-02-02 at 17 42 46" src="https://github.com/user-attachments/assets/18020c81-3ff8-4a17-a454-46553a85a891" />

