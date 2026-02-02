Self-evolving benchmark Generator (🧬 AstraZeneca: Clinical reasoning Assistant)

- Supports an OpenAI API–compatible endpoint (to be used for the question generation, the question answering, and the evaluation)
- Every question is novel
- Exponential moving average score

The LLM has 3 different parts: generator, solver, and judge. The generator produces a novel question every iteration, the solver answers the question, and the judge evaluates it.

During the utilization for research reasons, the generator part can be omitted. So when the model gets one question, the solver will answer it until the judge is 'satisfied'.
<img width="1316" height="701" alt="Screenshot 2026-02-01 at 21 37 29" src="https://github.com/user-attachments/assets/9288f9a8-c9b0-43f7-8df3-920c44dac9f9" />
