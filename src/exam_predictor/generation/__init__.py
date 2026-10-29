"""Step 11 · Write the predicted paper (RAG generation with checks).

The idea: the AI model writes the physics, and Python does everything that must be exact.

    numbers.py   reading, calculating and printing numbers the Cambridge way
    schemas.py   the JSON shapes the model must reply with
    prompts.py   past-question examples (the RAG part) and the prompts
    checks.py    Python finishes each question: labels, marks, mark codes, answers, and the checks
    writer.py    the loop: generate -> Python checks -> examiner review -> fix, for every blueprint question

Use:  from exam_predictor.generation.writer import write_paper, summary_table
"""
