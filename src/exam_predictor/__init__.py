"""Predict the next Cambridge 9702 Paper 4 with RAG.

Each module is one step of the project (the numbers match the original notebook):

    config       Step 1   settings (folders, models, limits)
    llm          Step 2   the AI language model from Hugging Face, with a JSON reader and a cache
    papers       Step 3   find the past papers and keep complete pairs
    extract      Step 4   read a PDF into question records
    dataset      Step 5-6 build the question table and look at any question
    syllabus     Step 7   the syllabus learning outcomes
    embeddings   Step 8   meaning vectors of every part
    tagging      Step 8   tag every part with a learning outcome (optional AI check)
    search       Step 8   hybrid search and "similar parts"
    analysis     Step 9   important topics, repeated tasks and trends
    blueprint    Step 10  predict the next paper and backtest the method
    generation   Step 11  write the predicted paper (RAG + Python checks + examiner review)
    export       Step 12  save the paper and mark scheme as Word and Markdown files
    archive      Step 13  ask the archive (RAG question answering)
    pipeline              run everything in order: python -m exam_predictor --help
"""

__version__ = "0.1.0"
