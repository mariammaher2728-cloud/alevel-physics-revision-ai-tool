# Physics Exam Predictor · Cambridge 9702 Paper 4

Analyse past **Cambridge International A Level Physics (9702) Paper 4** question papers and mark schemes with
**RAG** (retrieval-augmented generation), find the important topics, repeated tasks and trends, predict the shape of
the next paper, and let an open AI model write a realistic **practice paper with its mark scheme**.

Everything runs on a free Google Colab T4 GPU with open models from Hugging Face:
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (4-bit) writes, and
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) finds similar questions.

## What it does

1. **Question table** – reads every PDF into one row per marked part, e.g. `9702_s24_41_Q3(b)(ii)`, with its text,
   marks, context, command word and mark-scheme answer. Each paper is checked to total 100 marks.
2. **Syllabus tags** – tags every part with the syllabus learning outcome it tests, using embeddings. The AI
   double-checks the uncertain tags.
3. **Analysis** – marks per topic, the most tested outcomes, repeated tasks, command-word trends, the usual
   paper layout, a definitions bank and parts repeated word for word.
4. **Prediction** – a blueprint of the next paper, shaped like the most typical real paper, tested on past
   sittings against random guessing (a *backtest*).
5. **Practice paper** – the AI writes new questions in Cambridge style from real past examples. **Python does
   everything that must be exact**: marks, mark codes (B1/C1/A1), calculated answers, graphs, and checks for
   made-up numbers and copied parts. An "examiner" review then checks the physics. The result is saved as Word files.

## Repository layout

```text
physics-exam-predictor/
├── README.md
├── pyproject.toml                  package settings (pip install -e .)
├── requirements.txt                libraries
├── notebooks/
│   ├── colab_quickstart.ipynb      run the package in Colab, one step per cell
│   └── physics_exam_analysis_all_in_one.ipynb   the same project as one self-contained notebook
├── src/exam_predictor/
│   ├── config.py          Step 1   settings: folders, models, limits
│   ├── llm.py             Step 2   the AI model, JSON reading and the reply cache
│   ├── papers.py          Step 3   find the past papers and keep complete pairs
│   ├── extract.py         Step 4   read a PDF into question records
│   ├── dataset.py         Step 5-6 build the question table; look at any question
│   ├── syllabus.py        Step 7   the syllabus learning outcomes
│   ├── data/syllabus_outcomes.csv          the outcomes list (code | topic | outcome)
│   ├── embeddings.py      Step 8   meaning vectors of every part
│   ├── tagging.py         Step 8   tag every part with a learning outcome
│   ├── search.py          Step 8   hybrid search and "similar parts"
│   ├── analysis.py        Step 9   topics, repeated tasks and trends
│   ├── blueprint.py       Step 10  the blueprint of the next paper and the backtest
│   ├── generation/        Step 11  writing the paper
│   │   ├── numbers.py              calculating and printing numbers the Cambridge way
│   │   ├── schemas.py              the JSON shapes the model replies with
│   │   ├── prompts.py              past-question examples (RAG) and prompts
│   │   ├── checks.py               Python finishes each question and lists its problems
│   │   └── writer.py               generate → check → examiner review → fix
│   ├── export.py          Step 12  Word and Markdown files of the paper and mark scheme
│   ├── archive.py         Step 13  ask questions about the past papers (RAG answers)
│   ├── text_utils.py               small text helpers
│   ├── display.py                  tables and charts in a notebook or a terminal
│   └── pipeline.py                 run everything in order (command line)
├── tests/                          checks of the exact Python work (no GPU needed)
└── .github/workflows/tests.yml     runs the tests on GitHub
```

## Quick start in Google Colab

1. Put the past papers in Google Drive, e.g. `MyDrive/physics pastpapers/`. File names must look like
   `9702_s24_qp_41.pdf` (question paper) and `9702_s24_ms_41.pdf` (mark scheme); subfolders are fine.
2. Open `notebooks/colab_quickstart.ipynb` in Colab (*File → Open notebook → GitHub*).
3. Choose *Runtime → Change runtime type → T4 GPU*.
4. Set `REPO_URL` in the first cell to this repository's address, check the folders in Step 1, then *Runtime → Run all*.

The first run takes about an hour, mostly the AI writing questions. Replies, embeddings and tags are cached in the
work folder, so later runs take a few minutes.

## Run from the command line

```bash
pip install -e .                       # or: pip install -r requirements.txt
python -m exam_predictor --data "/path/to/physics pastpapers" --work ./results
python -m exam_predictor --data "/path/to/physics pastpapers" --work ./results --no-llm   # analysis + blueprint only, no GPU
python -m exam_predictor --help        # all options
```

In Colab, start the command with `!`. After `pip install -e .` the same command is also available as `exam-predictor`.

| Option | Default | Meaning |
|---|---|---|
| `--data` | `/content/drive/MyDrive/physics pastpapers` | folder with the past-paper PDFs |
| `--work` | `/content/drive/MyDrive/physics_exam_project` | folder for results and caches |
| `--target` | `October/November 2026` | name of the exam to predict |
| `--llm-model` | `Qwen/Qwen3.5-9B` | `Qwen/Qwen3.5-4B` is about twice as fast but writes weaker questions |
| `--max-tries` | `3` | attempts per generated question |
| `--no-tag-check` | | the AI does not double-check uncertain tags |
| `--no-review` | | skip the examiner review |
| `--no-llm` | | no AI model at all; stop after the blueprint |
| `--stop-after` | `paper` | `tags`, `analysis`, `blueprint` or `paper` |
| `--ask` | | a question for the archive (Step 13) |

## Outputs (in the work folder)

| File | Contents |
|---|---|
| `reports/predicted_paper.docx`, `.md` | the practice paper, with graphs |
| `reports/predicted_mark_scheme.docx`, `.md` | its mark scheme; unsolved problems are marked **CHECK** |
| `reports/blueprint.csv`, `backtest.csv` | the prediction and how well the method did on past sittings |
| `reports/topics.csv`, `learning_outcomes.csv`, `repeated_tasks.csv`, `template.csv`, `definitions_bank.csv`, `same_wording.csv` | the analysis tables |
| `reports/*.png` | charts: marks per topic, topic by year, command words |
| `reports/extraction_quality.csv` | the 100-mark check of every paper |
| `questions_tagged.parquet` | the question table with tags |
| `syllabus_outcomes.csv` | your editable copy of the learning outcomes |
| `llm_cache.json`, `part_vectors.npz` | caches; `llm_bad_replies.txt` keeps any unusable AI reply |

## How the practice paper is written

- **RAG:** for each blueprint question, up to 3 real past questions on the same topic (with their mark schemes) are
  given to the model as style examples.
- **Template:** the model fills in a JSON template with the exact parts and marks of a real question, one line per mark.
- **Python is the calculator:** the model writes each calculation as a Python expression and each graph as a
  formula `y = f(x)`, because language models often get arithmetic wrong. Python works out the answers, writes the
  `C1`/`A1` lines, draws the graphs, and checks for numbers that are not in the question's data.
- **Fix, don't restart:** any problem goes back to the model with the question, and the model corrects only that.
- **Examiner review:** the model then reads the finished question as a senior examiner and lists serious physics errors.

## Tests

```bash
pip install pytest pydantic numpy pandas
pytest -q
```

The tests check the exact Python work: reading and printing numbers, finishing mark schemes and reading the model's
JSON. They do not need a GPU or the AI models, and they run automatically on GitHub for every push.

## Notes and limits

- **Do not commit the past papers.** Cambridge papers are copyright; `.gitignore` keeps PDFs and results out of the repository.
- The learning outcomes in `data/syllabus_outcomes.csv` are paraphrased from the 9702 syllabus for 2025–2027.
  Edit your copy in the work folder to match the official wording.
- Tags, the prediction and the generated questions come from models and simple rules. The backtest and your own check
  of 40 random tags show how far to trust them. **Solve the generated paper yourself before using it.**
- Ideas: hand-check tag accuracy, compare the backtest with random guessing, and ask a teacher to rate a mix of real
  and generated questions without knowing which is which.
