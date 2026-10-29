"""Step 1 · Settings used by every step.

Change the defaults here, create Settings(...) with your own values in a notebook,
or pass them on the command line (python -m exam_predictor --help).
"""
from dataclasses import dataclass
from pathlib import Path

SESSION_CODE = {"m": "FM", "s": "MJ", "w": "ON"}   # Feb/March, May/June, Oct/Nov
SESSION_ORDER = {"m": 0, "s": 1, "w": 2}


@dataclass
class Settings:
    data_dir: Path = Path("/content/drive/MyDrive/physics pastpapers")     # 9702_s24_qp_41.pdf, 9702_s24_ms_41.pdf, ...
    work_dir: Path = Path("/content/drive/MyDrive/physics_exam_project")   # results and caches are saved here
    syllabus_start: int = 2022            # first exam year with the current syllabus content
    target_exam: str = "October/November 2026"
    llm_model: str = "Qwen/Qwen3.5-9B"    # "Qwen/Qwen3.5-4B" is about twice as fast but writes weaker questions
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    check_tags_with_llm: bool = True      # the AI double-checks the uncertain syllabus tags (cached after the first run)
    review: bool = True                   # the AI reads each finished question as a senior examiner
    max_tries: int = 3                    # attempts per generated question
    copy_limit: float = 0.95              # a new part this similar to a past part counts as copied

    def __post_init__(self):
        self.data_dir, self.work_dir = Path(self.data_dir), Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    @property
    def reports_dir(self) -> Path:
        """All tables, charts and the predicted paper are saved here."""
        path = self.work_dir / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path
