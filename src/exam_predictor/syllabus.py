"""Step 7 · The syllabus learning outcomes (topics 12-25) that every question part is tagged with.

- The list in data/syllabus_outcomes.csv is paraphrased from the 9702 syllabus for 2025-2027.
  The letter at the end of each code (13.4a) is just its order inside the sub-topic.
- Codes starting with L are old topics that only appear in papers before 2022.
- On the first run the list is copied to syllabus_outcomes.csv in the work folder. You can edit that copy,
  for example to match the official wording; delete it to get the defaults back.
"""
import shutil
from pathlib import Path

import pandas as pd

DEFAULT_OUTCOMES = Path(__file__).parent / "data" / "syllabus_outcomes.csv"


def load_outcomes(work_dir):
    """The outcomes table with columns code, topic and outcome (the file uses | between columns)."""
    path = Path(work_dir) / "syllabus_outcomes.csv"
    if not path.exists():
        shutil.copyfile(DEFAULT_OUTCOMES, path)
    los = pd.read_csv(path, sep="|", dtype=str)
    print(f"{len(los)} learning outcomes in {len(topic_order(los))} topics")
    return los


def topic_order(los):
    """Topic names in syllabus order."""
    return los.topic.drop_duplicates().tolist()


def legacy_topics(los):
    """Old topics that only appear in papers before 2022."""
    return set(los.topic[los.code.str.startswith("L")])


def outcome_info(los):
    """The outcomes table indexed by code, for looking up the topic and wording of a tag."""
    return los.set_index("code")
