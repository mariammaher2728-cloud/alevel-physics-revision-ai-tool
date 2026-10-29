"""Run the whole project in order, from the PDFs to the predicted paper.

From a terminal (or a Colab cell starting with !):
    python -m exam_predictor --data "/path/to/physics pastpapers" --work /path/to/results
    python -m exam_predictor --no-llm            # analysis and blueprint only: no AI model, no GPU needed
    python -m exam_predictor --help              # all options

From Python:
    from exam_predictor.config import Settings
    from exam_predictor.pipeline import run
    results = run(Settings(data_dir=..., work_dir=...))
"""
import argparse

import pandas as pd

from .analysis import run_analysis
from .archive import ask_archive
from .blueprint import backtest, best_recency, make_blueprint, typical_paper
from .config import Settings
from .dataset import analysis_pool, build_question_table, show_question
from .display import show
from .embeddings import Embedder, part_vectors
from .papers import find_papers, papers_per_session
from .search import SearchIndex
from .syllabus import load_outcomes, outcome_info, topic_order
from .tagging import tag_parts

STAGES = ["tags", "analysis", "blueprint", "paper"]


def heading(text):
    print(f"\n{'=' * 8} {text} {'=' * 8}")


def run(settings, stop_after="paper", ask=None):
    """Run every step up to `stop_after` (tags, analysis, blueprint or paper). Returns all results in a dict."""
    reports = settings.reports_dir
    needs_llm = settings.check_tags_with_llm or stop_after == "paper"
    results = {"settings": settings}

    heading("Step 2 · AI language model")
    llm = None
    if needs_llm:
        from .llm import LLM
        llm = LLM(settings.llm_model, settings.work_dir)
    else:
        print("Not needed for this run.")

    heading("Step 3 · Past papers")
    papers = find_papers(settings.data_dir)
    show(papers_per_session(papers))

    heading("Steps 4-6 · Question table")
    df, quality = build_question_table(papers, settings.syllabus_start, reports)
    show(quality[~quality.extraction_ok].head(15))
    show_question(df, df.question_id.iat[0], images=False)

    heading("Step 7 · Syllabus learning outcomes")
    los = load_outcomes(settings.work_dir)
    info = outcome_info(los)

    heading("Step 8 · Embeddings, syllabus tags and search")
    embedder = Embedder(settings.embed_model)
    vecs = part_vectors(df, embedder, settings.work_dir)
    df = tag_parts(df, los, vecs, embedder, llm if settings.check_tags_with_llm else None, settings.work_dir)
    index = SearchIndex(df, vecs, embedder)
    print("Parts per topic (current syllabus):")
    show(df[df.current].topic.value_counts().to_frame("parts"))
    show(df[df.current].sample(min(10, int(df.current.sum())), random_state=1)[["question_id", "text", "lo", "lo_text"]], 90)
    results.update(llm=llm, papers=papers, df=df, quality=quality, los=los, embedder=embedder, part_vecs=vecs, index=index)
    if stop_after == "tags":
        return results

    heading("Step 9 · Analysis")
    pool = analysis_pool(df)
    results.update(pool=pool, tables=run_analysis(pool, info, topic_order(los), reports, settings.syllabus_start))
    if stop_after == "analysis":
        return results

    heading("Step 10 · Blueprint and backtest")
    bt = backtest(pool, info)
    print("% of predicted learning outcomes that really appeared on a paper of that sitting:")
    show(pd.concat([bt, bt.mean().to_frame("average").T.round(1)]))
    recency = best_recency(bt)
    print(f"Using recency weight {recency} for the prediction.")
    bt.to_csv(reports / "backtest.csv")
    blueprint = make_blueprint(pool, info, recency)
    print(f"\nBlueprint for {settings.target_exam} ({blueprint.marks.sum()} marks), "
          f"shaped like {typical_paper(pool)} (the most typical past paper):")
    show(blueprint[["question", "topic", "marks", "outcomes", "parts"]], 120)
    blueprint.to_csv(reports / "blueprint.csv", index=False)
    results.update(backtest=bt, blueprint=blueprint)
    if stop_after == "blueprint":
        return results

    heading("Step 11 · Writing the predicted paper")
    from .export import save_paper, show_paper
    from .generation.writer import summary_table, write_paper
    generated = write_paper(blueprint, pool, llm, embedder, vecs, settings.max_tries, settings.copy_limit, settings.review)
    show(summary_table(generated))

    heading("Step 12 · Saving the paper and mark scheme")
    save_paper(generated, reports, settings.target_exam)
    show_paper(generated, reports, settings.target_exam)
    results.update(generated=generated)

    if ask:
        heading("Step 13 · Ask the archive")
        ask_archive(ask, index, llm, (settings.syllabus_start, 2100))
        show(index.similar(pool.question_id.iat[0]))
    return results


def main(argv=None):
    defaults = Settings.__dataclass_fields__
    p = argparse.ArgumentParser(prog="python -m exam_predictor", description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(defaults["data_dir"].default), help="folder with the past-paper PDFs")
    p.add_argument("--work", default=str(defaults["work_dir"].default), help="folder for results and caches")
    p.add_argument("--target", default=defaults["target_exam"].default, help="name of the exam to predict")
    p.add_argument("--syllabus-start", type=int, default=defaults["syllabus_start"].default)
    p.add_argument("--llm-model", default=defaults["llm_model"].default)
    p.add_argument("--embed-model", default=defaults["embed_model"].default)
    p.add_argument("--max-tries", type=int, default=defaults["max_tries"].default)
    p.add_argument("--copy-limit", type=float, default=defaults["copy_limit"].default)
    p.add_argument("--no-tag-check", action="store_true", help="do not let the AI double-check uncertain tags")
    p.add_argument("--no-review", action="store_true", help="skip the examiner review of each question")
    p.add_argument("--no-llm", action="store_true", help="no AI model at all: stop after the blueprint")
    p.add_argument("--stop-after", choices=STAGES, default="paper")
    p.add_argument("--ask", help='a question for the archive, e.g. "How are marks given for Wien\'s law?"')
    a = p.parse_args(argv)

    settings = Settings(data_dir=a.data, work_dir=a.work, target_exam=a.target, syllabus_start=a.syllabus_start,
                        llm_model=a.llm_model, embed_model=a.embed_model, max_tries=a.max_tries,
                        copy_limit=a.copy_limit, check_tags_with_llm=not (a.no_tag_check or a.no_llm),
                        review=not a.no_review)
    stop_after = "blueprint" if a.no_llm and STAGES.index(a.stop_after) > STAGES.index("blueprint") else a.stop_after
    run(settings, stop_after=stop_after, ask=None if a.no_llm else a.ask)


if __name__ == "__main__":
    main()
