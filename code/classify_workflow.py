#!/usr/bin/env python3
"""
LangGraph deployment for article topic classification.

Deploys the selected model (Gemini) with few-shot prompting inside a LangGraph
workflow that routes articles based on confidence scoring, automatically
flagging low-confidence labels for human review.

Requires:
    pip install google-genai langgraph pandas
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from typing import Literal, TypedDict

import pandas as pd
from google import genai
from langgraph.graph import END, START, StateGraph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gemini-2.5-flash"
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
FEW_SHOT_PER_CATEGORY = 2

CATEGORY_LABELS = {
    "campaign": "Campaign",
    "civil_policy": "Civil Policy",
    "controversy": "Controversy",
    "endorsements": "Endorsements",
    "governance": "Governance",
    "social_cause": "Social Cause",
    "media": "Media",
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPOLOGY_PATH = os.path.join(PROJECT_ROOT, "typology.txt")
DEFAULT_CATEGORIES_DIR = os.path.join(PROJECT_ROOT, "categories")

client = genai.Client()


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class ArticleState(TypedDict, total=False):
    title: str
    description: str
    url: str
    few_shot_block: str
    typology: str
    predicted_category: str
    confidence: float
    reasoning: str
    route: Literal["accepted", "human_review"]
    needs_human_review: bool


# ---------------------------------------------------------------------------
# Few-shot examples from annotated category exports
# ---------------------------------------------------------------------------


def load_typology(path: str = TYPOLOGY_PATH) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


def load_few_shot_examples(
    categories_dir: str = DEFAULT_CATEGORIES_DIR,
    per_category: int = FEW_SHOT_PER_CATEGORY,
) -> str:
    """Build a few-shot block from stratified category CSVs."""
    lines: list[str] = []
    pattern = os.path.join(categories_dir, "*.csv")
    for path in sorted(glob.glob(pattern)):
        basename = os.path.basename(path)
        if basename in ("category_summaries.csv",):
            continue
        stem = os.path.splitext(basename)[0]
        label = CATEGORY_LABELS.get(stem, stem.replace("_", " ").title())
        df = pd.read_csv(path, engine="python", encoding="utf-8", on_bad_lines="skip").fillna("")
        if df.empty:
            continue
        sample = df.head(per_category)
        for _, row in sample.iterrows():
            title = str(row.get("title", "")).strip()
            desc = str(row.get("description", "")).strip()
            text = f"{title}. {desc}".strip()
            if len(text) > 500:
                text = text[:500] + " ..."
            lines.append(f'Example ({label}): "{text}"')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------


def build_classification_prompt(state: ArticleState) -> str:
    title = state.get("title", "").strip()
    description = state.get("description", "").strip()
    article_text = f"{title}. {description}".strip()
    return (
        "You classify news articles about Zohran Mamdani into exactly one topic category.\n\n"
        f"TYPOLOGY:\n{state['typology']}\n\n"
        f"FEW-SHOT EXAMPLES (title + opening → label):\n{state['few_shot_block']}\n\n"
        f"ARTICLE TO CLASSIFY:\n\"{article_text}\"\n\n"
        "Respond with JSON only, no markdown:\n"
        '{"category": "<one of the typology categories>", '
        '"confidence": <float 0.0-1.0>, '
        '"reasoning": "<one sentence>"}\n'
        "confidence should reflect how certain you are; use values below 0.75 when ambiguous."
    )


def parse_classification_response(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError(f"Could not parse model response: {text[:200]}")
        return json.loads(match.group(0))


def call_classifier(prompt: str) -> dict:
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return parse_classification_response(response.text)


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------


def classify_article(state: ArticleState) -> ArticleState:
    prompt = build_classification_prompt(state)
    result = call_classifier(prompt)
    category = str(result.get("category", "")).strip()
    confidence = float(result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(result.get("reasoning", "")).strip()
    return {
        "predicted_category": category,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def accept_label(state: ArticleState) -> ArticleState:
    return {"route": "accepted", "needs_human_review": False}


def flag_for_human_review(state: ArticleState) -> ArticleState:
    return {"route": "human_review", "needs_human_review": True}


def route_by_confidence(
    state: ArticleState, *, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> Literal["accept", "human_review"]:
    if state.get("confidence", 0.0) >= threshold:
        return "accept"
    return "human_review"


def build_workflow(confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
    graph = StateGraph(ArticleState)
    graph.add_node("classify", classify_article)
    graph.add_node("accept", accept_label)
    graph.add_node("human_review", flag_for_human_review)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        lambda state: route_by_confidence(state, threshold=confidence_threshold),
        {"accept": "accept", "human_review": "human_review"},
    )
    graph.add_edge("accept", END)
    graph.add_edge("human_review", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def load_articles_from_categories(
    categories_dir: str = DEFAULT_CATEGORIES_DIR,
    limit: int | None = None,
) -> list[dict]:
    """Load articles from stratified category CSVs when the merged corpus is unavailable."""
    frames = []
    for path in sorted(glob.glob(os.path.join(categories_dir, "*.csv"))):
        if os.path.basename(path) == "category_summaries.csv":
            continue
        frames.append(
            pd.read_csv(path, engine="python", encoding="utf-8", on_bad_lines="skip").fillna("")
        )
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["url"], keep="first")
    if limit:
        df = df.head(limit)
    return [
        {
            "title": str(row.get("title", "")),
            "description": str(row.get("description", "")),
            "url": str(row.get("url", "")),
        }
        for _, row in df.iterrows()
    ]


def load_articles_csv(path: str, limit: int | None = None) -> list[dict]:
    df = pd.read_csv(path, engine="python", encoding="utf-8", on_bad_lines="skip").fillna("")
    if limit:
        df = df.head(limit)
    articles = []
    for _, row in df.iterrows():
        articles.append(
            {
                "title": str(row.get("title", "")),
                "description": str(row.get("description", "")),
                "url": str(row.get("url", "")),
            }
        )
    return articles


def run_workflow(
    articles: list[dict],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    rate_limit_s: float = 0.5,
) -> tuple[list[dict], list[dict]]:
    typology = load_typology()
    few_shot_block = load_few_shot_examples()
    app = build_workflow(confidence_threshold=confidence_threshold)

    accepted: list[dict] = []
    review_queue: list[dict] = []

    for i, article in enumerate(articles):
        initial: ArticleState = {
            "title": article["title"],
            "description": article["description"],
            "url": article.get("url", ""),
            "typology": typology,
            "few_shot_block": few_shot_block,
        }
        final = app.invoke(initial)
        record = {
            "title": article["title"],
            "url": article.get("url", ""),
            "predicted_category": final.get("predicted_category"),
            "confidence": final.get("confidence"),
            "reasoning": final.get("reasoning"),
            "route": final.get("route"),
        }
        if final.get("needs_human_review"):
            review_queue.append(record)
        else:
            accepted.append(record)
        if rate_limit_s and i < len(articles) - 1:
            time.sleep(rate_limit_s)

    return accepted, review_queue


def save_results(
    accepted: list[dict],
    review_queue: list[dict],
    output_dir: str,
    *,
    confidence_threshold: float,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    accepted_path = os.path.join(output_dir, "classified_accepted.csv")
    review_path = os.path.join(output_dir, "human_review_queue.csv")
    summary_path = os.path.join(output_dir, "classification_run.json")

    fieldnames = ["title", "url", "predicted_category", "confidence", "reasoning", "route"]

    for path, rows in ((accepted_path, accepted), (review_path, review_queue)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "model": MODEL,
        "confidence_threshold": confidence_threshold,
        "n_accepted": len(accepted),
        "n_human_review": len(review_queue),
        "accepted": accepted,
        "human_review_queue": review_queue,
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"Accepted: {len(accepted)} → {accepted_path}")
    print(f"Human review: {len(review_queue)} → {review_path}")
    print(f"Summary: {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify articles with few-shot Gemini + LangGraph confidence routing."
    )
    parser.add_argument(
        "--input",
        default=os.path.join(PROJECT_ROOT, "articles_annotated.csv"),
        help="CSV with title and description columns",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "output", "classification"),
        help="Directory for accepted labels and human review queue",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max articles to process")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum confidence to auto-accept (else human review)",
    )
    args = parser.parse_args()

    if os.path.isfile(args.input):
        articles = load_articles_csv(args.input, limit=args.limit)
    else:
        print(f"Input not found: {args.input}; using categories/*.csv instead.")
        articles = load_articles_from_categories(limit=args.limit)
    if not articles:
        print("No articles to classify.", file=sys.stderr)
        return 1

    print(f"Classifying {len(articles)} articles (threshold={args.threshold})...")
    accepted, review_queue = run_workflow(
        articles,
        confidence_threshold=args.threshold,
    )
    save_results(
        accepted,
        review_queue,
        args.output_dir,
        confidence_threshold=args.threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
