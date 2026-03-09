#!/usr/bin/env python3
"""
summarize_categories_with_llm.py

For each CSV in ./categories/:
 - reads title+description for each article
 - computes top-10 TF-IDF words for that category
 - sends a prompt to an LLM to produce a representative contextual summary
 - saves results to category_summaries.csv and category_summaries.json

Requires:
    pip install openai scikit-learn pandas
Set:
    export OPENAI_API_KEY="sk-..."
"""

import os
import json
import glob
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from google import genai
# --- GLOBAL TF-IDF FITTED ON THE FULL COLLECTION ---
FULL_CORPUS_CSV = "articles_annotated.csv"

df_full = pd.read_csv(FULL_CORPUS_CSV, engine="python", encoding="utf-8", on_bad_lines="skip").fillna("")
full_docs = (df_full["title"] + ". " + df_full["description"]).astype(str).tolist()

from sklearn.feature_extraction.text import TfidfVectorizer

global_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
)

# Fit on the entire dataset
global_vectorizer.fit(full_docs)


# --- LLM call: example using OpenAI Python client ---

# Configure Gemini API
client = genai.Client()

MODEL = "gemini-2.5-flash"

def call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 400) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return response.text

def top_n_tfidf_words(docs, n=10):
    """
    Compute TF-IDF for a category using:
    - TF from docs inside this category
    - IDF from the global training corpus
    """
    if not docs:
        return []

    # transform category docs with global IDF
    X = global_vectorizer.transform(docs)

    # sum TF-IDF scores across all documents in category
    import numpy as np
    s = X.sum(axis=0).A1

    # take top n features
    features = global_vectorizer.get_feature_names_out()
    top_idx = s.argsort()[::-1][:n]
    return [features[i] for i in top_idx]

# --- core processing ---
def summarize_all_categories(categories_dir="categories", output_csv="category_summaries.csv", output_json="category_summaries.json"):
    files = sorted(glob.glob(os.path.join(categories_dir, "*.csv")))
    rows = []
    for f in files:
        cat_name = os.path.splitext(os.path.basename(f))[0]
        print(f"Processing category file: {f} -> {cat_name}")
        df = pd.read_csv(f, engine="python", encoding="utf-8", on_bad_lines="skip").fillna("")
        # build corpus: title + description (only those two per project spec)
        docs = (df.get("title", "") + ". " + df.get("description", "")).astype(str).tolist()
        # limit docs considered in prompt to, say, 25 examples to keep LLM prompt size reasonable
        sample_for_prompt = docs[:25]

        top10 = top_n_tfidf_words(docs, n=10)

        # Compose LLM prompt
        prompt = (
            f"You are given a set of news article openings (title + opening sentence) about a political figure.\n"
            f"Category label: **{cat_name}**\n\n"
            f"Top 10 distinguishing words for this category (computed via TF-IDF): {', '.join(top10)}\n\n"
            f"Here are up to {len(sample_for_prompt)} example article title+openings (each entry is 'Title. Opening'):\n"
        )
        for i, d in enumerate(sample_for_prompt, 1):
            short = d.strip()
            if not short:
                continue
            if len(short) > 600:
                short = short[:600] + " ..."
            prompt += f"{i}. {short}\n"

        prompt += (
            "\nTask: Produce a concise representative contextual summary (about 100-220 words) that explains what the articles in this category are broadly about, "
            "the main themes, likely framing and who or what is discussed, and any important context a reader should know. "
            "Write in neutral academic tone. Start with a 1-2 sentence summary summary, then 3-4 short bullet points with notable details.\n\n"
            "Important: Base your summary only on the examples and the top-10 words above. Do not invent quotes or facts not inferable from the snippets.\n\n"
            "Output format:\n"
            "SUMMARY: < sentence summary>\n"
            "DETAILS:\n"
            "- ...\n"
            "- ...\n"
        )

        # Call the LLM
        try:
            llm_output = call_llm(prompt)
        except Exception as e:
            print("LLM call failed:", e)
            llm_output = f"<<LLM call failed: {e}>>"

        row = {
            "category_file": f,
            "category_label": cat_name,
            "n_articles_in_file": len(df),
            "top10_tfidf": top10,
            "llm_summary_raw": llm_output
        }
        rows.append(row)
        # be polite with rate limits
        time.sleep(1.0)

    # Save to CSV & JSON
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)

    print(f"Saved summaries to {output_csv} and {output_json}")

if __name__ == "__main__":
    summarize_all_categories()

