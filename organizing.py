import pandas as pd
import csv

df = pd.read_csv(
    "articles_annotated.csv",
    engine="python",
    encoding="utf-8",
    on_bad_lines="skip"
)

# Prevent NaN issues
df = df.fillna("")

def categorize(row):
    text = (row["title"] + " " + row["description"]).lower()

    if "endorse" in text:
        return "Endorsements"
    if "[t]" in text or "terror" in text:
        return "Social Cause"
    if "[a]" in text or "antisemit" in text:
        return "Social Cause"
    if "[c]" in text or "campaign" in text:
        return "Campaign"
    if any(x in text for x in ["policy", "abolish", "ban", "proposal", "legislation", "program"]):
        return "Civil Policy"
    if any(x in text for x in ["racism", "Israel", "antisemitism", "scandal", "attack", "accus", "alleg", "ghosted", "ties to", "backroom"]):
        return "Controversy"

    return "Campaign"


def sentiment(row):
    text = (row["title"] + " " + row["description"]).lower()

    positives = ["endorsement", "backs", "praises", "supports", "endorses", "touts", "defends", "victory", "defended"]
    negatives = ["nepo", "liar", "mad", "lack" "Hamas", "anti", "radical", "racist", "sexist", "islamophobia", "critic", "criticize", "antisemitism", "slams", "blasts", "criticize", "accused", "ghosted", "attack", "negative"]

    if any(w in text for w in positives):
        return "Positive"
    if any(w in text for w in negatives):
        return "Negative"
    return "Neutral"



# Annotate inclusively rows 71 to 374

start = 71
end = 374

df.loc[start:end, "categories"] = df.loc[start:end].apply(categorize, axis=1)
df.loc[start:end, "sentiment"] = df.loc[start:end].apply(sentiment, axis=1)


df.to_csv("articles_annotated.csv", index=False)

