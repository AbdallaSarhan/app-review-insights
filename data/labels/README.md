# Labelling guide

`human_labels.csv` holds 200 reviews. Fill in the `intent` column with **one** of the labels below. The quickest way is the keyboard labeller, which shows one review at a time and saves after every key press (1–6 to label, `b` back, `n` skip, `q` quit):

```bash
python -m reviewinsights.label
```

When every row is labelled, run:

```bash
python -m reviewinsights.benchmark score
```

Star ratings and the model's answers are deliberately hidden. Judge only the text.

| label | use it when the review… |
|---|---|
| `bug` | reports something broken: crashes, lost data or progress, features not working, sync or login failures |
| `pricing` | complains or asks about cost, subscriptions, paywalls, trials, refunds or ads |
| `feature_request` | asks for something new, or for an existing feature to work differently |
| `usability` | describes confusing, tedious or annoying design that isn't broken: onboarding, navigation, notifications, UI |
| `praise` | is positive, with no request and no problem |
| `other` | is too vague to act on, off-topic, or not about the app |

**If several labels fit, pick the one highest in the table.** For example, "Love it, but it crashes when I add a habit" is `bug`, and "Great app, please add dark mode" is `feature_request`.

The model gets exactly these definitions and the same tie-break rule (`src/reviewinsights/llm.py`), so any disagreement between you and the model is about judgement, not about different instructions.

Tips: label in one or two sittings so you stay consistent, and don't open the LLM labels until you're done.
