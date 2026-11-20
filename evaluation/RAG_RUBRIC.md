# RAG scoring rubric

Score each saved answer by hand in `evaluation/results/rag_outputs_<stamp>.csv`
(columns `faithfulness` and `answer_relevance`, values 0, 1 or 2). Read the answer
next to the cited passages in the matching `.json` file (or the Wikipedia pages).
Do not score from memory of what is true; score against the retrieved passages.

## Faithfulness: is every claim supported by the cited passages?
- **2**: every claim is supported by the cited passages, and the citations point to the passages that support them.
- **1**: mostly supported, but one claim is unsupported, over-stated, or cited to the wrong passage.
- **0**: contains claims that are contradicted by or absent from the passages, or has no usable citations.

## Answer relevance: does the answer address the question?
- **2**: directly and completely answers the question.
- **1**: partly answers it, is vague, or answers a nearby question.
- **0**: does not answer it, or answers a different question.

## Queries with `expected_answerable = no`
The right behaviour is the insufficient-evidence reply with no citations.
- Correct abstention: faithfulness 2, answer_relevance 2.
- Confident answer anyway: faithfulness 0, answer_relevance 0.

## Rules
- Score in a blinded pass if you can: hide `reference_answer` until you have scored.
- Fill `reviewer_notes` for every 0 or 1.
- If a provider error occurred (`error` column not empty), leave the scores blank and
  re-run that query instead of scoring it.
- A second reviewer (Shahzaib) should independently score a sample of at least 10 rows,
  to check agreement.
- Report scores only after all rows are reviewed. `--summarize` averages only rows
  where both scores are filled in, and says how many that is.