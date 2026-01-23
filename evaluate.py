from rouge_score import rouge_scorer

def evaluate_rouge(reference_summary, generated_summary):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    scores = scorer.score(reference_summary, generated_summary)
    return scores
if __name__ == "__main__":
    # Example manual evaluation (offline)
    reference = """
    Data analytics involves extracting insights from data.
    It includes data collection, cleaning, and analysis.
    """

    generated = """
    Data analytics is the process of examining raw data.
    It includes data collection and exploratory analysis.
    """

    rouge_scores = evaluate_rouge(reference, generated)

    for metric, score in rouge_scores.items():
        print(f"{metric.upper()} -> F1: {score.fmeasure:.2f}")
