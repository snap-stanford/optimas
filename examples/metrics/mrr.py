def mrr(candidate_ids, final_scores, answer_ids):
    """
    Compute Mean Reciprocal Rank (MRR) for a single query.

    Args:
        candidate_ids (List[Any]): List of candidate IDs.
        final_scores (List[float]): List of corresponding scores for the candidate IDs.
        answer_ids (Set[Any] or List[Any]): Ground truth answer IDs.

    Returns:
        float: Reciprocal rank (0 if no correct answer is found).
    """
    if not candidate_ids or not final_scores or not answer_ids:
        return 0.0

    # Sort candidates by descending score
    sorted_candidates = [cid for _, cid in sorted(zip(final_scores, candidate_ids), key=lambda x: -x[0])]

    # Find the rank (1-based) of the first correct answer
    for rank, cid in enumerate(sorted_candidates, start=1):
        if cid in answer_ids:
            return 1.0 / rank

    return 0.0  # No correct answer found
