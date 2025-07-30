
import sys
import ujson as json
import re
import string
from collections import Counter


def normalize_answer(s):
    """
    Normalize answer string for consistent evaluation.
    Steps:
    1. Convert to lowercase
    2. Remove articles (a, an, the)
    3. Remove punctuation
    4. Standardize whitespace
    
    Args:
        s (str): Input answer string
    
    Returns:
        str: Normalized answer string
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)))).strip()

def f1_score(answer, gd_answer):
    """
    Calculate F1 score between answer and ground truth.
    Special handling for yes/no/noanswer questions.
    
    Args:
        answer (str): Predicted answer
        gd_answer (str): Ground truth answer
    
    Returns:
        float: f1_score
    """
    normalized_answer = normalize_answer(answer)
    normalized_gd_answer = normalize_answer(gd_answer)

    ZERO_METRIC = 0

    # Special handling for yes/no/noanswer questions
    if normalized_answer in ['yes', 'no', 'noanswer'] and normalized_answer != normalized_gd_answer:
        return ZERO_METRIC
    if normalized_gd_answer in ['yes', 'no', 'noanswer'] and normalized_answer != normalized_gd_answer:
        return ZERO_METRIC

    # Calculate token-based metrics
    answer_tokens = normalized_answer.split()
    gd_answer_tokens = normalized_gd_answer.split()
    common = Counter(answer_tokens) & Counter(gd_answer_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return ZERO_METRIC
    
    precision = 1.0 * num_same / len(answer_tokens)
    recall = 1.0 * num_same / len(gd_answer_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
