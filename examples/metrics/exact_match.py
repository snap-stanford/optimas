import re
import string


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def strip(text):
        return text.strip()

    return white_space_fix(remove_articles(remove_punc(strip(lower(s)))))

def EM(answer, gd_answer):
    norm_answer = normalize_answer(answer)
    norm_gd_answer = normalize_answer(gd_answer)
    result = norm_answer == norm_gd_answer
    return float(result)
