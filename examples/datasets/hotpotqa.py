import joblib
from dspy.datasets.hotpotqa import HotPotQA
from optimas.wrappers.example import Example


def dataset_engine(train_size: int = 5000, dev_size: int = 250, test_path: str = "examples/data/hotpot-qa-distractor-sample.joblib", **kwargs):
    """
    Load and prepare the HotPotQA dataset in Example format.

    Args:
        train_size (int): Number of training examples to sample. Default is 5000.
        dev_size (int): Number of validation examples to sample. Default is 250.
        test_path (str): Path to the preprocessed distractor test set.

    Returns:
        tuple: (trainset, valset, testset), where each is a list of Example objects.
    """
    # Load HotPotQA train/dev sets
    dataset = HotPotQA(train_seed=1, train_size=train_size, dev_size=dev_size, test_size=0)
    trainset = [
        Example(question=ex.question, gd_answer=ex.answer).with_inputs("question")
        for ex in dataset.train
    ]
    valset = [
        Example(question=ex.question, gd_answer=ex.answer).with_inputs("question")
        for ex in dataset.dev
    ]

    # Load distractor test set
    hotpot_test = joblib.load(test_path).reset_index(drop=True)
    testset = [
        Example(
            question=row["question"],
            gd_answer=row["answer"]
        ).with_inputs("question")
        for _, row in hotpot_test.iterrows()
    ]

    return trainset, valset, testset


if __name__ == "__main__":

    trainset, valset, testset = dataset_engine()
    print(f"Loaded {len(trainset)} training, {len(valset)} validation, and {len(testset)} test examples.")
    print(trainset[0])
    # => 
    # Example({
    # 'question': 'At My Window was released by which American singer-songwriter?', 
    # 'gd_answer': 'John Townes Van Zandt'}
    # ) (input_keys={'question'})