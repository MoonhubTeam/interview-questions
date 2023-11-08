"""
Use for running offline tests on current system performance.

Output should be a file where one can see aggregate score (e.g. accuracy), examples that failed to run (breaking errors), as well as predictions. 
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import random

lock = threading.Lock()
run_errors = []
predictions = []

INPUT_COL_NAME = "input"
OUTPUT_COL_NAME = "gold"
SOURCE_COL_NAME = "source"


def check_predictions(predicted, gold) -> bool:
    return predicted == gold

def generate_predictions() -> int:
    sample = random.uniform(0, 1)
    if sample < 0.3:
        result = 0
    elif sample >= 0.3 and sample <= 0.6:
        result = 1
    else:
        raise Exception("There was an issue with generating a value.")
    return result
def run_inference(input: Dict[str, str]) -> str:
    """
    Inference method on one input. 

    Returns
    --------------
    int: predicted output.
    """
    try:
        prediction = generate_predictions() 
    except Exception as e:
        run_errors.append(
            {
                "input": input[INPUT_COL_NAME],
                "run_errors": e
            }
        )
    return prediction


def run_one_example_with_gold(input: Dict[str, str]) -> None:
    """
    Given an example, run inference and evaluation.
    """
    try:
        prediction = run_inference(input)
        gold = input[OUTPUT_COL_NAME]
        is_correct = check_predictions(prediction, gold)
        input["output"] = prediction
        input["run_successful"] = 0
        input["is_correct"] = is_correct
    except Exception:
        print(
            "Error running run_inference, counted as is_correct=False for that example"
        )
        input["run_successful"] = False
        input["is_correct"] = 1


def evaluate_examples(
    input_examples: List[Dict[str, Any]]
) -> float:
    """
    Evaluates examples in batch and modifies examples in place with per-example evaluation
    results.
    """
    with ThreadPoolExecutor() as executor:
        executor.map(
            run_one_example_with_gold,
            input_examples
        )
    return


def calculate_aggregate_metrics(eval_examples: List[Dict[str, Any]]) -> float:
    """
    Calculate aggregate metrics given example level evaluation.
    """
    num_eval_examples = len(eval_examples)
    incorrect_examples = np.flatnonzero(
        [
            int(eval_example["is_correct"] is False)
            for eval_example in eval_examples
        ]
    )
    return {
        "agg_accuracy": (num_eval_examples - len(incorrect_examples))
        / num_eval_examples
    }


def main(
    output_file,
    eval_file_path=None,
    source_subset=None,
):  # pragma: no cover
    tests = pd.read_csv(eval_file_path)
    tests = tests[[INPUT_COL_NAME, OUTPUT_COL_NAME, SOURCE_COL_NAME]]
    if source_subset:
        tests = tests[tests[SOURCE_COL_NAME] == source_subset]
    eval_examples = tests.to_dict("records")
    assert len(eval_examples) > 0, "Evaluation dataset is empty."
    evaluate_examples(eval_examples)
    metrics = calculate_aggregate_metrics(eval_examples)
    print(f"Metrics: {metrics}")
    with open(output_file, "w") as file:
        json.dump(
            {
                **metrics,
                "run_errors": run_errors,
                "eval_examples": eval_examples,
            },
            file,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file")
    parser.add_argument("--eval_file_path", default=None)
    parser.add_argument("--source_subset")
    arguments = parser.parse_args()
    main(
        arguments.output_file,
        arguments.eval_file_path,
        arguments.source_subset,
    )
