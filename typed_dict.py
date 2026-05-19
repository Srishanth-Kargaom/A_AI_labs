import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ssl_fix  # noqa

'''
TypedDict — Lightweight Typed Dictionaries

TypedDict is Python's built-in way to attach type hints to plain dicts
without the overhead of a full class. It is useful when you want IDE
type-checking and documentation benefits but do not need runtime validation
(that is Pydantic's job).

In LangChain's structured output, TypedDict is the lightest-weight schema
option — good for quick prototyping where you trust the LLM output format
and do not need strict validation.

Note: TypedDict does NOT validate at runtime. Passing wrong types will not
raise an error — it is purely a static type hint tool.
'''

from typing import TypedDict, Optional

class MLExperiment(TypedDict):
    experiment_name:    str
    model_architecture: str
    dataset:            str
    accuracy:           float
    notes:              Optional[str]

experiment: MLExperiment = {
    "experiment_name":    "ResNet50 Baseline",
    "model_architecture": "ResNet-50",
    "dataset":            "CIFAR-10",
    "accuracy":           93.4,
    "notes":              "No augmentation applied in this run."
}

print("Experiment record:")
for key, value in experiment.items():
    print(f"  {key}: {value}")
