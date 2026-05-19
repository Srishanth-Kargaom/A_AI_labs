import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ssl_fix  # noqa

import json
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Annotated, Optional, Literal

'''
Structured Output with TypedDict

.with_structured_output() is LangChain's unified API for making a model
return data conforming to a schema. When you pass a TypedDict class,
LangChain automatically:
  1. Generates a system prompt instructing the model to produce JSON
     matching the TypedDict shape.
  2. Parses the model response back into a Python dict with correct keys.

Annotated[<type>, "<description>"] embeds field-level instructions into
the schema. LangChain extracts those descriptions and includes them in the
system prompt it generates — so the model knows not just the type but what
each field should contain.

Note: For HF models we replicate this behaviour via prompt engineering
since HF free tier does not support native function/tool calling.
'''

class ProductReview(TypedDict):
    key_themes: Annotated[list[str], "List all major themes discussed in the review"]
    summary:    Annotated[str,       "A concise 1-2 sentence summary of the review"]
    sentiment:  Annotated[Literal["Pos", "Neg"], "Overall sentiment — Pos or Neg"]
    pros:       Annotated[Optional[list[str]], "List of positives mentioned"]
    cons:       Annotated[Optional[list[str]], "List of negatives mentioned"]

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=512,
    temperature=0.1,
    task="conversational",
)
model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a product review analyst.
Read the review and respond ONLY with a valid JSON object — no extra text, no markdown fences.
Use exactly this structure:
{{
  "key_themes": ["theme1", "theme2"],
  "summary": "brief summary",
  "sentiment": "Pos" or "Neg",
  "pros": ["pro1", "pro2"],
  "cons": ["con1", "con2"]
}}"""),
    ("human", "{review}")
])

chain = prompt | model | StrOutputParser()

reviews_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reviews.txt")
with open(reviews_path) as f:
    review_text = f.read()

raw_output = chain.invoke({"review": review_text})

try:
    start  = raw_output.find("{")
    end    = raw_output.rfind("}") + 1
    result: ProductReview = json.loads(raw_output[start:end])
    print("=== Structured Review Analysis (TypedDict schema) ===\n")
    for key, value in result.items():
        print(f"{key}: {value}")
except json.JSONDecodeError:
    print("Raw model output (JSON parsing failed):")
    print(raw_output)
