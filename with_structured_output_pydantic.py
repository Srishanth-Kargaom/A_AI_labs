import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ssl_fix  # noqa

import json
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Optional, Literal

'''
Structured Output with Pydantic

Pydantic adds runtime validation on top of what TypedDict gives you.
When combined with .with_structured_output(), LangChain:
  1. Reads Field(description=...) annotations to build the system prompt.
  2. Parses the model JSON response into a fully validated Pydantic object.
  3. Raises ValidationError if the model returns something that violates
     the schema (e.g. a sentiment value outside the allowed Literal options).

.model_dump_json() serialises the validated Pydantic object to a JSON string,
making it trivial to log, store, or pass to a downstream API.

Key advantage over TypedDict: you get real validation. A Literal["pos","neg"]
field will fail if the model hallucinates "neutral" — TypedDict will not.
'''

class ProductReview(BaseModel):
    key_themes: list[str]            = Field(description="All major topics covered in the review")
    summary:    str                  = Field(description="A concise summary of the overall review")
    sentiment:  Literal["pos","neg"] = Field(description="Overall sentiment: pos or neg")
    pros:       Optional[list[str]]  = Field(default=None, description="Positive aspects mentioned")
    cons:       Optional[list[str]]  = Field(default=None, description="Negative aspects mentioned")
    reviewer:   Optional[str]        = Field(default=None, description="Name of the reviewer if mentioned")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=512,
    temperature=0.1,
    task="conversational",
)
model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a product review analyst. Analyse the review and respond ONLY with a valid JSON object.
Expected format:
{{
  "key_themes": ["list of major themes"],
  "summary": "brief overall summary",
  "sentiment": "pos" or "neg",
  "pros": ["list of pros"],
  "cons": ["list of cons"],
  "reviewer": "reviewer name if found or null"
}}
No extra text. No markdown. JSON only."""),
    ("human", "{review}")
])

chain = prompt | model | StrOutputParser()

reviews_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reviews.txt")
with open(reviews_path) as f:
    review_text = f.read()

raw_output = chain.invoke({"review": review_text})

try:
    start = raw_output.find("{")
    end   = raw_output.rfind("}") + 1
    data  = json.loads(raw_output[start:end])

    # Pydantic validates — will raise if types or constraints are wrong
    review_obj = ProductReview(**data)

    print("=== Validated Review Analysis (Pydantic) ===\n")
    print(review_obj.model_dump_json(indent=2))

except Exception as e:
    print(f"Error: {e}")
    print("Raw output:")
    print(raw_output)
