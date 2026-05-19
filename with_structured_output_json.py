import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ssl_fix  # noqa

import json
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

'''
Structured Output via Raw JSON Schema

Instead of Pydantic or TypedDict, you can define the expected output shape
using a plain JSON Schema dict. This approach is:
  - Framework-agnostic (JSON Schema is a universal standard)
  - Useful when you want to define schemas dynamically or load from config
  - The same format that OpenAI function calling and structured outputs use

"required" lists fields the model must always include.
"type": ["string", "null"] marks a field as nullable — the model may return
null if the information is not present in the input.
'''

json_schema = {
    "title": "ProductReview",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "All key topics and themes discussed in the review"
        },
        "summary": {
            "type": "string",
            "description": "A brief, objective summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg"],
            "description": "Overall sentiment of the review"
        },
        "pros": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "List of positive points raised in the review"
        },
        "cons": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "List of negative points raised in the review"
        },
        "reviewer": {
            "type": ["string", "null"],
            "description": "Name of the reviewer if mentioned"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=512,
    temperature=0.1,
    task="conversational",
)
model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a structured data extraction assistant.
Read the product review and return ONLY a JSON object matching this schema exactly:
{{
  "key_themes": [<list of string>],
  "summary": <string>,
  "sentiment": "pos" or "neg",
  "pros": [<list of string>] or null,
  "cons": [<list of string>] or null,
  "reviewer": <string> or null
}}
Required fields: key_themes, summary, sentiment.
No text outside the JSON object."""),
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
    result = json.loads(raw_output[start:end])
    print("=== Structured Output via JSON Schema ===\n")
    print(json.dumps(result, indent=2))
except json.JSONDecodeError:
    print("Model did not return valid JSON. Raw output:")
    print(raw_output)
