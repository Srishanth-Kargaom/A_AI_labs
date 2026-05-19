import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssl_fix  # noqa — must be first, fixes SSL for college/corporate networks

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser

'''
How LangChain Manages Model Integrations

The LangChain ecosystem avoids a monolithic design by splitting model
integrations into focused, independently maintained packages:

langchain-core        -> Defines the base interfaces and abstractions
                         (Runnable, BaseMessage, etc.) that every integration
                         must follow. Think of it as the contract layer.

langchain-huggingface -> A dedicated partner package for HuggingFace models.
                         Bundles ChatHuggingFace (chat format) and
                         HuggingFaceEndpoint (serverless API calls).
                         Updated independently when HF releases new features.

langchain-community   -> A staging area for integrations that are widely used
                         but not yet spun off into their own package.

This separation means upgrading the HF integration never forces you to
upgrade unrelated parts of your stack.
'''

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=150,
    temperature=0.7,
    task="conversational",
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

print("Streaming a short story about a data scientist:\n")
result = model.invoke("Write a short story about a data scientist who discovers an anomaly in a dataset. Keep it under 100 words.")
print(parser.invoke(result))
print()
