import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ssl_fix  # noqa

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

'''
HuggingFace Chat Models via LangChain

Instead of running a model locally (like Ollama) or paying per token
(like OpenAI), HuggingFace Inference API gives access to powerful
open-source models for free within usage limits.

ChatHuggingFace wraps HuggingFaceEndpoint and handles the chat message
format automatically — no need to manually add special tokens.

Key parameters:
  repo_id        -> The HF model identifier (username/model-name)
  max_new_tokens -> Caps how many tokens the model generates
  temperature    -> Controls creativity (0 = deterministic, 1 = creative)
  task           -> "conversational" for chat/instruction models
'''

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=200,
    temperature=0.6,
    task="conversational",
)

model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor. Keep answers under 100 words."),
    ("human", "Explain {topic} simply.")
])

chain = prompt | model | StrOutputParser()

print("Response:\n")
result = chain.invoke({"topic": "how attention mechanism works in transformers"})
print(result)
print()
