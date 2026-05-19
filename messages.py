import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ssl_fix  # noqa

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

'''
Message Types in LangChain

LangChain uses typed message objects rather than raw strings so that
different parts of the pipeline know who said what:

  SystemMessage  -> Sets the assistant persona at the start of conversation
  HumanMessage   -> The user input for each turn
  AIMessage      -> The model response stored for multi-turn memory

By accumulating these in a list and passing the full list on every call,
we give the model its memory. The model is stateless by nature so the
history list creates the illusion of an ongoing conversation.
'''

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=256,
    temperature=0.5,
    task="conversational",
)

model = ChatHuggingFace(llm=llm)

# Start with a system persona — this stays at the top of every call
conversation_history = [
    SystemMessage(content="You are a knowledgeable AI research assistant specializing in machine learning topics. Keep answers concise.")
]

print("=== Multi-turn AI Research Assistant (type 'exit' to quit) ===\n")

while True:
    query = input("Your question: ").strip()
    if query.lower() == "exit":
        break

    # Add user message to history
    conversation_history.append(HumanMessage(content=query))

    # Pass full history every time — this is how the model remembers
    response = model.invoke(conversation_history)

    print(f"\nAssistant: {response.content}\n")

    # Store model reply so next turn has full context
    conversation_history.append(AIMessage(content=response.content))

print("\n=== Full Conversation History ===")
for msg in conversation_history:
    label = type(msg).__name__
    print(f"[{label}] {msg.content}\n")
