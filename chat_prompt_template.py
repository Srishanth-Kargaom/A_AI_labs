import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ssl_fix  # noqa

'''
ChatPromptTemplate and MessagesPlaceholder

ChatPromptTemplate lets you define a structured conversation layout with
named slots. Instead of concatenating strings manually, you declare the
shape of the prompt once and fill in values later via .invoke().

MessagesPlaceholder is a special slot that accepts a list of pre-existing
message objects (e.g. loaded chat history) and inserts them verbatim into
the final prompt. This is how you inject previous turns without hard-coding
them into the template itself.

Use-case: Store chat history in a file between sessions, load it, pass it
to the placeholder, and the model receives full context without the template
needing to know the history contents.
'''

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# ── Example 1: Simple dynamic template ───────────────────────────────────
template = ChatPromptTemplate([
    ("system", "You are a skilled {domain} coach."),
    ("human", "Explain {concept} in {domain} using a simple analogy.")
])

prompt = template.invoke({
    "domain": "machine learning",
    "concept": "overfitting"
})

print("=== Example 1: Dynamic Domain Template ===")
print(prompt)
print()

# ── Example 2: Template with chat history injection ───────────────────────
chat_template = ChatPromptTemplate([
    ("system", "You are a friendly AI customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

chat_history = []

# Load prior conversation turns from file
history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.txt")
with open(history_path) as f:
    chat_history.extend(f.readlines())

print("=== Example 2: Loaded Chat History ===")
print(chat_history)
print()

prompt2 = chat_template.invoke({
    "chat_history": chat_history,
    "query": "Has my replacement order been shipped yet?"
})

print("=== Assembled Prompt with History ===")
print(prompt2)
