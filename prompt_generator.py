import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ssl_fix  # noqa

'''
PromptTemplate — Separating Structure from Content

Hard-coding prompts as plain strings scattered across your codebase becomes
a maintenance nightmare at scale. PromptTemplate solves this by treating
prompts as versioned, reusable artifacts with named placeholders.

Benefits:
  - Swap values without touching the template logic
  - Save templates to disk (JSON) and load them elsewhere
  - Validate that all required variables are present at invocation time

The .save() method serialises the template to JSON so it can be loaded
in another module via load_prompt() — useful when your UI layer and your
LLM logic live in separate files.
'''

from langchain_core.prompts import PromptTemplate

template_string = """
You are an expert Machine Learning tutor. Your job is to explain the given
ML algorithm in a way that matches the learner's background and goals.

Algorithm      : {algorithm}
Teaching Style : {style}
Response Length: {length}

Guidelines:
- Match the complexity of your explanation to the requested style.
- Include intuitive analogies to build mental models.
- Add a minimal Python code snippet where it adds value.
- If the algorithm is unfamiliar, respond with: "Insufficient information available."
- Stick strictly to the requested response length.
"""

prompt_template = PromptTemplate(
    input_variables=["algorithm", "style", "length"],
    template=template_string,
)

# Save to JSON so research_tool.py can load it without redefining it
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_template.json")
prompt_template.save(save_path)
print(f"Template saved to: {save_path}")

# Quick preview of a formatted prompt
sample = prompt_template.format(
    algorithm="Gradient Boosting",
    style="Beginner-friendly with analogies",
    length="Medium (200-300 words)"
)
print("\n=== Sample Formatted Prompt ===\n")
print(sample)
