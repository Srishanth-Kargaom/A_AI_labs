# LangChain Day 1 — HuggingFace Edition

## Setup
1. Fill in your token in the .env file:
   HUGGINGFACEHUB_API_TOKEN=hf_your_token_here

2. Install dependencies:
   pip install langchain langchain-core langchain-huggingface langchain-community huggingface_hub sentence-transformers scikit-learn "pydantic[email]" python-dotenv streamlit

## Run Order

cd into your project root (where ssl_fix.py is) then:

  python main.py
  python chatmodels\hf_mistral.py
  python Messages\messages.py
  python embeddingmodels\embeddings.py
  python prompts\prompt_generator.py        <- run this BEFORE research_tool
  python prompts\chat_prompt_template.py
  streamlit run prompts\research_tool.py
  python structured_output\pydantic_demo.py
  python structured_output\typed_dict.py
  python structured_output\with_structured_output_typeddict.py
  python structured_output\with_structured_output_pydantic.py
  python structured_output\with_structured_output_json.py
