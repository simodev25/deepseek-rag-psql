import ollama

response = ollama.embeddings(
    model="deepseek-coder",
    prompt="Hello world"
)

print(response['embedding'])
