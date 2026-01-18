import ollama

response = ollama.chat(model='gemma', messages=[
  {
    'role': 'user',
    'content': 'Why is the Tortoise and Hare algorithm efficient?',
  },
])

print(response['message']['content'])
