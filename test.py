from llm import GroqLLM

llm = GroqLLM()

response = llm("short 1 line answer on who you are?")
print(response)
