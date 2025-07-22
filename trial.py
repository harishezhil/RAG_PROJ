from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(model="gemma2-9b-it",temperature=0)  # Or try another model if available
response = llm.invoke("Who founded Flipkart?")
print(response)