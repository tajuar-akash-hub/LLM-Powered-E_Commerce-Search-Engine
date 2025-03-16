from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

llm = ChatGroq(
    groq_api_key="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", 
    model_name="llama-3.3-70b-versatile"  
)

if __name__ == "__main__":
    response = llm.invoke("what is temperature , no preamble, straight answer ")
    print(response.content)