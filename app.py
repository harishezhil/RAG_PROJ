# flipkart_rag_app.py
import sys
import asyncio
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from prompt import chain_of_thought_prompt
from utils import load_documents, build_faiss_index, load_index
from retriever import retrieve_answers
from model import Result
from langchain.output_parsers import PydanticOutputParser


from rapidfuzz import fuzz, process
import spacy
import streamlit as st
import os

if os.path.exists(".env"):
    load_dotenv()


# #  Auto-download fallback  
# #________________________________________________________________________________________________________________________  
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     import subprocess
#     subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
#     nlp = spacy.load("en_core_web_sm")
# # _________________________________________________________________________________________________________________________




# nlp = spacy.load("en_core_web_sm")
# def extract_keywords(text):
#     doc = nlp(text)
#     return [token.lemma_.lower() for token in doc if token.pos_ in {"NOUN", "PROPN", "NUM"} and not token.is_stop]


# def compute_rapid_fuzz(text1: str, text2: str) -> bool:
#     # Normalize whitespace and lowercase
#     t1 = text1.strip().lower()
#     t2 = text2.strip().lower()

#     # Check for substring match
#     if t1 in t2 or t2 in t1:
#         return 1

#     # Check fuzzy match
#     return fuzz.token_sort_ratio(t1, t2)/100
# def compute_keyword_fuzzy_score(pred, gold):
#     fuzz_score=compute_rapid_fuzz(pred,gold)
   
#     pred_keywords = extract_keywords(pred)
#     gold_keywords = extract_keywords(gold)
#     if gold_keywords:
#         matches=0
#         for gk in gold_keywords:
#             best = process.extractOne(gk, pred_keywords, scorer=fuzz.partial_ratio)
#             if best and best[1] > 80:
#                 matches += 1
#         key_score=matches / len(gold_keywords)
#     else:
#         key_score=0

    
#     print(fuzz_score,"-",key_score)
#     return key_score if key_score>fuzz_score else fuzz_score

# # Workaround for Windows + Python 3.12 + Streamlit + PyTorch
# # plssss workkkk
# if sys.platform == "win32" and sys.version_info >= (3, 12):
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())



# # trying somtheing new
# nlp = spacy.load("en_core_web_sm")
# def extract_keywords(text):
#     doc = nlp(text)
#     return [token.lemma_.lower() for token in doc if token.pos_ in {"NOUN", "PROPN", "NUM"} and not token.is_stop]

# def compute_rapid_fuzz(text1: str, text2: str) -> float:
#     t1 = text1.strip().lower()
#     t2 = text2.strip().lower()
#     if t1 in t2 or t2 in t1:
#         return 1
#     return fuzz.token_sort_ratio(t1, t2)/100

# def compute_keyword_fuzzy_score(pred, gold):
#     fuzz_score = compute_rapid_fuzz(pred, gold)
#     pred_keywords = extract_keywords(pred)
#     gold_keywords = extract_keywords(gold)
#     if gold_keywords:
#         matches = 0
#         for gk in gold_keywords:
#             best = process.extractOne(gk, pred_keywords, scorer=fuzz.partial_ratio)
#             if best and best[1] > 80:
#                 matches += 1
#         key_score = matches / len(gold_keywords)
#     else:
#         key_score = 0
#     return max(key_score, fuzz_score)
# #________________________________________________________________________________________________________________________



def compute_metrics(test_set, index, metadata, llm, parser,prompt):
    format_instruct=parser.get_format_instructions()
    correct = 0
    f1_scores = []
    context_utilization_scores = []
    faithfulness_scores = []
    
    for test in test_set:
        retrieved = retrieve_answers(test["query"], index, metadata,8)
        all_content=""
        relevant_retrieved_count = 0
        retrieved_count = len(retrieved)

        for doc in retrieved:
            source= doc["source"]
            content = doc["chunk"]
            if source==test["source"]:
                relevant_retrieved_count+=1
            all_content += content + "\n"
        
                  
        format_prompt = prompt.format(content=all_content, query=test["query"],format_instructions=format_instruct)
        response = llm.invoke(format_prompt)
        result = parser.parse(response.content)
        answer=result.content
        
        print("Query:", test["query"])
        print("content:",content)
        print("Expected:", test["expected"])
        print("Predicted:", answer)
        
        print("-" * 40)
        # if answer!= "No Information found":
        #     correct += 1
            
        # # Use keyword fuzzy score        
        # faithfulness = compute_keyword_fuzzy_score(answer, test["expected"])
        # if not isinstance(faithfulness, (int, float)):
        #     faithfulness = 0.0  # or any fallback threshold 

        # if faithfulness > 0.85:
        #     correct += 1
        
    
        # new
        precision = relevant_retrieved_count / retrieved_count if retrieved_count > 0 else 0
        total_relevant = sum(1 for t in test_set if t["source"] == test["source"])
        recall = relevant_retrieved_count / total_relevant if total_relevant > 0 else 0
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        # new
        
        # f1_score = relevant_retrieved_count / retrieved_count if retrieved_count > 0 else 0
        f1_scores.append(f1_score)
        
        # --- Context Utilization Score ---
        context_utilization = fuzz.token_set_ratio(answer.lower(), all_content.lower()) / 100
        context_utilization_scores.append(context_utilization)
        
        # --- Faithfulness Score ---
        faithfulness = fuzz.token_sort_ratio(answer.lower(), test["expected"].lower()) / 100
        faithfulness_scores.append(faithfulness)
        if faithfulness > 0.65:
            correct += 1
        
    accuracy = correct / len(test_set) if test_set else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_util = sum(context_utilization_scores) / len(context_utilization_scores) if context_utilization_scores else 0
    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0

    return accuracy, avg_f1, avg_util, avg_faith



# Load FAISS index
try:
    index, metadata = load_index()
except Exception as e:
    print("Index loading failed, rebuilding index...", e)
    documents = load_documents("data")
    build_faiss_index(documents)
    index, metadata = load_index()

# Input

llm = ChatGroq(model="gemma2-9b-it",temperature=0,api_key=os.getenv("GROQ_API_KEY"))
prompt = chain_of_thought_prompt()
parser=PydanticOutputParser(pydantic_object=Result)



# Expanded test set: list of (query, expected_answer)
test_set = [
    {
        "source":"flipkart1.txt",
        "query": "Who founded Flipkart?",
        "expected": "Sachin Bansal and Binny Bansal"
    },
    {
        "source":"flipkart1.txt",
        "query": "What strategic move did Flipkart make in 2014 to expand its fashion retail presence?",
        "expected": "In May 2014, Flipkart acquired Myntra, an online fashion retailer, for US$280 million. Myntra continued to operate as a standalone subsidiary."
    },
    {
        "source":"flipkart1.txt",
        "query": "Describe Flipkart’s entry into the Web3 and Metaverse space",
        "expected": "In 2022, Flipkart entered the Web3 and NFT space by enabling buyers of Nothing Phone (1) to receive NFTs via the Nothing Community Dots app on Polygon. Later, Flipkart launched a metaverse shopping platform called Flipverse in partnership with eDAO, providing a virtual mall-like experience."
    },
    {
        "source":"flipkart1.txt",
        "query": "What are two AI-driven initiatives Flipkart launched in 2024 to enhance online shopping experiences?",
        "expected": "In 2024, Flipkart launched Flippi, an AI-powered shopping assistant, and Vibes, an AI-integrated video shopping experience as part of its Swipe Screen initiative."
    },
    {
        "source":"flipkart1.txt",
        "query": "What was the outcome of Walmart’s acquisition of Flipkart in 2018, and how did it evolve?",
        "expected": "In August 2018, Walmart acquired a 77 controlling stake in Flipkart for US$16 billion, valuing it at US$20 billion. Walmart later increased its equity to 81.3 in November 2018."
    }
]



acc, avg_f1, avg_util, avg_faith = compute_metrics(test_set, index, metadata, llm,parser, prompt)
print(acc,"-",avg_f1,"-",avg_util,"-",avg_faith)

st.set_page_config(page_title="Flipkart RAG Assistant", page_icon="🛍️", layout="wide")

st.title("🛍️ Flipkart RAG Assistant")
st.markdown("Ask a business-related question about Flipkart and get a smart, context-based answer using Retrieval-Augmented Generation (RAG).")

query = st.text_area("🔍 Ask your question:", height=100)

if st.button("🚀 Get Answer") and query:
    with st.spinner("Thinking like an analyst... 🧠"):
        retrieved = retrieve_answers(query, index, metadata, 8)
        all_content = "\n".join([doc["chunk"] for doc in retrieved])
        format_instruct = parser.get_format_instructions()
        format_prompt = prompt.format(content=all_content, query=query, format_instructions=format_instruct)
        response = llm.invoke(format_prompt)
        result = parser.parse(response.content)

        st.markdown("---")
        st.markdown("### ✅ Final Answer")
        st.success(f"**{result.content}**")

        st.markdown("### 🤔 How the model thought (Chain of Thought)")
        st.info(result.reasoning)

        with st.expander("🧩 Retrieved Chunks (Click to Expand)"):
            for doc in retrieved:
                st.markdown(f"**📄 Source:** `{doc['source']}`")
                st.markdown(f"```text\n{doc['chunk']}\n```")

if st.button("📊 Run Evaluation"):
    acc, avg_f1, util_score, faith_score = compute_metrics(test_set, index, metadata, llm, parser, prompt)
    st.markdown("### 🧪 Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("F1 Score", f"{avg_f1:.2f}")
    col3.metric("Context Util.", f"{util_score:.2f}")
    col4.metric("Faithfulness", f"{faith_score:.2f}")
    
    
if os.path.exists("chunks_output.txt"):
    with open("chunks_output.txt", "r", encoding="utf-8") as f:
        st.text_area("📄 View Generated Chunks", f.read(), height=400)

    with open("chunks_output.txt", "rb") as f:
        st.download_button("⬇️ Download Chunk File", f, file_name="chunks_output.txt")


# st.title("Flipkart RAG App")
# query = st.text_area("Ask a question about Flipkart's business:", height=100)

# if st.button("Search") and query:
#     retrieved = retrieve_answers(query, index, metadata, 8)
#     all_content = "\n".join([doc["chunk"] for doc in retrieved])
#     format_instruct = parser.get_format_instructions()
#     format_prompt = prompt.format(content=all_content, query=query, format_instructions=format_instruct)
#     response = llm.invoke(format_prompt)
#     result = parser.parse(response.content)
#     answer = result.content
    
#     st.markdown("Final Answer:")
#     st.markdown(result.Answer)

#     st.markdown("### Retrieved Chunks:")
#     for doc in retrieved:
#         st.markdown(f"**Source:** `{doc['source']}`")
#         st.markdown(f"```text\n{doc['chunk']}\n```")


#     # st.markdown("**Chain of Thought Reasoning:**")
#     # st.markdown(result.content)
#     # st.markdown(f"**Final Answer:** {result.content}")

#     st.markdown("Chain of Thought:")
#     st.markdown(result.Chain_of_Thought)

# if st.button("Evaluate"):
#     acc, avg_f1, util_score, faith_score = compute_metrics(test_set, index, metadata, llm, parser, prompt)
#     st.markdown(f"**Test Set Accuracy:** {acc:.2f}")
#     st.markdown(f"**Avg F1 Score:** {avg_f1:.2f}")
#     st.markdown(f"**Avg Context Utilization Score:** {util_score:.2f}")
#     st.markdown(f"**Avg Faithfulness Score:** {faith_score:.2f}")

