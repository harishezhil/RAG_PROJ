from langchain.prompts import PromptTemplate

def chain_of_thought_prompt():
    prompt = PromptTemplate(
        input_variables=["query", "content", "format_instructions"],
        template=(
            """
            You are a business analyst assistant. You are given a user question and related context from multiple documents (chunks).

            Your job is to:
            1. Think step-by-step about the question and the context.
            2. Clearly state the final answer FIRST.
            3. Then explain in detail how you arrived at that answer in the "reasoning" field.

            Instructions:
            - If the answer is "No", say "No" and explain why it's incorrect or unsupported using the chunks.
            - If the answer is "Yes", provide a complete, accurate answer with supporting evidence.
            - If the answer is not directly stated but can be inferred, use logic and context to answer.
            - NEVER respond with "No Information found".
            - ALWAYS give a confident answer based on the best available evidence from the context.

            Your final output must follow this JSON format and be wrapped in a code block:
            ```json
            {{
              "content": "<final answer (Yes/No + explanation)>",
              "reasoning": "<step-by-step reasoning showing how the answer was derived, what chunks were used, and why>"
            }}
            ```

            Question:
            {query}

            Context:
            {content}

            Output Format Specification:
            {format_instructions}

            Make sure your response:
            1. Starts with the final answer in the 'content' field.
            2. Shows clear reasoning in the 'reasoning' field.
            3. Uses only the context above — do not use outside knowledge.
            """
        )
    )
    return prompt



# from langchain.prompts import PromptTemplate

# def chain_of_thought_prompt():
#     prompt = PromptTemplate(
#         input_variables=["query", "content", "format_instructions"],
#         template=(
#     """
#         You are a business analyst assistant. Answer the given question using only the context provided.

#         Question: {query}

#         Context: {content}

#         Output Format Specification:
#         {format_instructions}

#         Instructions:
#         - Carefully analyze the entire context provided above.
#         - Identify all relevant facts across the retrieved sections.
#         - Think step-by-step using a clear chain-of-thought style to build your answer.

#         - Clearly explain:
#             * Which chunks (sources) you used
#             * What specific information from them supports your answer
#             * How you connected those facts to answer the question
#             * Why other chunks were ignored (if applicable)
#         - You must combine facts from multiple parts of the context if needed.
#         - The answer may require synthesizing data spread across different sources or chunks.
#         - Include a "reasoning" field in the JSON where you show step-by-step thoughts.
#         - Avoid copying directly unless fully supported by context.
#             - Include a "reasoning" field that explains:
#               * What you identified from which chunks
#               * How you logically connected it to the question
#               * Why this answer is correct
#             - Your final answer should go into the "content" field.
#         - Final output must look like this:
#         ```json
#         {{
#             "content": "<final answer>",
#             "reasoning": "<step-by-step explanation of how you derived the answer>"
#         }}
#         ```
#         - If no suitable answer is found, return content as "No Information found".
#         - **Return only the JSON Output**. Do not include any explanation, justification, or extra commentary.
#         - The final JSON output must be valid **JSON** and enclosed in a `json` code block.

#         Reasoning Steps:
#             1. Did you locate relevant content?
#             2. Did you connect multiple chunks if needed?
#             3. Did you explain how the context supports your answer?
#             4. Did you return a clear answer inside 'content'?
#     """
# )

#     )
#     return prompt

#                 # 1. Identify relevant information from the context.
#                 # 2. Analyze how the information answers the question.
#                 # 3. Formulate the answer in the specified JSON format.

#         # 1. Identify all relevant facts across the provided sections.
#         # 2. Merge related data if it spans multiple chunks (e.g., date + event).
#         # 3. Explain the logic inside the "reasoning" field.
#         # # 4. Return final conclusion in "content".