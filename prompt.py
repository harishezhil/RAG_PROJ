from langchain.prompts import PromptTemplate

def chain_of_thought_prompt():
    prompt = PromptTemplate(
        input_variables=["query", "content", "format_instructions"],
        template=(
    """
        You are a business analyst assistant. Answer the given question using only the context provided.

        Question: {query}

        Context: {content}

        Output Format Specification:
        {format_instructions}

        Instructions:
        - You must combine facts from multiple parts of the context if needed.
        - The answer may require synthesizing data spread across different sources or chunks.
        - Use reasoning to deduce answers, not just copy exact lines.
        - Think step-by-step (chain of thought) to deduce the answer from the given context.
        - Include a "reasoning" field in the JSON where you show step-by-step thoughts.
        - Final output must look like this:
        ```json
        {{
            "content": "<final answer>",
            "reasoning": "<why this answer was chosen, what chunks were used, and how the answer was formed>"
        }}
        ```
        - If no suitable answer is found, return content as "No Information found".
        - **Return only the JSON Output**. Do not include any explanation, justification, or extra commentary.
        - The final JSON output must be valid **JSON** and enclosed in a `json` code block.

        Reasoning Steps:
        1. Identify all relevant facts across the provided sections.
        2. Merge related data if it spans multiple chunks (e.g., date + event).
        3. Explain the logic inside the "reasoning" field.
        4. Return final conclusion in "content".
    """
)

    )
    return prompt

                # 1. Identify relevant information from the context.
                # 2. Analyze how the information answers the question.
                # 3. Formulate the answer in the specified JSON format.
