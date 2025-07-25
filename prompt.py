from langchain.prompts import PromptTemplate

def chain_of_thought_prompt():
    prompt = PromptTemplate(
        input_variables=["query", "content", "format_instructions"],
        template=(
            """
            You are a business analyst assistant. You are given a user question and related context from multiple documents (chunks).

            Your job is to:
            1. Think step-by-step about the question and the context.
            2. Clearly state the final answer FIRST in the "content" field.
            3. Then explain in detail how you arrived at that answer in the "reasoning" field.

            Instructions:
            - Base your answer entirely on the provided context. Do not use outside knowledge.
            - Combine facts across multiple chunks if necessary.
            - If the question includes a false assumption (e.g. wrong year, incorrect claim, missing fact):
                - Explicitly contradict the assumption (e.g., "No, that did not happen")
                - Then explain the correct fact using the context.
            - If the answer is yes, provide evidence from the context and clarify.
            - If the answer is no, explain why it's incorrect or unsupported using the retrieved chunks.
            - AVOID responding with "No Information found".
            - ALWAYS make a decision and support it using the available evidence.

            Example format:
            ```json
            {{
              "content": "No, Flipkart did not launch a Web3 platform in 2023. Instead, it launched Flipverse in 2022.",
              "reasoning": "Chunk 3 from flipkart5.txt states that Flipkart entered the Metaverse in 2022. No evidence supports a 2023 launch."
            }}
            ```

            Your output must follow this JSON structure and be wrapped in a code block:
            ```json
            {{
              "content": "<final answer>",
              "reasoning": "<step-by-step justification, including which chunks you used and why>"
            }}
            ```

            Question:
            {query}

            Context:
            {content}

            Output Format Specification:
            {format_instructions}
            """
        )
    )
    return prompt
