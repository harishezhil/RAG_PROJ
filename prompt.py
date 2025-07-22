from langchain.prompts import PromptTemplate

def chain_of_thought_prompt():
    prompt = PromptTemplate(
        input_variables=["query", "content","format_instructions"],
        template=(
        """
            You are a business analyst assistant. You need answer the given question using only the context provided. 

            Question: {query}

            Context: {content}

            Output Format Specification:
            {format_instructions}

            Innstruction:
            - Think step-by-step to deduce the answer from the given context.
            - If no suitable answer is found, return content as "No Information found" .
            - **Return only the JSON Output**. Do not include any explanation, justification, or extra commentary.
            - The final JSON output must be valid **JSON** and enclosed in a `json` code block.
            - json code block format:
                ```json
                <your_json_output_here>
            - Ensure you follow the structure and keys exactly as described.
     

        """
           
        )
    )
    return prompt
