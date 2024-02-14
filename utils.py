def generate_prompt_sql(input_question, context, output=""):
    """
    Generates a prompt for fine-tuning the LLM model for text-to-SQL tasks.

    Parameters:
        input_question (str): The input text or question to be converted to SQL.
        context (str): The schema or context in which the SQL query operates.
        output (str, optional): The expected SQL query as the output.

    Returns:
        str: A formatted string serving as the prompt for the fine-tuning task.
    """
    return f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. 
    You are given a question and context regarding one or more tables. 
    You must output the SQL query that answers the question.
    ### Input:{input_question}
    ### Context:{context}
    ### Response:{output}"""

def _extract_sections(output):
    input_section = output.split("### Input:")[1].split("### Context:")[0]
    context_section = output.split("### Context:")[1].split("### Response:")[0]
    response_section = output.split("### Response:")[1]
    return input_section, context_section, response_section