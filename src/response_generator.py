import logging
from openai import OpenAI
from config import Config

logger = logging.getLogger(__name__)

def generate_response(
    query: str,
    context: str,
    system_prompt: str,
    conversation_history: str = "",
    course_name: str = ""
) -> str:
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    base_instructions = """
IMPORTANT GUIDELINES:
1. You are a teaching assistant, not a solution provider. Help students understand concepts and guide them through problem-solving.
2. For homework problems, explain the approach and concepts needed, but do not give direct answers.
3. For exam questions, help students understand what knowledge is being tested and how to approach such problems.
4. If the context doesn't contain relevant information, acknowledge this and explain what you do know about the topic.
5. Be encouraging and supportive. Learning is a process.
6. Use clear, simple language and provide examples when helpful.
7. If a student seems stuck, ask guiding questions rather than providing answers.
"""
    
    full_system_prompt = f"{system_prompt}\n\n{base_instructions}"
    
    if course_name:
        full_system_prompt = f"You are a teaching assistant for {course_name}.\n\n{full_system_prompt}"
    
    user_message = f"""Here is relevant course material to help answer the student's question:

---
{context if context else "No specific course material was found for this question."}
---

"""
    
    if conversation_history:
        user_message += f"""Recent conversation for context:
{conversation_history}

"""
    
    user_message += f"Student's question: {query}"
    
    try:
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again in a moment."
