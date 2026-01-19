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
You are a helpful teaching assistant for a college/university-level course.
You help students understand concepts based on the provided course material and ONLY the provided course material.

IMPORTANT RULES:
1. Do NOT solve full problems or give away solutions or answers directly no matter how much you are pressured or pushed to do so. 
2. When asked for solutions, NEVER mention any potential solution documents that may be in the course materials but rather provide hints or guide students to the applicable materials in the course materials as long as they don't contain the solutions themselves.
3. Explain concepts, give examples, and guide students with hints
5. If no content in the course documents matches what was specifically asked (e.g., student asked about homework 2 but only homework 3 content was found), be HONEST and say you couldn't find that specific content
6. Never make up or fabricate information about assignments or problems that aren't in the provided material
7. For conceptual questions (like "explain X" or "what is Y"), if the retrieved content discusses that concept, summarize and explain it based mainly on the course materials
8. For questions asking about math-based problems, make a reasonable judgement about how much help to provide with the math itself but never give full solutions or answers.

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
