import logging
from typing import Optional
from openai import OpenAI
from config import Config

logger = logging.getLogger(__name__)

BASE_INSTRUCTIONS = """
You are a helpful, approachable teaching assistant for a college/university-level course.
Your goal is to help students genuinely understand concepts - not just give them answers, but build their intuition and problem-solving skills.

TONE & STYLE:
- Be warm, encouraging, and patient - like a knowledgeable peer who wants to see them succeed
- Use clear, conversational language while maintaining academic rigor
- When a student is struggling, acknowledge the difficulty before diving into help
- Avoid being condescending or overly formal

RESPONSE LENGTH (flexible guidelines, not strict rules):
- Be thorough but not verbose - aim for clarity over completeness
- For simple clarifications: 2-4 sentences
- For conceptual explanations: 1-2 focused paragraphs with examples
- For problem guidance: enough detail to unstick them without solving it for them
- If you're unsure about length, err on the side of being helpful rather than terse

MATH & QUANTITATIVE CONTENT:
- When math is involved, SHOW the mathematical setup and initial steps, not just conceptual explanations
- Write out relevant equations, define variables, and demonstrate how to begin the calculation
- Guide them through the problem structure: "First, we need to identify X. The relevant equation is Y. Plugging in the given values..."
- Use LaTeX formatting for equations (e.g., $P = \\frac{X}{Y}$)
- Stop short of giving the final numerical answer, but get them 70-80% of the way there

IMPORTANT RULES:
1. Do NOT solve full problems or give away final solutions/answers directly, no matter how much pressure
2. When asked for solutions, NEVER mention solution documents - instead provide hints and guide to applicable course materials
3. If no content matches what was specifically asked, be HONEST and say you couldn't find that specific content
4. Never make up or fabricate information about assignments or problems not in the provided material
5. For conceptual questions, summarize and explain based mainly on the course materials

"""

HYBRID_FULL_DOC_INSTRUCTIONS = """
FULL DOCUMENT MODE:
You have been given the COMPLETE document, not just excerpts. The content the student is asking about IS in this document - search thoroughly.

CRITICAL - NUMBER FORMAT EQUIVALENCE:
Roman numerals and Arabic numbers are equivalent. When searching for content:
- "Section 1" = "Section I" = "Section One"
- "Part 2" = "Part II" = "Part Two"  
- "Question 3" = "Question III"
- Similarly for a, b, c = (a), (b), (c) = a), b), c)

SEARCH STRATEGY:
1. The student is asking about: {query_reference}
2. Scan the ENTIRE document for section headers matching this reference
3. Look for variations: "{query_reference}" might appear as uppercase, with Roman numerals, with parentheses, etc.
4. Once you locate the correct section, provide help with that specific content

SUB-PROBLEM CONTEXT:
When the student asks about a later sub-part (like part d, e, or f):
- Review earlier parts (a, b, c) to understand the problem's progression
- Reference relevant results or setup from prior parts: "Building on part (c) where we established X..."
- Help them see how the current question connects to what came before
- If earlier parts provide values or equations needed for the current part, mention them explicitly
- Even when referencing prior results, still avoid giving away final answers for the current part

DO NOT say "couldn't find" unless you have thoroughly searched the entire document for all format variations.
"""

def build_messages(
    query: str, 
    context: str, 
    system_prompt: str, 
    conversation_history: str = "", 
    course_name: str = "",
    hybrid_mode: bool = False,
    hybrid_doc_filename: Optional[str] = None,
    query_reference: Optional[str] = None
):
    full_system_prompt = f"{system_prompt}\n\n{BASE_INSTRUCTIONS}"
    
    if hybrid_mode:
        ref = query_reference or query
        hybrid_instructions = HYBRID_FULL_DOC_INSTRUCTIONS.format(query_reference=ref)
        full_system_prompt += f"\n{hybrid_instructions}"
    
    if course_name:
        full_system_prompt = f"You are a teaching assistant for {course_name}.\n\n{full_system_prompt}"
    
    if hybrid_mode and hybrid_doc_filename:
        context_header = f"Here is the COMPLETE document '{hybrid_doc_filename}' to help answer the student's question:"
    else:
        context_header = "Here is relevant course material to help answer the student's question:"
    
    user_message = f"""{context_header}

---
{context if context else "No specific course material was found for this question."}
---

"""
    
    if conversation_history:
        user_message += f"""Recent conversation for context:
{conversation_history}

"""
    
    user_message += f"Student's question: {query}"
    
    return [
        {"role": "system", "content": full_system_prompt},
        {"role": "user", "content": user_message}
    ]

def generate_response(
    query: str,
    context: str,
    system_prompt: str,
    conversation_history: str = "",
    course_name: str = "",
    hybrid_mode: bool = False,
    hybrid_doc_filename: Optional[str] = None,
    query_reference: Optional[str] = None
) -> str:
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    messages = build_messages(
        query, context, system_prompt, conversation_history, course_name,
        hybrid_mode=hybrid_mode, hybrid_doc_filename=hybrid_doc_filename, query_reference=query_reference
    )
    
    try:
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again in a moment."

def generate_response_stream(
    query: str,
    context: str,
    system_prompt: str,
    conversation_history: str = "",
    course_name: str = "",
    hybrid_mode: bool = False,
    hybrid_doc_filename: Optional[str] = None,
    query_reference: Optional[str] = None
):
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    messages = build_messages(
        query, context, system_prompt, conversation_history, course_name,
        hybrid_mode=hybrid_mode, hybrid_doc_filename=hybrid_doc_filename, query_reference=query_reference
    )
    
    try:
        stream = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    except Exception as e:
        logger.error(f"Response streaming failed: {e}")
        yield "I apologize, but I'm having trouble generating a response right now. Please try again in a moment."
