"""
Agent functions for the Agentic RAG System
Contains all individual agent implementations for the multi-step workflow
"""

import os
from typing import TypedDict, List
import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AgentState(TypedDict):
    """State management for the agentic workflow"""
    question: str
    chat_history: List[str]
    rewritten_question: str
    context: List[str]
    answer: str
    iteration_count: int
    evaluation_feedback: str
    needs_refinement: bool

class AgenticRAGAgents:
    """Collection of all agents in the RAG pipeline"""
    
    def __init__(self, retriever=None):
        self.retriever = retriever
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            api_key=api_key
        )
    
    def input_guard(self, state: AgentState) -> AgentState:
        """Security validation agent - checks for malicious inputs"""
        blocked_patterns = ["ignore previous instructions", "jailbreak", "override safety", "bypass filters"]
        question = state["question"].lower()
        
        for pattern in blocked_patterns:
            if pattern in question:
                return {**state, "answer": "⚠️ Query blocked due to unsafe instruction."}
        
        if len(state["question"].strip()) < 3:
            return {**state, "answer": "❓ Please provide a more detailed question."}
        
        return state
    
    def rewrite_query(self, state: AgentState) -> AgentState:
        """Query optimization agent - improves queries for better retrieval"""
        is_refinement = state.get('iteration_count', 0) > 0
        feedback = state.get('evaluation_feedback', '')
        chat_history = "\n".join(state.get('chat_history', [])[-4:])  # Last 2 exchanges
        
        if is_refinement and feedback:
            prompt = f"""
            The previous answer was evaluated and needs improvement. Rewrite the query to address the identified issues:

            Original Question: {state['question']}
            Previous Rewritten Query: {state.get('rewritten_question', state['question'])}
            Evaluation Feedback: {feedback}
            Recent Conversation: {chat_history}
            
            Create a more targeted and specific query that addresses the feedback. Focus on:
            - Using different keywords that might appear in the documents
            - Being more specific about what information is needed
            - Avoiding generic placeholders - work with what the user actually asked
            
            Improved Query:
            """
        else:
            prompt = f"""
            The user has uploaded documents and is asking a question about them. 
            Rewrite this question to better find relevant information from the uploaded documents.

            Original Question: {state['question']}
            Recent Conversation: {chat_history}
            
            Guidelines for rewriting:
            - Keep the core intent of the original question
            - Add synonyms and related terms that might appear in documents  
            - Focus on key information the user wants (dates, requirements, deadlines, etc.)
            - DO NOT add generic placeholders like [company name] or [job title]
            - Make it specific to what the user is actually asking about
            - Consider document types: application forms, job postings, announcements, etc.
            
            Rewritten Query:
            """

        try:
            response = self.llm.invoke(prompt)
            rewritten_question = response.content.strip()
        except Exception as e:
            st.error(f"Query rewriting failed: {str(e)}")
            rewritten_question = state["question"]
        
        return {**state, "rewritten_question": rewritten_question}
    
    def retrieve_docs(self, state: AgentState) -> AgentState:
        """Document retrieval agent - finds relevant context"""
        if not self.retriever:
            return {**state, "context": ["No retriever configured"]}
        
        try:
            docs = self.retriever.invoke(state["rewritten_question"])
            context = [doc.page_content for doc in docs]
        except Exception as e:
            st.error(f"Document retrieval failed: {str(e)}")
            context = ["Document retrieval failed"]
        
        return {**state, "context": context}
    
    def generate_answer(self, state: AgentState) -> AgentState:
        """Answer generation agent - creates responses based on context"""
        history = "\n".join(state["chat_history"][-6:])
        context = "\n\n".join(state["context"])

        prompt = f"""
        You are a helpful AI assistant. Answer the question using the provided context. 
        Use your knowledge to provide helpful responses based on the context.
        If context contains relevant information, provide a comprehensive answer.
        Only say "Information not found in documents" if the context is completely unrelated to the question.

        Conversation History:
        {history}

        Context:
        {context}

        Question:
        {state['question']}

        Provide a clear, comprehensive answer based on the available information:
        """

        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip()
        except Exception as e:
            st.error(f"Answer generation failed: {str(e)}")
            answer = "Sorry, I encountered an error while generating the answer."

        return {**state, "answer": answer}
    
    def grounding_check(self, state: AgentState) -> AgentState:
        """Grounding validation agent - ensures answer accuracy"""
        answer = state['answer']
        
        if any(phrase in answer.lower() for phrase in [
            "not found in documents", "information not found", "no relevant information"
        ]):
            return state
        
        if len(answer) < 200:
            return state
            
        context_str = " ".join(state['context'][:3])
        validation_prompt = f"""
        Does the answer contain any information that directly contradicts or is completely unrelated to the provided context?
        Only respond "CONTRADICTS" if there are clear factual errors or completely made-up information.
        Otherwise respond "ACCEPTABLE".

        Context: {context_str[:800]}...
        Answer: {answer[:500]}...
        
        Response (CONTRADICTS or ACCEPTABLE):
        """

        try:
            result = self.llm.invoke(validation_prompt)
            if "CONTRADICTS" in result.content.upper():
                return {**state, "answer": "⚠️ Answer may contain inaccurate information based on the available context."}
        except Exception:
            pass

        return state
    
    def evaluate_answer(self, state: AgentState) -> AgentState:
        """Answer quality evaluation agent - assesses and determines refinement needs"""
        evaluation_prompt = f"""
        You are an expert evaluator. Assess the quality of the following answer to determine if it needs refinement.

        Original Question: {state['question']}
        Rewritten Query: {state['rewritten_question']}  
        Generated Answer: {state['answer']}
        Current Iteration: {state.get('iteration_count', 0)}

        Evaluation Guidelines:
        1. Focus ONLY on whether the answer addresses what the user actually asked
        2. Do not expect additional information that the user did not request
        3. If the user asked a simple, specific question, a direct answer is sufficient
        4. Only suggest REFINE if the answer is clearly inadequate, vague, or doesn't address the question

        Evaluation Criteria:
        - Relevance: Does it answer the specific question asked?
        - Accuracy: Is the information correct based on available context?
        - Completeness: Does it address what was specifically asked (not more)?
        - Clarity: Is it clear and understandable?

        Examples of when to ACCEPT:
        - User asks "What is the deadline?" → Answer provides the deadline = ACCEPT
        - User asks "What are the requirements?" → Answer lists requirements = ACCEPT
        - User asks specific question → Answer provides specific information = ACCEPT

        Examples of when to REFINE:
        - Answer is vague or unclear
        - Answer doesn't address the question at all
        - Answer contains obvious errors
        - Answer says "information not found" when it should be available

        Based on these criteria, determine:
        DECISION: [ACCEPT/REFINE] 
        FEEDBACK: [Brief explanation]

        Remember: If the answer directly addresses what the user asked, it should be ACCEPTED even if it doesn't include additional related information they didn't request.
        """

        try:
            response = self.llm.invoke(evaluation_prompt)
            evaluation_result = response.content
        except Exception:
            evaluation_result = "DECISION: ACCEPT\nFEEDBACK: Evaluation service unavailable, accepting current answer."

        lines = evaluation_result.strip().split('\n')
        decision = "ACCEPT"
        feedback = "Answer meets quality standards."
        
        for line in lines:
            if line.startswith("DECISION:"):
                decision = line.split(":", 1)[1].strip()
            elif line.startswith("FEEDBACK:"):
                feedback = line.split(":", 1)[1].strip()

        current_iteration = state.get('iteration_count', 0)
        max_iterations = 2
        
        needs_refinement = (
            decision == "REFINE" and 
            current_iteration < max_iterations and
            len(state.get('answer', '')) > 0 and
            "error" not in state.get('answer', '').lower()
        )

        return {
            **state,
            "evaluation_feedback": feedback,
            "needs_refinement": needs_refinement,
            "iteration_count": current_iteration + (1 if needs_refinement else 0)
        }
    
    def output_guard(self, state: AgentState) -> AgentState:
        """Output filtering agent - final safety checks"""
        banned_words = ["confidential", "private data", "classified", "sensitive information"]
        answer = state["answer"].lower()
        
        for word in banned_words:
            if word in answer:
                return {**state, "answer": "⚠️ Response blocked due to policy violation."}
        
        return state
    
    def update_memory(self, state: AgentState) -> AgentState:
        """Memory management agent - updates conversation history"""
        updated_history = state["chat_history"] + [
            f"User: {state['question']}",
            f"Assistant: {state['answer']}"
        ]
        
        if len(updated_history) > 20:
            updated_history = updated_history[-20:]
        
        return {**state, "chat_history": updated_history}

def create_initial_state(question: str, chat_history: List[str] = None) -> AgentState:
    """Create initial state for the workflow"""
    if chat_history is None:
        chat_history = []
    
    return {
        "question": question,
        "chat_history": chat_history,
        "rewritten_question": "",
        "context": [],
        "answer": "",
        "iteration_count": 0,
        "evaluation_feedback": "",
        "needs_refinement": False
    }