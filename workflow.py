"""
Workflow management using LangGraph for the Agentic RAG System
Orchestrates the coordinated AI pipeline with conditional logic and iterative refinement
"""

from langgraph.graph import StateGraph, END
from agents import AgenticRAGAgents, AgentState

class AgenticRAGWorkflow:
    """
    LangGraph workflow manager for the agentic RAG system
    Coordinates multiple agents with iterative refinement capabilities
    """
    
    def __init__(self, retriever=None):
        self.retriever = retriever
        self.agents = AgenticRAGAgents(retriever)
        self.app = None
        self._build_workflow()
    
    def _build_workflow(self):
        """Build the LangGraph workflow with all agents and conditional logic"""
        
        workflow = StateGraph(AgentState)
        
        # Add all agent nodes
        workflow.add_node("input_guard", self.agents.input_guard)
        workflow.add_node("rewrite", self.agents.rewrite_query)
        workflow.add_node("retrieve", self.agents.retrieve_docs)
        workflow.add_node("generate", self.agents.generate_answer)
        workflow.add_node("grounding", self.agents.grounding_check)
        workflow.add_node("evaluate", self.agents.evaluate_answer)
        workflow.add_node("output_guard", self.agents.output_guard)
        workflow.add_node("memory", self.agents.update_memory)
        
        # Set entry point
        workflow.set_entry_point("input_guard")
        
        # Define linear workflow edges
        workflow.add_edge("input_guard", "rewrite")
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "grounding")
        workflow.add_edge("grounding", "evaluate")
        
        # Conditional edge for iterative refinement
        workflow.add_conditional_edges(
            "evaluate",
            self._should_refine,
            {
                "refine": "rewrite",      # Loop back for refinement
                "finish": "output_guard"  # Proceed to end
            }
        )
        
        workflow.add_edge("output_guard", "memory")
        workflow.add_edge("memory", END)
        
        # Compile the workflow
        self.app = workflow.compile()
    
    def _should_refine(self, state: AgentState) -> str:
        """Decision function for iterative refinement"""
        answer = state.get("answer", "")
        if any(phrase in answer for phrase in ["blocked", "error", "failed"]):
            return "finish"
        
        if state.get("needs_refinement", False):
            return "refine"
        else:
            return "finish"
    
    def update_retriever(self, retriever):
        """Update the retriever for all agents"""
        self.retriever = retriever
        self.agents.retriever = retriever
    
    def invoke(self, initial_state: AgentState) -> AgentState:
        """Execute the complete agentic workflow"""
        if not self.app:
            raise ValueError("Workflow not properly initialized")
        
        try:
            result = self.app.invoke(initial_state)
            return result
        except Exception as e:
            return {
                **initial_state,
                "answer": f"⚠️ Workflow execution failed: {str(e)}",
                "evaluation_feedback": "System error occurred",
                "needs_refinement": False
            }
    
    def get_workflow_visualization(self) -> str:
        """Return a text representation of the workflow"""
        return """
        🤖 Agentic RAG Workflow:
        
        1. 🛡️  Input Guard      → Security validation
        2. ✏️  Query Rewrite    → Optimization for retrieval  
        3. 🔍 Document Retrieve → Vector similarity search
        4. 🧠 Answer Generate   → Context-based response
        5. ✅ Grounding Check  → Accuracy validation
        6. 📊 Evaluate Answer  → Quality assessment
           ↩️  [Refinement Loop] → Back to step 2 if needed
        7. 🛡️  Output Guard     → Final safety check
        8. 💭 Memory Update    → History management
        
        ✨ Features:
        • Up to 2 refinement iterations
        • Autonomous decision-making
        • Quality-driven improvements
        • Comprehensive safety controls
        """

def create_workflow(retriever=None) -> AgenticRAGWorkflow:
    """Create and return a configured workflow instance"""
    return AgenticRAGWorkflow(retriever)