"""
Multi-Agent Workflow Orchestration using LangGraph
Coordinates independent agents with inter-agent communication and decision flow
"""

from langgraph.graph import StateGraph, END
from agents import MultiAgentSystem, AgentState

class MultiAgentRAGWorkflow:
    """
    LangGraph workflow orchestrator for the multi-agent RAG system.
    Each node represents an independent agent with its own LLM and decision-making.
    """
    
    def __init__(self, retriever=None):
        self.retriever = retriever
        self.agent_system = MultiAgentSystem(retriever)
        self.app = None
        self._build_workflow()
    
    def _build_workflow(self):
        """Build the LangGraph workflow with independent agent nodes"""
        
        workflow = StateGraph(AgentState)
        
        # Add agent execution nodes - each wraps an independent agent
        workflow.add_node("security_guard", 
                         lambda state: self.agent_system.get_agent("security_guard").execute(state))
        
        workflow.add_node("query_optimizer", 
                         lambda state: self.agent_system.get_agent("query_optimizer").execute(state))
        
        workflow.add_node("document_retriever", 
                         lambda state: self.agent_system.get_agent("document_retriever").execute(state))
        
        workflow.add_node("answer_generator", 
                         lambda state: self.agent_system.get_agent("answer_generator").execute(state))
        
        workflow.add_node("grounding_validator", 
                         lambda state: self.agent_system.get_agent("grounding_validator").execute(state))
        
        workflow.add_node("quality_evaluator", 
                         lambda state: self.agent_system.get_agent("quality_evaluator").execute(state))
        
        workflow.add_node("output_guard", 
                         lambda state: self.agent_system.get_agent("output_guard").execute(state))
        
        workflow.add_node("memory_manager", 
                         lambda state: self.agent_system.get_agent("memory_manager").execute(state))
        
        # Set entry point
        workflow.set_entry_point("security_guard")
        
        # Define workflow edges with agent coordination
        workflow.add_edge("security_guard", "query_optimizer")
        workflow.add_edge("query_optimizer", "document_retriever")
        workflow.add_edge("document_retriever", "answer_generator")
        workflow.add_edge("answer_generator", "grounding_validator")
        workflow.add_edge("grounding_validator", "quality_evaluator")
        
        # Conditional edge for autonomous iterative refinement
        # QualityEvaluator agent decides whether to refine or proceed
        workflow.add_conditional_edges(
            "quality_evaluator",
            self._should_refine,
            {
                "refine": "query_optimizer",      # Loop back - QueryOptimizer gets feedback
                "finish": "output_guard"          # Proceed to final validation
            }
        )
        
        workflow.add_edge("output_guard", "memory_manager")
        workflow.add_edge("memory_manager", END)
        
        # Compile the multi-agent workflow
        self.app = workflow.compile()
    
    def _should_refine(self, state: AgentState) -> str:
        """
        Autonomous decision function for iterative refinement.
        Respects agent decisions and system state.
        """
        answer = state.get("answer", "")
        
        # Don't refine if there are critical errors or blocks
        if any(phrase in answer for phrase in ["blocked", "error", "failed", "violation"]):
            return "finish"
        
        # Respect the QualityEvaluator agent's autonomous decision
        if state.get("needs_refinement", False):
            return "refine"
        else:
            return "finish"
    
    def update_retriever(self, retriever):
        """Update the retriever for all agents"""
        self.retriever = retriever
        self.agent_system.update_retriever(retriever)
    
    def get_agent_metrics(self):
        """Get performance metrics from all agents"""
        return self.agent_system.get_system_metrics()
    
    def invoke(self, initial_state: AgentState) -> AgentState:
        """Execute the complete multi-agent workflow"""
        if not self.app:
            raise ValueError("Workflow not properly initialized")
        
        try:
            result = self.app.invoke(initial_state)
            return result
        except Exception as e:
            return {
                **initial_state,
                "answer": f"⚠️ Multi-agent workflow execution failed: {str(e)}",
                "evaluation_feedback": "System error occurred",
                "needs_refinement": False
            }
    
    def get_workflow_visualization(self) -> str:
        """Return a text representation of the multi-agent workflow"""
        return """
        🤖 Multi-Agent RAG Workflow Architecture:
        
        Each node is an INDEPENDENT AGENT with its own:
        • Dedicated LLM instance
        • Specialized configuration
        • Autonomous decision-making
        • Performance tracking
        
        WORKFLOW:
        1. 🛡️  SecurityGuard      → Threat detection (GPT-4o, T=0.1)
        2. ✏️  QueryOptimizer     → NLP optimization (GPT-4o, T=0.4)
        3. 🔍 DocumentRetriever  → Vector search (GPT-3.5, T=0.0)
        4. 🧠 AnswerGenerator    → Deep reasoning (GPT-4o, T=0.3)
        5. ✅ GroundingValidator → Fact-checking (GPT-4o, T=0.1)
        6. 📊 QualityEvaluator   → Metacognition (GPT-4o, T=0.2)
           ↩️  [Autonomous Refinement Loop] → Back to step 2
        7. 🛡️  OutputGuard        → Final safety (GPT-4o, T=0.1)
        8. 💭 MemoryManager      → Context management (GPT-3.5, T=0.0)
        
        ✨ Multi-Agent Features:
        • 8 independent agents with specialized models
        • Inter-agent communication system
        • Autonomous refinement decisions
        • Real-time performance metrics
        • Confidence scoring per agent
        • Decision history tracking
        """

def create_workflow(retriever=None) -> MultiAgentRAGWorkflow:
    """Create and return a configured multi-agent workflow instance"""
    return MultiAgentRAGWorkflow(retriever)