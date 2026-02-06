"""Framework-specific feedback extractors for GEPA optimization.

This module provides specialized feedback extraction logic for different
AI frameworks supported by Optimas, enabling richer reflection data for
GEPA optimization.
"""

from typing import Any, Dict, Optional
from optimas.optim.gepa_adapter import FeedbackExtractor, ComponentTrace


class CrewAIFeedbackExtractor:
    """Feedback extractor for CrewAI components."""
    
    def extract_feedback(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        score: float,
        trace: Optional[ComponentTrace] = None,
        error: Optional[Exception] = None
    ) -> str:
        """Extract CrewAI-specific feedback from component execution."""
        feedback_parts = [f"Performance Score: {score:.3f}"]
        
        # Add task and response information
        if inputs:
            task_info = self._extract_task_info(inputs)
            if task_info:
                feedback_parts.append(f"Task: {task_info}")
        
        if outputs:
            response_info = self._extract_response_info(outputs)
            if response_info:
                feedback_parts.append(f"Agent Response: {response_info}")
        
        # Add agent reasoning if available
        if trace and hasattr(trace, 'metadata'):
            reasoning = trace.metadata.get('agent_reasoning', '')
            if reasoning:
                feedback_parts.append(f"Agent Reasoning: {reasoning}")
            
            tools_used = trace.metadata.get('tools_used', [])
            if tools_used:
                feedback_parts.append(f"Tools Used: {', '.join(tools_used)}")
        
        # Add error information
        if error:
            feedback_parts.append(f"Execution Error: {str(error)}")
        
        # Add performance assessment
        performance_assessment = self._assess_performance(score, outputs, error)
        if performance_assessment:
            feedback_parts.append(f"Assessment: {performance_assessment}")
        
        return " | ".join(feedback_parts)
    
    def _extract_task_info(self, inputs: Dict[str, Any]) -> str:
        """Extract meaningful task information from inputs."""
        # Common input field names for tasks
        task_fields = ['task', 'query', 'question', 'input', 'request']
        
        for field in task_fields:
            if field in inputs:
                task_text = str(inputs[field])
                return task_text[:200] + "..." if len(task_text) > 200 else task_text
        
        # Fallback: concatenate all inputs
        if inputs:
            combined = " ".join(str(v) for v in inputs.values())
            return combined[:200] + "..." if len(combined) > 200 else combined
        
        return ""
    
    def _extract_response_info(self, outputs: Dict[str, Any]) -> str:
        """Extract meaningful response information from outputs."""
        # Common output field names
        response_fields = ['output', 'response', 'answer', 'result', 'content']
        
        for field in response_fields:
            if field in outputs:
                response_text = str(outputs[field])
                return response_text[:300] + "..." if len(response_text) > 300 else response_text
        
        # Fallback: concatenate all outputs
        if outputs:
            combined = " ".join(str(v) for v in outputs.values())
            return combined[:300] + "..." if len(combined) > 300 else combined
        
        return ""
    
    def _assess_performance(self, score: float, outputs: Dict[str, Any], error: Optional[Exception]) -> str:
        """Provide performance assessment for feedback."""
        if error:
            return "Task failed with error - agent needs better error handling or clearer instructions"
        
        if score >= 0.8:
            return "Excellent performance - agent handled task well"
        elif score >= 0.6:
            return "Good performance - some room for improvement in agent response quality"
        elif score >= 0.4:
            return "Fair performance - agent partially understood task but needs better guidance"
        elif score >= 0.2:
            return "Poor performance - agent struggled with task, needs clearer instructions or better context"
        else:
            return "Very poor performance - agent failed to understand or complete task properly"


class OpenAIFeedbackExtractor:
    """Feedback extractor for OpenAI Agent components."""
    
    def extract_feedback(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        score: float,
        trace: Optional[ComponentTrace] = None,
        error: Optional[Exception] = None
    ) -> str:
        """Extract OpenAI Agent-specific feedback from component execution."""
        feedback_parts = [f"Performance Score: {score:.3f}"]
        
        # Add input/output analysis
        if inputs:
            input_analysis = self._analyze_inputs(inputs)
            if input_analysis:
                feedback_parts.append(f"Input Analysis: {input_analysis}")
        
        if outputs:
            output_analysis = self._analyze_outputs(outputs)
            if output_analysis:
                feedback_parts.append(f"Output Analysis: {output_analysis}")
        
        # Add model behavior insights
        if trace and hasattr(trace, 'metadata'):
            model_info = trace.metadata.get('model_behavior', '')
            if model_info:
                feedback_parts.append(f"Model Behavior: {model_info}")
            
            function_calls = trace.metadata.get('function_calls', [])
            if function_calls:
                feedback_parts.append(f"Function Calls: {', '.join(function_calls)}")
        
        # Add error analysis
        if error:
            error_analysis = self._analyze_error(error)
            feedback_parts.append(f"Error Analysis: {error_analysis}")
        
        # Add improvement suggestions
        improvement_suggestion = self._suggest_improvements(score, outputs, error)
        if improvement_suggestion:
            feedback_parts.append(f"Improvement Suggestion: {improvement_suggestion}")
        
        return " | ".join(feedback_parts)
    
    def _analyze_inputs(self, inputs: Dict[str, Any]) -> str:
        """Analyze input characteristics."""
        analysis_parts = []
        
        # Check input complexity
        total_length = sum(len(str(v)) for v in inputs.values())
        if total_length > 1000:
            analysis_parts.append("complex/lengthy input")
        elif total_length < 50:
            analysis_parts.append("simple/short input")
        
        # Check for specific input types
        if any('question' in k.lower() for k in inputs.keys()):
            analysis_parts.append("question-answering task")
        if any('code' in str(v).lower() for v in inputs.values()):
            analysis_parts.append("involves code")
        if any('data' in k.lower() for k in inputs.keys()):
            analysis_parts.append("data processing task")
        
        return ", ".join(analysis_parts) if analysis_parts else "standard input"
    
    def _analyze_outputs(self, outputs: Dict[str, Any]) -> str:
        """Analyze output characteristics."""
        analysis_parts = []
        
        # Check output length and structure
        for key, value in outputs.items():
            value_str = str(value)
            if len(value_str) > 500:
                analysis_parts.append(f"{key}: detailed response")
            elif len(value_str) < 20:
                analysis_parts.append(f"{key}: brief response")
            
            # Check for structured content
            if value_str.count('\n') > 3:
                analysis_parts.append(f"{key}: structured/multi-line")
            if any(marker in value_str.lower() for marker in ['```', 'json', 'xml']):
                analysis_parts.append(f"{key}: contains formatted content")
        
        return ", ".join(analysis_parts) if analysis_parts else "standard output"
    
    def _analyze_error(self, error: Exception) -> str:
        """Analyze error for actionable insights."""
        error_str = str(error).lower()
        
        if 'timeout' in error_str:
            return "Request timeout - consider shorter instructions or simpler tasks"
        elif 'rate limit' in error_str:
            return "Rate limit exceeded - implement backoff strategy"
        elif 'token' in error_str:
            return "Token limit issues - instructions may be too long"
        elif 'format' in error_str or 'parse' in error_str:
            return "Output formatting issues - clarify expected response format"
        elif 'permission' in error_str or 'auth' in error_str:
            return "Authentication/permission issues - check API configuration"
        else:
            return f"General error: {str(error)[:100]}"
    
    def _suggest_improvements(self, score: float, outputs: Dict[str, Any], error: Optional[Exception]) -> str:
        """Suggest specific improvements based on performance."""
        if error:
            return "Fix error handling and provide clearer instructions"
        
        if score >= 0.8:
            return "Consider fine-tuning for edge cases or adding more specific examples"
        elif score >= 0.6:
            return "Add more specific guidance or examples to improve consistency"
        elif score >= 0.4:
            return "Simplify instructions and provide clearer task definition"
        elif score >= 0.2:
            return "Completely revise instructions with step-by-step guidance"
        else:
            return "Restart with basic instructions and clear examples"


class DSPyFeedbackExtractor:
    """Feedback extractor for DSPy components (enhanced version)."""
    
    def extract_feedback(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        score: float,
        trace: Optional[ComponentTrace] = None,
        error: Optional[Exception] = None
    ) -> str:
        """Extract DSPy-specific feedback from component execution."""
        feedback_parts = [f"DSPy Module Score: {score:.3f}"]
        
        # Add signature analysis
        if trace and hasattr(trace, 'metadata'):
            signature_info = trace.metadata.get('signature', '')
            if signature_info:
                feedback_parts.append(f"Signature: {signature_info}")
        
        # Add reasoning analysis if available
        reasoning_fields = ['reasoning', 'rationale', 'explanation', 'thought']
        for field in reasoning_fields:
            if field in outputs:
                reasoning = str(outputs[field])[:200]
                feedback_parts.append(f"Reasoning: {reasoning}")
                break
        
        # Add input/output field analysis
        io_analysis = self._analyze_io_fields(inputs, outputs)
        if io_analysis:
            feedback_parts.append(f"I/O Analysis: {io_analysis}")
        
        # Add error information
        if error:
            feedback_parts.append(f"DSPy Error: {str(error)}")
        
        # Add optimization hints
        optimization_hint = self._get_optimization_hint(score, outputs)
        if optimization_hint:
            feedback_parts.append(f"Optimization Hint: {optimization_hint}")
        
        return " | ".join(feedback_parts)
    
    def _analyze_io_fields(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> str:
        """Analyze DSPy input/output field characteristics."""
        analysis = []
        
        # Input field analysis
        if inputs:
            input_fields = list(inputs.keys())
            analysis.append(f"inputs({', '.join(input_fields)})")
        
        # Output field analysis
        if outputs:
            output_fields = list(outputs.keys())
            analysis.append(f"outputs({', '.join(output_fields)})")
            
            # Check for incomplete outputs
            empty_outputs = [k for k, v in outputs.items() if not v or str(v).strip() == '']
            if empty_outputs:
                analysis.append(f"empty_fields({', '.join(empty_outputs)})")
        
        return ", ".join(analysis)
    
    def _get_optimization_hint(self, score: float, outputs: Dict[str, Any]) -> str:
        """Provide DSPy-specific optimization hints."""
        if score >= 0.8:
            return "Consider adding few-shot examples for edge cases"
        elif score >= 0.6:
            return "Refine instruction clarity and add more context"
        elif score >= 0.4:
            return "Simplify instruction and add clear format requirements"
        elif score >= 0.2:
            return "Use simpler language and provide step-by-step guidance"
        else:
            return "Start with basic instruction template and minimal requirements"




def get_feedback_extractor(component_type: str) -> FeedbackExtractor:
    """Factory function to get appropriate feedback extractor.
    
    Args:
        component_type: Type of component ('crewai', 'openai', 'dspy')
        
    Returns:
        Appropriate feedback extractor instance
    """
    extractors = {
        'crewai': CrewAIFeedbackExtractor(),
        'openai': OpenAIFeedbackExtractor(), 
        'dspy': DSPyFeedbackExtractor(),
        'default': DefaultFeedbackExtractor()
    }
    
    return extractors.get(component_type.lower(), extractors['default'])


# Import DefaultFeedbackExtractor to avoid circular import
from optimas.optim.gepa_adapter import DefaultFeedbackExtractor