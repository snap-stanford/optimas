"""Tests for Universal GEPA integration across frameworks."""

import pytest
import random
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from optimas.arch.base import BaseComponent
from optimas.wrappers.example import Example
from optimas.wrappers.prediction import Prediction
from optimas.optim.universal_gepa import UniversalGEPAOptimizer, GEPAOptimizationResult
from optimas.optim.gepa_adapter import OptimasGEPAAdapter, ComponentTrace
from optimas.optim.feedback_extractors import (
    CrewAIFeedbackExtractor, 
    OpenAIFeedbackExtractor,
    get_feedback_extractor
)


class SimpleTestComponent(BaseComponent):
    """Simple test component for GEPA testing."""
    
    def __init__(self, text_variable: str = "Hello, world!"):
        super().__init__(
            description="Simple test component",
            input_fields=["input"],
            output_fields=["output"],
            variable=text_variable
        )
    
    def forward(self, **inputs):
        # Simple echo with variable prepended
        input_text = inputs.get("input", "")
        output_text = f"{self.variable} {input_text}"
        return {"output": output_text}


class MockCrewAIComponent(BaseComponent):
    """Mock CrewAI component for testing."""
    
    def __init__(self):
        self.agent = Mock()
        self.agent.role = "Test Agent"
        self.agent.goal = "Test goal"
        self.agent.backstory = "Test backstory"
        
        super().__init__(
            description="Mock CrewAI component",
            input_fields=["task"],
            output_fields=["result"],
            variable="Test backstory"
        )
    
    def forward(self, **inputs):
        task = inputs.get("task", "")
        result = f"Agent {self.agent.role}: {task}"
        return {"result": result}
    
    @property
    def gepa_optimizable_components(self) -> Dict[str, str]:
        """Return CrewAI-specific optimizable components."""
        components = {}
        if hasattr(self.agent, 'backstory') and self.agent.backstory:
            components['backstory'] = self.agent.backstory
        if hasattr(self.agent, 'goal') and self.agent.goal:
            components['goal'] = self.agent.goal
        if hasattr(self.agent, 'role') and self.agent.role:
            components['role'] = self.agent.role
        return components
    
    def apply_gepa_updates(self, updates: Dict[str, str]) -> None:
        """Apply GEPA updates to CrewAI agent components."""
        if 'backstory' in updates:
            self.agent.backstory = updates['backstory']
            self.update(updates['backstory'])
        if 'goal' in updates:
            self.agent.goal = updates['goal']
        if 'role' in updates:
            self.agent.role = updates['role']
    
    def extract_execution_trace(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract CrewAI-specific execution traces."""
        trace_info = super().extract_execution_trace(inputs, outputs)
        trace_info.update({
            "framework": "crewai",
            "agent_role": getattr(self.agent, 'role', ''),
            "agent_goal": getattr(self.agent, 'goal', ''),
            "agent_backstory": getattr(self.agent, 'backstory', ''),
        })
        return trace_info


class MockOpenAIComponent(BaseComponent):
    """Mock OpenAI component for testing."""
    
    def __init__(self):
        self.agent = Mock()
        self.agent.name = "TestAgent"
        self.agent.instructions = "You are a helpful assistant"
        self.agent.model = "gpt-4o"
        
        super().__init__(
            description="Mock OpenAI component",
            input_fields=["query"],
            output_fields=["response"],
            variable="You are a helpful assistant"
        )
    
    def forward(self, **inputs):
        query = inputs.get("query", "")
        response = f"Assistant: {query} (Instructions: {self.agent.instructions})"
        return {"response": response}
    
    @property
    def gepa_optimizable_components(self) -> Dict[str, str]:
        """Return OpenAI Agent-specific optimizable components."""
        components = {}
        if hasattr(self.agent, 'instructions') and self.agent.instructions:
            components['instructions'] = self.agent.instructions
        return components
    
    def apply_gepa_updates(self, updates: Dict[str, str]) -> None:
        """Apply GEPA updates to OpenAI Agent components."""
        if 'instructions' in updates:
            self.agent.instructions = updates['instructions']
            self.update(updates['instructions'])
    
    def extract_execution_trace(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract OpenAI Agent-specific execution traces."""
        trace_info = super().extract_execution_trace(inputs, outputs)
        trace_info.update({
            "framework": "openai",
            "agent_name": getattr(self.agent, 'name', ''),
            "agent_model": getattr(self.agent, 'model', ''),
            "agent_instructions": getattr(self.agent, 'instructions', ''),
        })
        return trace_info


class TestGEPAInterfaceMethods:
    """Test GEPA interface methods in BaseComponent."""
    
    def test_gepa_optimizable_components_string_variable(self):
        """Test gepa_optimizable_components with string variable."""
        component = SimpleTestComponent("Test prompt")
        components = component.gepa_optimizable_components
        
        assert isinstance(components, dict)
        assert len(components) == 1
        assert "SimpleTestComponent_text" in components
        assert components["SimpleTestComponent_text"] == "Test prompt"
    
    def test_gepa_optimizable_components_dict_variable(self):
        """Test gepa_optimizable_components with dict variable."""
        component = BaseComponent(
            description="Test",
            variable={"prompt": "Hello", "system": "You are helpful"}
        )
        components = component.gepa_optimizable_components
        
        assert isinstance(components, dict)
        assert "prompt" in components
        assert "system" in components
        assert components["prompt"] == "Hello"
        assert components["system"] == "You are helpful"
    
    def test_apply_gepa_updates_string_variable(self):
        """Test apply_gepa_updates with string variable."""
        component = SimpleTestComponent("Original")
        
        updates = {"SimpleTestComponent_text": "Updated text"}
        component.apply_gepa_updates(updates)
        
        assert component.variable == "Updated text"
    
    def test_apply_gepa_updates_dict_variable(self):
        """Test apply_gepa_updates with dict variable."""
        component = BaseComponent(
            description="Test",
            variable={"prompt": "Hello", "system": "You are helpful"}
        )
        
        updates = {"prompt": "Updated prompt"}
        component.apply_gepa_updates(updates)
        
        assert component.variable["prompt"] == "Updated prompt"
        assert component.variable["system"] == "You are helpful"
    
    def test_extract_execution_trace(self):
        """Test extract_execution_trace method."""
        component = SimpleTestComponent("Test")
        
        inputs = {"input": "test input"}
        outputs = {"output": "test output"}
        
        trace = component.extract_execution_trace(inputs, outputs)
        
        assert isinstance(trace, dict)
        assert trace["component_name"] == "SimpleTestComponent"
        assert "inputs_summary" in trace
        assert "outputs_summary" in trace


class TestGEPAAdapter:
    """Test OptimasGEPAAdapter functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.component = SimpleTestComponent("Hello")
        self.metric_fn = Mock(return_value=0.8)
        self.adapter = OptimasGEPAAdapter(
            component=self.component,
            metric_fn=self.metric_fn,
            max_workers=1
        )
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        assert self.adapter.component == self.component
        assert self.adapter.metric_fn == self.metric_fn
        assert self.adapter.max_workers == 1
    
    def test_evaluate_batch(self):
        """Test batch evaluation."""
        examples = [
            Example(input="test1", output="expected1").with_inputs("input"),
            Example(input="test2", output="expected2").with_inputs("input")
        ]
        
        candidate = {"SimpleTestComponent_text": "New prompt"}
        
        result = self.adapter.evaluate(examples, candidate, capture_traces=False)
        
        assert len(result.outputs) == 2
        assert len(result.scores) == 2
        assert result.trajectories is None
        
        # Check that metric was called
        assert self.metric_fn.call_count == 2
    
    def test_evaluate_batch_with_traces(self):
        """Test batch evaluation with trace capture."""
        examples = [Example(input="test", output="expected").with_inputs("input")]
        candidate = {"SimpleTestComponent_text": "New prompt"}
        
        result = self.adapter.evaluate(examples, candidate, capture_traces=True)
        
        assert len(result.outputs) == 1
        assert len(result.scores) == 1
        assert result.trajectories is not None
        assert len(result.trajectories) == 1
        assert isinstance(result.trajectories[0], ComponentTrace)
    
    def test_make_reflective_dataset(self):
        """Test reflective dataset creation."""
        # Create mock evaluation result
        outputs = [{"output": "test output"}]
        scores = [0.7]
        traces = [ComponentTrace(
            inputs={"input": "test"},
            outputs={"output": "test output"},
            component_name="SimpleTestComponent",
            variable_state="Hello",
            execution_time=0.1
        )]
        
        eval_batch = type('EvaluationBatch', (), {
            'outputs': outputs,
            'scores': scores,
            'trajectories': traces
        })()
        
        candidate = {"SimpleTestComponent_text": "Hello"}
        components_to_update = ["SimpleTestComponent_text"]
        
        reflective_data = self.adapter.make_reflective_dataset(
            candidate, eval_batch, components_to_update
        )
        
        assert isinstance(reflective_data, dict)
        assert "SimpleTestComponent_text" in reflective_data
        assert len(reflective_data["SimpleTestComponent_text"]) == 1
        
        example = reflective_data["SimpleTestComponent_text"][0]
        assert "Inputs" in example
        assert "Generated Outputs" in example
        assert "Feedback" in example
        assert "Score" in example


class TestFrameworkSpecificAdapters:
    """Test framework-specific component adaptations."""
    
    def test_crewai_component_gepa_interface(self):
        """Test CrewAI component GEPA interface."""
        component = MockCrewAIComponent()
        
        # Test gepa_optimizable_components
        components = component.gepa_optimizable_components
        assert "backstory" in components
        assert "goal" in components
        assert "role" in components
        
        # Test apply_gepa_updates
        updates = {"backstory": "New backstory", "role": "New role"}
        component.apply_gepa_updates(updates)
        
        assert component.agent.backstory == "New backstory"
        assert component.agent.role == "New role"
        
        # Test extract_execution_trace
        trace = component.extract_execution_trace(
            {"task": "test"}, {"result": "done"}
        )
        assert trace["framework"] == "crewai"
        assert "agent_role" in trace
    
    def test_openai_component_gepa_interface(self):
        """Test OpenAI component GEPA interface."""
        component = MockOpenAIComponent()
        
        # Test gepa_optimizable_components
        components = component.gepa_optimizable_components
        assert "instructions" in components
        
        # Test apply_gepa_updates
        updates = {"instructions": "New instructions"}
        component.apply_gepa_updates(updates)
        
        assert component.agent.instructions == "New instructions"
        
        # Test extract_execution_trace
        trace = component.extract_execution_trace(
            {"query": "test"}, {"response": "answer"}
        )
        assert trace["framework"] == "openai"
        assert "agent_name" in trace


class TestFeedbackExtractors:
    """Test framework-specific feedback extractors."""
    
    def test_crewai_feedback_extractor(self):
        """Test CrewAI feedback extractor."""
        extractor = CrewAIFeedbackExtractor()
        
        inputs = {"task": "Write a summary"}
        outputs = {"output": "This is a summary"}
        score = 0.8
        
        feedback = extractor.extract_feedback(inputs, outputs, score)
        
        assert isinstance(feedback, str)
        assert "Performance Score: 0.800" in feedback
        assert "Task:" in feedback
        assert "Agent Response:" in feedback
        assert "Excellent performance" in feedback
    
    def test_openai_feedback_extractor(self):
        """Test OpenAI feedback extractor."""
        extractor = OpenAIFeedbackExtractor()
        
        inputs = {"query": "What is AI?"}
        outputs = {"response": "AI is artificial intelligence"}
        score = 0.6
        
        feedback = extractor.extract_feedback(inputs, outputs, score)
        
        assert isinstance(feedback, str)
        assert "Performance Score: 0.600" in feedback
        assert "Input Analysis:" in feedback
        assert "Output Analysis:" in feedback
        assert "Improvement Suggestion:" in feedback
    
    def test_feedback_extractor_factory(self):
        """Test feedback extractor factory function."""
        crewai_extractor = get_feedback_extractor("crewai")
        assert isinstance(crewai_extractor, CrewAIFeedbackExtractor)
        
        openai_extractor = get_feedback_extractor("openai")
        assert isinstance(openai_extractor, OpenAIFeedbackExtractor)
        
        default_extractor = get_feedback_extractor("unknown")
        assert default_extractor.__class__.__name__ == "DefaultFeedbackExtractor"


class TestUniversalGEPAOptimizer:
    """Test UniversalGEPAOptimizer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reflection_lm = Mock(return_value="Improved instruction text")
        self.optimizer = UniversalGEPAOptimizer(
            reflection_lm=self.reflection_lm,
            max_metric_calls=10,
            seed=42
        )
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.reflection_lm == self.reflection_lm
        assert self.optimizer.max_metric_calls == 10
        assert self.optimizer.seed == 42
    
    def test_budget_validation(self):
        """Test budget parameter validation."""
        # Should raise error with no budget
        with pytest.raises(ValueError, match="Exactly one budget parameter"):
            UniversalGEPAOptimizer(reflection_lm=self.reflection_lm)
        
        # Should raise error with multiple budgets
        with pytest.raises(ValueError, match="Exactly one budget parameter"):
            UniversalGEPAOptimizer(
                reflection_lm=self.reflection_lm,
                max_metric_calls=10,
                num_iters=5
            )
    
    def test_detect_framework_type(self):
        """Test framework type detection."""
        # Test generic component
        component = SimpleTestComponent()
        framework_type = self.optimizer._detect_framework_type(component)
        assert framework_type == "generic"
        
        # Test CrewAI component detection based on class name
        class CrewAITestComponent(SimpleTestComponent):
            pass
        crewai_component = CrewAITestComponent()
        crewai_component.agent = Mock()
        crewai_component.agent.role = "test"
        framework_type = self.optimizer._detect_framework_type(crewai_component)
        assert framework_type == "crewai"
        
        # Test OpenAI component detection based on class name  
        class OpenAITestComponent(SimpleTestComponent):
            pass
        openai_component = OpenAITestComponent()
        openai_component.agent = Mock()
        openai_component.agent.instructions = "test"
        framework_type = self.optimizer._detect_framework_type(openai_component)
        assert framework_type == "openai"
    
    def test_create_default_metric(self):
        """Test default metric creation."""
        component = SimpleTestComponent()
        component.output_fields = ["output"]
        metric = self.optimizer._create_default_metric(component)
        
        # Test metric with matching outputs
        gold = Example(input="test", output="expected").with_inputs("input")
        pred = Mock()
        pred.output = "expected"
        
        score = metric(gold, pred)
        assert score == 1.0
        
        # Test metric with non-matching outputs
        pred.output = "different"
        score = metric(gold, pred)
        assert score == 0.0
    
    @patch('gepa.optimize')
    def test_optimize_with_universal_adapter(self, mock_gepa_optimize):
        """Test optimization with universal adapter."""
        # Mock GEPA result
        mock_result = Mock()
        mock_result.best_candidate = {"SimpleTestComponent_text": "Optimized text"}
        mock_result.val_aggregate_scores = [0.9]
        mock_result.total_metric_calls = 5
        mock_gepa_optimize.return_value = mock_result
        
        component = SimpleTestComponent("Original text")
        trainset = [Example(input="test", output="expected").with_inputs("input")]
        
        result = self.optimizer.optimize_component(
            component=component,
            trainset=trainset
        )
        
        assert isinstance(result, GEPAOptimizationResult)
        assert result.framework_type == "generic"
        assert result.final_score == 0.9
        assert result.total_evaluations == 5
        assert "SimpleTestComponent_text" in result.optimized_components
        
        # Check that component was updated
        assert component.variable == "Optimized text"


class TestIntegrationTests:
    """Integration tests for the full GEPA system."""
    
    @patch('gepa.optimize')
    def test_end_to_end_optimization_generic_component(self, mock_gepa_optimize):
        """Test end-to-end optimization for generic component."""
        # Mock GEPA result
        mock_result = Mock()
        mock_result.best_candidate = {"SimpleTestComponent_text": "Optimized prompt"}
        mock_result.val_aggregate_scores = [0.85]
        mock_result.total_metric_calls = 8
        mock_gepa_optimize.return_value = mock_result
        
        # Create component and optimizer
        component = SimpleTestComponent("Original prompt")
        reflection_lm = Mock(return_value="Better instruction")
        optimizer = UniversalGEPAOptimizer(
            reflection_lm=reflection_lm,
            max_metric_calls=20
        )
        
        # Create training data
        trainset = [
            Example(input="hello", output="hello world").with_inputs("input"),
            Example(input="test", output="test case").with_inputs("input")
        ]
        
        # Define metric
        def simple_metric(gold, pred, trace=None):
            return 0.7 if "world" in pred.output else 0.3
        
        # Run optimization
        result = optimizer.optimize_component(
            component=component,
            trainset=trainset,
            metric_fn=simple_metric
        )
        
        # Verify results
        assert result.framework_type == "generic"
        assert result.final_score == 0.85
        assert component.variable == "Optimized prompt"
    
    def test_error_handling_no_optimizable_components(self):
        """Test handling of components with no optimizable parts."""
        # Component with no variable
        component = BaseComponent(description="Test", variable=None)
        
        reflection_lm = Mock(return_value="Feedback")
        optimizer = UniversalGEPAOptimizer(
            reflection_lm=reflection_lm,
            max_metric_calls=10
        )
        
        trainset = [Example(input="test", output="expected").with_inputs("input")]
        
        result = optimizer.optimize_component(component, trainset)
        
        assert result.best_candidate == {}
        assert result.optimized_components == []
        assert result.total_evaluations == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])