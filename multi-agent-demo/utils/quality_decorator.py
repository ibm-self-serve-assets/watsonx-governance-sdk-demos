"""
Custom decorator for evaluating quality metrics on agent outputs.

This decorator wraps node functions and evaluates their outputs using
the MetricsEvaluator.evaluate() approach that's proven to work.
"""

from functools import wraps
from typing import Callable, List, Any, Dict
from ibm_watsonx_gov.config import GenAIConfiguration
from ibm_watsonx_gov.metrics import FaithfulnessMetric, ContextRelevanceMetric
from ibm_watsonx_gov.evaluators import MetricsEvaluator


def evaluate_output_quality(
    project_id: str,
    llm_judge,
    input_fields: List[str],
    context_fields: List[str],
    output_field: str,
    metric_type: str = "faithfulness"  # or "context_relevance"
):
    """
    Custom decorator to evaluate quality metrics on node outputs.
    
    Args:
        project_id: Watsonx project ID
        llm_judge: LLMJudge instance
        input_fields: List of input field names from state
        context_fields: List of context field names from state
        output_field: Dot-notation path to output field (e.g., "action_recommendation.final_output")
        metric_type: Type of metric to compute ("faithfulness" or "context_relevance")
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: Dict[str, Any], config=None) -> Dict[str, Any]:
            # Call the original function
            result = func(state, config)
            
            # Extract the output text using dot notation
            output_parts = output_field.split('.')
            output_value = result
            for part in output_parts:
                if isinstance(output_value, dict):
                    output_value = output_value.get(part, "")
                else:
                    output_value = ""
                    break
            
            # Skip evaluation if output is empty or error
            if not output_value or isinstance(output_value, dict) and "error" in output_value:
                return result
            
            try:
                # Prepare evaluation data
                eval_data = {
                    "output": str(output_value)
                }
                
                # Add input fields
                for field in input_fields:
                    if field in state:
                        eval_data[field] = state[field]
                
                # Add context fields
                for field in context_fields:
                    if field in state:
                        context_value = state[field]
                        # Convert dict to string for evaluation
                        if isinstance(context_value, dict):
                            context_value = str(context_value)
                        eval_data[field] = context_value
                
                # Create configuration
                config_dict = {
                    "input_fields": input_fields,
                    "output_fields": ["output"]
                }
                if context_fields:
                    config_dict["context_fields"] = context_fields
                
                gen_config = GenAIConfiguration(**config_dict)
                
                # Create MetricsEvaluator (not AgenticEvaluator)
                metric_evaluator = MetricsEvaluator(
                    project_id=project_id,
                    llm_judge=llm_judge
                )
                
                # Create metric
                if metric_type == "faithfulness":
                    metric = FaithfulnessMetric(configuration=gen_config)
                else:  # context_relevance
                    metric = ContextRelevanceMetric(configuration=gen_config)
                
                # Evaluate using MetricEvaluator
                eval_result = metric_evaluator.evaluate(
                    data=eval_data,
                    metrics=[metric]
                )
                
                # Store metric result in state for later retrieval
                metric_key = f"{func.__name__}_{metric_type}_score"
                if eval_result and hasattr(eval_result, 'to_df'):
                    df = eval_result.to_df()
                    if not df.empty and len(df.columns) > 0:
                        score = df.iloc[0, 0]  # Get first value
                        result[metric_key] = float(score)
                        print(f"[QUALITY METRIC] {func.__name__} {metric_type}: {score:.4f}")
                
            except Exception as e:
                print(f"[WARNING] Quality evaluation failed for {func.__name__}: {str(e)}")
            
            return result
        
        return wrapper
    return decorator
