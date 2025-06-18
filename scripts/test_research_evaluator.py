import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from deep_research_agent.evaluations.evaluators.research_evaluator import ResearchEvaluator

def main():
    # Initialize the evaluator
    evaluator = ResearchEvaluator()
    
    # Test case 1: Research report with ground truth
    research_output = """
    # Research Report: Latest Developments in Quantum Computing
    
    ## Executive Summary
    
    Quantum computing has made significant progress in recent years, with IBM achieving 433 qubits in their Osprey processor [IBM. (2022). IBM Quantum Roadmap.]. This represents a major milestone in the field's development. The technology shows promise for applications in cryptography and drug discovery [National Quantum Initiative. (2023). Quantum Applications Report.].
    
    ## Key Findings
    
    1. Hardware Development
       - IBM's 433-qubit processor demonstrates scalability
       - Error correction techniques are improving
       - New quantum algorithms are being developed
    
    2. Applications
       - Cryptography: Post-quantum encryption standards
       - Drug Discovery: Molecular simulation capabilities
       - Optimization: Solving complex logistics problems
    """
    
    ground_truth = """
    Key points that must be included:
    1. IBM's 433-qubit processor (2022)
    2. Applications in cryptography and drug discovery
    3. Error correction progress
    4. National Quantum Initiative involvement
    """
    
    # Evaluate the research output
    print("\nEvaluating research output with ground truth...")
    metrics = evaluator.evaluate(
        research_output=research_output,
        ground_truth=ground_truth,
        context={"query": "What are the latest developments in quantum computing?"}
    )
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print("=" * 80)
    print(f"Accuracy: {metrics.accuracy:.2f}")
    print(f"Relevance: {metrics.relevance:.2f}")
    print(f"Coherence: {metrics.coherence:.2f}")
    print("\nAdditional Metrics:")
    for key, value in metrics.additional_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Test case 2: Research report without ground truth
    research_output_2 = """
    # Analysis of Edge AI Deployment Challenges
    
    ## Executive Summary
    
    Deploying AI models on edge devices presents unique challenges and opportunities. The primary considerations include model size optimization, power consumption, and real-time processing requirements [Qualcomm. (2023). Edge AI Deployment Guide.]. Recent advances in model compression techniques have enabled more efficient deployment [Google. (2023). Efficient Edge AI Models.].
    
    ## Key Challenges
    
    1. Hardware Constraints
       - Limited computational resources
       - Power consumption limitations
       - Memory constraints
    
    2. Model Optimization
       - Quantization techniques
       - Pruning methods
       - Knowledge distillation
    """
    
    print("\nEvaluating research output without ground truth...")
    metrics_2 = evaluator.evaluate(
        research_output=research_output_2,
        context={"query": "What are the challenges of deploying AI on edge devices?"}
    )
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print("=" * 80)
    print(f"Accuracy: {metrics_2.accuracy:.2f}")
    print(f"Relevance: {metrics_2.relevance:.2f}")
    print(f"Coherence: {metrics_2.coherence:.2f}")
    print("\nAdditional Metrics:")
    for key, value in metrics_2.additional_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main() 