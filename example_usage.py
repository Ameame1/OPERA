#!/usr/bin/env python3
"""
OPERA Example Usage
Demonstrates the core functionality of the OPERA system
"""

import json
from opera_cot_rag_baseline import OPERACoTRAGBaseline

def main():
    """Demonstrate OPERA functionality"""
    
    # Initialize OPERA CoT+RAG baseline
    print("Initializing OPERA system...")
    opera = OPERACoTRAGBaseline()
    
    # Example questions
    test_questions = [
        "What is the capital of the country where the inventor of the telephone was born?",
        "Who directed the movie that won the Academy Award for Best Picture in 2010?",
    ]
    
    # Process each question
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        # Get answer with full trajectory
        answer, trajectory = opera.answer_question(question)
        
        # Display results
        print(f"\nFinal Answer: {answer}")
        print(f"\nExecution Summary:")
        print(f"- Total time: {trajectory.total_time:.2f} seconds")
        print(f"- Retrieval rounds: {trajectory.retrieval_rounds}")
        print(f"- Execution steps: {len(trajectory.execution_traces)}")
        
        # Show plan decomposition
        print(f"\nStrategic Plan:")
        for i, sub_q in enumerate(trajectory.strategic_plan['sub_questions'], 1):
            deps = f" (depends on: {sub_q['dependencies']})" if sub_q['dependencies'] else ""
            print(f"  Step {i}: {sub_q['sub_question']}{deps}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()