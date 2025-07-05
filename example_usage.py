#!/usr/bin/env python3
"""
Example usage of the token counter
"""

from token_counter import TokenCounter
import json

def main():
    # Example 1: Count tokens in a JSON file
    print("=== Example 1: Counting tokens in test_labeled_comments.json ===")
    counter = TokenCounter("gpt-4")
    result = counter.count_tokens_in_file("test_labeled_comments.json")
    
    if "error" not in result:
        print(f"File: {result['file_path']}")
        print(f"Tokens: {result['token_count']:,}")
        print(f"Characters: {result['character_count']:,}")
        print(f"Estimated cost: ${result['estimated_cost']['total_cost_usd']:.4f}")
        print()
    
    # Example 2: Count tokens in a Python file
    print("=== Example 2: Counting tokens in a Python file ===")
    result = counter.count_tokens_in_file("token_counter.py")
    
    if "error" not in result:
        print(f"File: {result['file_path']}")
        print(f"Tokens: {result['token_count']:,}")
        print(f"Characters: {result['character_count']:,}")
        print(f"Estimated cost: ${result['estimated_cost']['total_cost_usd']:.4f}")
        print()
    
    # Example 3: Compare different models
    print("=== Example 3: Comparing different models ===")
    models = ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "gemini-pro"]
    
    for model in models:
        counter = TokenCounter(model)
        result = counter.count_tokens_in_file("test_labeled_comments.json")
        
        if "error" not in result:
            print(f"{model:20} | {result['token_count']:6,} tokens | ${result['estimated_cost']['total_cost_usd']:.4f}")
    
    # Example 4: Count tokens in text directly
    print("\n=== Example 4: Counting tokens in text directly ===")
    sample_text = "Hello, this is a sample text to count tokens for different LLM models."
    
    for model in ["gpt-4", "claude-3-sonnet"]:
        counter = TokenCounter(model)
        token_count = counter.count_tokens(sample_text)
        print(f"{model:20} | {token_count:6} tokens | Text: '{sample_text}'")

if __name__ == "__main__":
    main() 