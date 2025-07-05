#!/usr/bin/env python3
"""
Token Counter for LLM Models
A utility to count tokens in files for various LLM models.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Install with: pip install tiktoken")

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")

# Common model encodings
MODEL_ENCODINGS = {
    # OpenAI models
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base", 
    "gpt-4o": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    
    # Anthropic models
    "claude-3": "cl100k_base",
    "claude-3-opus": "cl100k_base",
    "claude-3-sonnet": "cl100k_base",
    "claude-3-haiku": "cl100k_base",
    "claude-2": "cl100k_base",
    "claude-instant": "cl100k_base",
    
    # Google models
    "gemini-pro": "cl100k_base",
    "gemini-pro-vision": "cl100k_base",
    
    # Grok models
    "grok-beta": "cl100k_base",
    "grok-2": "cl100k_base",
    "grok-3-beta": "cl100k_base",
    "grok-3-mini-beta": "cl100k_base",
    "grok-3-mini": "cl100k_base",
    
    # Llama models
    "llama-2": "llama",
    "llama-3": "llama",
    "llama-3.1": "llama",
    
    # Mistral models
    "mistral": "cl100k_base",
    "mixtral": "cl100k_base",
    
    # Default fallback
    "default": "cl100k_base"
}

class TokenCounter:
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.encoding_name = MODEL_ENCODINGS.get(model_name.lower(), MODEL_ENCODINGS["default"])
        self.encoding = None
        self.tokenizer = None
        
        # Initialize tiktoken
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding(self.encoding_name)
            except KeyError:
                print(f"Warning: Encoding '{self.encoding_name}' not found, using cl100k_base")
                self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Initialize transformers tokenizer for specific models
        if TRANSFORMERS_AVAILABLE and model_name.lower().startswith(("llama", "mistral")):
            try:
                model_id = self._get_model_id(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            except Exception as e:
                print(f"Warning: Could not load transformers tokenizer: {e}")
    
    def _get_model_id(self, model_name: str) -> str:
        """Get the HuggingFace model ID for a given model name."""
        model_mapping = {
            "llama-2": "meta-llama/Llama-2-7b-chat-hf",
            "llama-3": "meta-llama/Llama-3-8b-chat-hf",
            "llama-3.1": "meta-llama/Llama-3.1-8b-chat-hf",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
            "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1"
        }
        return model_mapping.get(model_name.lower(), "meta-llama/Llama-2-7b-chat-hf")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the appropriate method."""
        if self.tokenizer:
            # Use transformers tokenizer for specific models
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        elif self.encoding:
            # Use tiktoken for OpenAI-style models
            tokens = self.encoding.encode(text)
            return len(tokens)
        else:
            # Fallback: rough estimation (1 token ≈ 4 characters for English)
            return len(text) // 4
    
    def count_tokens_in_file(self, file_path: str) -> Dict[str, Any]:
        """Count tokens in a file and return detailed information."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        try:
            # Read file content
            if file_path.suffix.lower() in ['.json']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    text = json.dumps(content, ensure_ascii=False, indent=2)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            # Count tokens
            token_count = self.count_tokens(text)
            char_count = len(text)
            word_count = len(text.split())
            line_count = len(text.splitlines())
            
            # Calculate costs (approximate)
            cost_info = self._estimate_cost(token_count)
            
            return {
                "file_path": str(file_path),
                "file_size_bytes": file_path.stat().st_size,
                "model": self.model_name,
                "token_count": token_count,
                "character_count": char_count,
                "word_count": word_count,
                "line_count": line_count,
                "estimated_cost": cost_info,
                "file_type": file_path.suffix.lower(),
                "encoding_used": self.encoding_name if self.encoding else "transformers"
            }
            
        except Exception as e:
            return {"error": f"Error processing file: {str(e)}"}
    
    def _estimate_cost(self, token_count: int) -> Dict[str, float]:
        """Estimate cost for different models (approximate)."""
        # Approximate costs per 1K tokens (as of 2024)
        costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "gemini-pro": {"input": 0.0005, "output": 0.0015},
            "grok-beta": {"input": 0.0001, "output": 0.0001},
            "grok-3-beta": {"input": 0.003, "output": 0.015},
            "grok-3-mini-beta": {"input": 0.0003, "output": 0.0005},
            "grok-3-mini": {"input": 0.0003, "output": 0.0005}
        }
        
        model_key = self.model_name.lower()
        if model_key in costs:
            input_cost = (token_count / 1000) * costs[model_key]["input"]
            output_cost = (token_count / 1000) * costs[model_key]["output"]
            return {
                "input_cost_usd": round(input_cost, 4),
                "output_cost_usd": round(output_cost, 4),
                "total_cost_usd": round(input_cost + output_cost, 4)
            }
        else:
            return {"note": "Cost estimation not available for this model"}

def main():
    parser = argparse.ArgumentParser(description="Count tokens in files for LLM models")
    parser.add_argument("file_path", nargs="?", help="Path to the file to analyze")
    parser.add_argument("--model", "-m", default="gpt-4", 
                       help="Model name (default: gpt-4)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--list-models", action="store_true",
                       help="List supported models")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Supported models:")
        for model in sorted(MODEL_ENCODINGS.keys()):
            if model != "default":
                print(f"  - {model}")
        return
    
    if not args.file_path:
        parser.error("file_path is required unless --list-models is specified")
    
    # Initialize token counter
    counter = TokenCounter(args.model)
    
    # Count tokens
    result = counter.count_tokens_in_file(args.file_path)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    # Display results
    if args.verbose:
        print(json.dumps(result, indent=2))
    else:
        print(f"File: {result['file_path']}")
        print(f"Model: {result['model']}")
        print(f"Tokens: {result['token_count']:,}")
        print(f"Characters: {result['character_count']:,}")
        print(f"Words: {result['word_count']:,}")
        print(f"Lines: {result['line_count']:,}")
        
        if "estimated_cost" in result and "total_cost_usd" in result["estimated_cost"]:
            cost = result["estimated_cost"]
            print(f"Estimated cost: ${cost['total_cost_usd']:.4f} USD")
            print(f"  - Input: ${cost['input_cost_usd']:.4f}")
            print(f"  - Output: ${cost['output_cost_usd']:.4f}")

if __name__ == "__main__":
    main() 