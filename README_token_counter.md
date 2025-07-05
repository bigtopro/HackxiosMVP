# Token Counter for LLM Models

A comprehensive Python utility to count tokens in files for various LLM models, with cost estimation and support for multiple model types.

## Features

- **Multiple Model Support**: GPT-4, Claude, Gemini, Grok, Llama, Mistral, and more
- **Accurate Token Counting**: Uses official tokenizers (tiktoken, transformers)
- **Cost Estimation**: Approximate cost calculation for different models
- **File Type Support**: Works with text files, JSON, Python, and more
- **Command Line Interface**: Easy to use from terminal
- **Programmatic API**: Can be imported and used in your code

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_token_counter.txt
```

Or install manually:
```bash
pip install tiktoken transformers torch
```

## Usage

### Command Line Interface

#### Basic Usage
```bash
python token_counter.py your_file.txt
```

#### Specify a Model
```bash
python token_counter.py your_file.json --model gpt-4
python token_counter.py your_file.txt --model claude-3-sonnet
```

#### Verbose Output
```bash
python token_counter.py your_file.txt --verbose
```

#### List Supported Models
```bash
python token_counter.py --list-models
```

### Programmatic Usage

```python
from token_counter import TokenCounter

# Initialize counter for a specific model
counter = TokenCounter("gpt-4")

# Count tokens in a file
result = counter.count_tokens_in_file("your_file.txt")
print(f"Tokens: {result['token_count']:,}")
print(f"Cost: ${result['estimated_cost']['total_cost_usd']:.4f}")

# Count tokens in text directly
text = "Hello, world!"
token_count = counter.count_tokens(text)
print(f"Tokens: {token_count}")
```

## Supported Models

### OpenAI Models
- gpt-4, gpt-4-turbo, gpt-4o
- gpt-3.5-turbo, gpt-3.5-turbo-16k

### Anthropic Models
- claude-3, claude-3-opus, claude-3-sonnet, claude-3-haiku
- claude-2, claude-instant

### Google Models
- gemini-pro, gemini-pro-vision

### Grok Models
- grok-beta, grok-2

### Open Source Models
- llama-2, llama-3, llama-3.1
- mistral, mixtral

## Example Output

```
File: test_labeled_comments.json
Model: gpt-4
Tokens: 1,234
Characters: 4,567
Words: 890
Lines: 45
Estimated cost: $0.0370 USD
  - Input: $0.0370
  - Output: $0.0740
```

## Cost Estimation

The tool provides approximate cost estimates based on current pricing (as of 2024):

- **GPT-4**: $0.03/1K input, $0.06/1K output
- **GPT-4 Turbo**: $0.01/1K input, $0.03/1K output
- **Claude-3 Sonnet**: $0.003/1K input, $0.015/1K output
- **Gemini Pro**: $0.0005/1K input, $0.0015/1K output

## File Types Supported

- **Text files** (.txt, .md, .py, .js, .html, etc.)
- **JSON files** (.json) - properly formatted
- **Code files** (.py, .js, .java, .cpp, etc.)
- **Any UTF-8 encoded text file**

## Examples

### Count tokens in your existing files:
```bash
# Count tokens in your JSON file
python token_counter.py test_labeled_comments.json

# Count tokens in your Python script
python token_counter.py batch_label_comments_with_openrouter.py

# Compare costs across models
python token_counter.py sampled_comments.json --model gpt-4
python token_counter.py sampled_comments.json --model claude-3-sonnet
```

### Run the example script:
```bash
python example_usage.py
```

## Notes

- Token counting accuracy depends on the model's actual tokenizer
- Cost estimates are approximate and may vary
- For large files, processing may take a moment
- The tool automatically handles JSON files by parsing and re-serializing them

## Troubleshooting

1. **"tiktoken not available"**: Install with `pip install tiktoken`
2. **"transformers not available"**: Install with `pip install transformers torch`
3. **File encoding issues**: Ensure files are UTF-8 encoded
4. **Large files**: The tool can handle large files but may take time to process

## License

This tool is provided as-is for educational and development purposes. 