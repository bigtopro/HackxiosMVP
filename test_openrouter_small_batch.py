import json
import requests
import time

# === CONFIGURATION ===
SYSTEM_PROMPT_PATH = 'systemPromt.JSON'
COMMENTS_PATH = 'sampled_comments.json'
OUTPUT_PATH = 'test_labeled_comments.json'
BATCH_SIZE = 25  # Small test batch

# === OPENROUTER API KEY ===
OPENROUTER_API_KEY = "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"
SITE_URL = "https://your-site.com"
SITE_NAME = "Comment Labeler"

def send_to_openrouter(system_prompt, comments_batch):
    """Send a batch of comments to OpenRouter API for labeling."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    user_content = f"Please process the following comments according to the system instructions:\n\n" + "\n".join(comments_batch)
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": SITE_URL,
        "X-Title": SITE_NAME,
    }
    
    data = {
        "model": "qwen/qwen3-235b-a22b:free",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        "temperature": 0.1,
        "max_tokens": 4000
    }
    
    try:
        print(f"[DEBUG] Sending request to OpenRouter...")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f"[DEBUG] Raw response: {content[:500]}...")
        
        # Try to parse the response as JSON
        try:
            # Look for JSON in the response
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                parsed_response = json.loads(json_str)
                # If it's a direct array, wrap it in a comments object
                if isinstance(parsed_response, list):
                    return {"comments": parsed_response}
                return parsed_response
            else:
                # Try looking for object format
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    parsed_response = json.loads(json_str)
                    return parsed_response
                else:
                    print(f"[DEBUG] No JSON found in response")
                    return None
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON decode error: {e}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] Request error: {e}")
        return None

def main():
    # Load system prompt
    with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
        system_data = json.load(f)
    system_prompt = system_data["system_prompt"]

    # Load comments and take only first 10
    with open(COMMENTS_PATH, 'r', encoding='utf-8') as f:
        all_comments = json.load(f)
    
    test_comments = all_comments[:BATCH_SIZE]
    print(f"Testing with {len(test_comments)} comments...")
    print(f"Sample comments: {test_comments[:3]}...")

    try:
        response = send_to_openrouter(system_prompt, test_comments)
        
        if response and 'comments' in response:
            results = response['comments']
            print(f"✓ Successfully processed {len(results)} comments")
            
            # Save test results
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Test results saved to {OUTPUT_PATH}")
            
            # Show first result as example
            if results:
                print(f"\nExample result:")
                print(json.dumps(results[0], indent=2))
        else:
            print("✗ Failed to get valid response")
            
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    main() 