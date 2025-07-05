import json
import requests
import time

# === CONFIGURATION ===
SYSTEM_PROMPT_PATH = 'systemPromt.JSON'  # Path to your system prompt JSON
COMMENTS_PATH = 'non_english_comments.json'  # Path to your comments JSON (array of strings)
OUTPUT_PATH = 'labeled_comments.json'  # Output file for consolidated results
BATCH_SIZE = 50

# === FILL IN YOUR OPENROUTER API KEY HERE ===
OPENROUTER_API_KEY = "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"  # Replace with your actual API key
SITE_URL = "https://your-site.com"  # Optional: Your site URL
SITE_NAME = "Comment Labeler"  # Optional: Your site name

# === MAIN SCRIPT ===
def send_to_openrouter(system_prompt, comments_batch):
    """Send a batch of comments to OpenRouter API for labeling."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # Combine system prompt and comments
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
        "temperature": 0.1,  # Low temperature for consistent labeling
        "max_tokens": 4000  # Adjust based on your needs
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Try to parse the response as JSON
        try:
            # Look for JSON array first (Qwen format)
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
                    print(f"[DEBUG] No JSON found in response: {content[:200]}...")
                    return None
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON decode error: {e}")
            print(f"[DEBUG] Response content: {content[:500]}...")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] Request error: {e}")
        return None

def main():
    # Load system prompt
    with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
        system_data = json.load(f)
    system_prompt = system_data["system_prompt"]

    # Load comments
    with open(COMMENTS_PATH, 'r', encoding='utf-8') as f:
        comments = json.load(f)

    all_results = []
    total = len(comments)
    print(f"Processing {total} comments in batches of {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch = comments[i:i+BATCH_SIZE]
        batch_num = i//BATCH_SIZE + 1
        print(f"Processing batch {batch_num} ({i+1}-{min(i+BATCH_SIZE, total)})...")

        try:
            response = send_to_openrouter(system_prompt, batch)
            
            if response and 'comments' in response:
                batch_results = response['comments']
                all_results.extend(batch_results)
                print(f"  ✓ Successfully processed {len(batch_results)} comments")
            else:
                print(f"  ✗ Failed to get valid response for batch {batch_num}")
                
        except Exception as e:
            print(f"  ✗ Error processing batch {batch_num}: {e}")
            continue
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)

    # Write all results to output file
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Done! Labeled results written to {OUTPUT_PATH}")
    print(f"Total comments processed: {len(all_results)}")

if __name__ == "__main__":
    main() 