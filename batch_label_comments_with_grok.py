import json
import os
from grok_client import GrokClient

# === CONFIGURATION ===
SYSTEM_PROMPT_PATH = 'systemPromt.JSON'  # Path to your system prompt JSON
COMMENTS_PATH = 'non_english_comments.json'  # Path to your comments JSON (array of strings)
OUTPUT_PATH = 'labeled_comments.json'  # Output file for consolidated results
BATCH_SIZE = 50

# === FILL IN YOUR GROK COOKIES HERE ===
cookies = {
    "x-anonuserid": "4fdd7f55-e5a4-4856-8295-29a22cff3b30",
    "x-challenge": "E%2BhXiKcF2w5k5XgxBHSumwqaPCQRiinVKFtAt8ifiwRX3wI%2FtQOkix5C87U0DpSR%2FFFxub41zTkeu3V7jIDtcohtM3d9XjogsQpaUKXwN9KLF%2F8rjFg6FG8KF2MDMyvZZEskkHYZWSr5fb87iHiRaQAGtkoqyJdr3qtreU4wTvpaVOqr260%3D",
    "x-signature": "2f8ecyMGwX4p6UN3zinSdUOV2tz5F%2BSmijVxgH%2FOQegqPQVL1bvqTAXJZvutkDX%2FxqzzXZ2QGO0GImkbtCaLHg%3D%3D",
    "sso": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzZXNzaW9uX2lkIjoiMTFhZDE2ZjAtY2U3ZS00NjY4LTk3ZTItZjQ5MjEzYTNjNWI1In0.HMuCVdTEsSea_m9ay7Eb0Uo5nR43SoJHl3II9fa4yjc",
    "sso-rw": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzZXNzaW9uX2lkIjoiMTFhZDE2ZjAtY2U3ZS00NjY4LTk3ZTItZjQ5MjEzYTNjNWI1In0.HMuCVdTEsSea_m9ay7Eb0Uo5nR43SoJHl3II9fa4yjc"
}

# === MAIN SCRIPT ===
def main():
    # Load system prompt
    with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
        system_data = json.load(f)
    system_prompt = system_data["system_prompt"]

    # Load comments
    with open(COMMENTS_PATH, 'r', encoding='utf-8') as f:
        comments = json.load(f)

    # Prepare Grok client
    client = GrokClient(cookies)

    all_results = []
    total = len(comments)
    print(f"Processing {total} comments in batches of {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch = comments[i:i+BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1} ({i+1}-{min(i+BATCH_SIZE, total)})...")

        # Combine system prompt and comments into a single string
        prompt = f"{system_prompt}\n\nComments to process (one per line):\n" + "\n".join(batch)

        try:
            response = client.send_message(prompt)
            print(f"[DEBUG] Raw response type: {type(response)}")
            print(f"[DEBUG] Raw response (first 500 chars): {str(response)[:500]}")
            if isinstance(response, str):
                try:
                    response_json = json.loads(response)
                except Exception as je:
                    print(f"[DEBUG] JSON decode error: {je}")
                    response_json = None
            else:
                response_json = response
            batch_results = response_json.get('comments', []) if response_json else []
            all_results.extend(batch_results)
        except Exception as e:
            print(f"Error processing batch {i//BATCH_SIZE + 1}: {e}")
            continue

    # Write all results to output file
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Done! Labeled results written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 