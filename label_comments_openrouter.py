import json
import requests
import os
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import threading
from queue import Queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('comment_processing.log')  # Output to file
    ]
)
logger = logging.getLogger(__name__)

# OpenRouter API configuration
API_KEYS = [
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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"
]

# Available models in order of preference
MODELS = [
    "qwen/qwen3-235b-a22b:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1-0528:free",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat:free"
]

API_URL = "https://openrouter.ai/api/v1/chat/completions"
BATCH_SIZE = 50
RATE_LIMIT_DELAY = 30  # Reduced delay between batches
CHECKPOINT_FILE = "resume_checkpoint.txt"
FAILED_KEYS_FILE = "failed_keys.json"
FAILED_MODELS_FILE = "failed_models.json"

class APIKeyManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.failed_keys = self._load_failed_keys()
        self.key_usage = {key: {"attempts": 0, "successes": 0, "failures": 0} for key in api_keys}
        
        # Remove any previously failed keys from the rotation
        self.api_keys = [key for key in self.api_keys if key not in self.failed_keys]
        if not self.api_keys:
            raise ValueError("No valid API keys available!")
        
        logger.info(f"Initialized with {len(self.api_keys)} API keys")
    
    def _load_failed_keys(self) -> List[str]:
        """Load the list of failed keys from disk."""
        try:
            with open(FAILED_KEYS_FILE, 'r') as f:
                failed_keys = json.load(f)
                if failed_keys:
                    logger.warning(f"Loaded {len(failed_keys)} previously failed keys")
                return failed_keys
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_failed_keys(self):
        """Save the current list of failed keys to disk."""
        with open(FAILED_KEYS_FILE, 'w') as f:
            json.dump(self.failed_keys, f)
    
    def get_current_key(self) -> str:
        """Get the current API key."""
        return self.api_keys[self.current_key_index]
    
    def mark_key_failed(self, key: str):
        """Mark a key as failed and rotate to the next one."""
        if key not in self.failed_keys:
            logger.error(f"API key failed (last 10 chars: ...{key[-10:]})")
            self.failed_keys.append(key)
            self._save_failed_keys()
            self.key_usage[key]["failures"] += 1
        
        # Remove the key from the rotation
        if key in self.api_keys:
            self.api_keys.remove(key)
        
        if not self.api_keys:
            logger.critical("All API keys have failed!")
            raise ValueError("All API keys have failed!")
        
        # Log key usage statistics
        logger.info(f"Key usage stats for failed key: {self.key_usage[key]}")
        
        # Adjust the current index if needed
        self.current_key_index = self.current_key_index % len(self.api_keys)
    
    def rotate_key(self):
        """Rotate to the next available API key."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        current_key = self.get_current_key()
        logger.info(f"Rotating to next API key (last 10 chars: ...{current_key[-10:]})")
        logger.info(f"Current key stats: {self.key_usage[current_key]}")

class ModelManager:
    def __init__(self, models: List[str]):
        self.models = models
        self.current_model_index = 0
        self.failed_models = self._load_failed_models()
        self.model_usage = {model: {"attempts": 0, "successes": 0, "failures": 0} for model in models}
        
        # Remove any previously failed models from the rotation
        self.models = [model for model in self.models if model not in self.failed_models]
        if not self.models:
            raise ValueError("No valid models available!")
        
        logger.info(f"Initialized with {len(self.models)} models")
        logger.info(f"Available models: {', '.join(self.models)}")
    
    def _load_failed_models(self) -> List[str]:
        """Load the list of failed models from disk."""
        try:
            with open(FAILED_MODELS_FILE, 'r') as f:
                failed_models = json.load(f)
                if failed_models:
                    logger.warning(f"Loaded {len(failed_models)} previously failed models")
                return failed_models
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_failed_models(self):
        """Save the current list of failed models to disk."""
        with open(FAILED_MODELS_FILE, 'w') as f:
            json.dump(self.failed_models, f)
    
    def get_current_model(self) -> str:
        """Get the current model."""
        return self.models[self.current_model_index]
    
    def mark_model_failed(self, model: str):
        """Mark a model as failed and rotate to the next one."""
        if model not in self.failed_models:
            logger.error(f"Model failed: {model}")
            self.failed_models.append(model)
            self._save_failed_models()
            self.model_usage[model]["failures"] += 1
        
        # Remove the model from the rotation
        if model in self.models:
            self.models.remove(model)
        
        if not self.models:
            logger.critical("All models have failed!")
            raise ValueError("All models have failed!")
        
        # Log model usage statistics
        logger.info(f"Model usage stats for failed model: {self.model_usage[model]}")
        
        # Adjust the current index if needed
        self.current_model_index = self.current_model_index % len(self.models)
    
    def rotate_model(self):
        """Rotate to the next available model."""
        self.current_model_index = (self.current_model_index + 1) % len(self.models)
        current_model = self.get_current_model()
        logger.info(f"Rotating to next model: {current_model}")
        logger.info(f"Current model stats: {self.model_usage[current_model]}")

def read_all_comments(file_path: str) -> List[str]:
    """Read all comments from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        comments = json.load(f)
    return comments

def get_last_processed_index() -> int:
    """Get the last processed index from checkpoint file."""
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0

def save_checkpoint(index: int):
    """Save the current index to checkpoint file."""
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(index))

def create_prompt(comments: List[str]) -> List[Dict]:
    """Create the messages array for the API request."""
    # Read system prompt from file
    with open('systemPromt.JSON', 'r', encoding='utf-8') as f:
        system_prompt_data = json.load(f)
        system_prompt = system_prompt_data['system_prompt']

    # Format comments with indices
    indexed_comments = [f"{i}: {comment}" for i, comment in enumerate(comments)]
    formatted_comments = "\n".join(indexed_comments)
    
    # Create the messages array
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please label these {len(comments)} YouTube comments according to the guidelines. Return ONLY the JSON object with no additional text or formatting:\n\n{formatted_comments}"}
    ]
    
    return messages

def process_batch_threadsafe(comments, base_index, key, model, thread_id):
    """Process a batch of comments using a single API key and model."""
    messages = create_prompt(comments)
    max_key_retries = 3
    for key_retry_count in range(max_key_retries):
        try:
            logger.info(f"Thread-{thread_id}: Using API key ...{key[-10:]} with model {model} for batch {base_index}-{base_index+len(comments)-1}")
            response = requests.post(
                url=API_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/yourusername/LUCY-1",
                    "X-Title": "LUCY-1 Comment Analyzer"
                },
                data=json.dumps({
                    "model": model,
                    "messages": messages
                }, ensure_ascii=False).encode('utf-8'),
                timeout=480
            )
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                content = content.replace("```json", "").replace("```", "").strip()
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "comments" in parsed and parsed["comments"]:
                        enriched_comments = []
                        for comment_result in parsed["comments"]:
                            if not isinstance(comment_result, dict) or "index" not in comment_result or "label" not in comment_result:
                                continue
                            original_index = base_index + comment_result["index"]
                            if 0 <= comment_result["index"] < len(comments):
                                enriched_comments.append({
                                    "index": original_index,
                                    "label": comment_result["label"],
                                    "text": comments[comment_result["index"]]
                                })
                        if enriched_comments:
                            logger.info(f"Thread-{thread_id}: Successfully processed batch {base_index}-{base_index+len(comments)-1} with key ...{key[-10:]} and model {model}")
                            return {"comments": enriched_comments}
                except (json.JSONDecodeError, ValueError):
                    logger.warning(f"Thread-{thread_id}: Failed to parse response for batch {base_index}-{base_index+len(comments)-1}")
                    pass
            if response.status_code == 429 or "quota exceeded" in response.text.lower():
                logger.warning(f"Thread-{thread_id}: Rate limit/quota exceeded for key ...{key[-10:]} and model {model}")
                return None
            time.sleep(5)
        except Exception as e:
            logger.error(f"Thread-{thread_id}: Error processing batch {base_index}-{base_index+len(comments)-1}: {str(e)}")
            time.sleep(5)
    logger.error(f"Thread-{thread_id}: Failed to process batch {base_index}-{base_index+len(comments)-1} after all retries")
    return None


def worker(batch_queue, result_queue, key_list, model_list, key_lock, rate_limit_delay, thread_id):
    key_idx = thread_id % len(key_list)  # Each thread starts with a different key
    model_idx = thread_id % len(model_list)
    logger.info(f"Thread-{thread_id}: Starting with API key ...{key_list[key_idx][-10:]} and model {model_list[model_idx]}")
    
    while True:
        try:
            batch, base_index = batch_queue.get(timeout=2)
        except:
            break  # No more batches
        
        success = False
        for attempt in range(len(key_list)):
            with key_lock:
                key = key_list[key_idx]
                model = model_list[model_idx % len(model_list)]
                logger.info(f"Thread-{thread_id}: Attempt {attempt+1}/{len(key_list)} - Using key ...{key[-10:]} with model {model}")
            
            result = process_batch_threadsafe(batch, base_index, key, model, thread_id)
            if result:
                result_queue.put(result)
                success = True
                break
            else:
                # Rotate to next key/model
                with key_lock:
                    key_idx = (key_idx + 1) % len(key_list)
                    model_idx = (model_idx + 1) % len(model_list)
                    logger.info(f"Thread-{thread_id}: Rotating to key ...{key_list[key_idx][-10:]} and model {model_list[model_idx % len(model_list)]}")
        
        if not success:
            logger.error(f"Thread-{thread_id}: Failed to process batch {base_index}-{base_index+len(batch)-1} with all keys")
        
        time.sleep(rate_limit_delay)
        batch_queue.task_done()


def main_parallel():
    logger.info("Starting parallel comment processing script")
    
    # Check for existing progress
    start_index = get_last_processed_index()
    logger.info(f"Resuming from index: {start_index}")
    
    with open('sampled_comments.json', 'r', encoding='utf-8') as f:
        all_comments = json.load(f)
    total_comments = len(all_comments)
    logger.info(f"Found {total_comments} comments to process")

    batch_queue = Queue()
    # Only add batches from the start_index onwards
    for i in range(start_index, total_comments, BATCH_SIZE):
        batch_queue.put((all_comments[i:i+BATCH_SIZE], i))

    result_queue = Queue()
    key_lock = threading.Lock()
    num_threads = min(len(API_KEYS), batch_queue.qsize())
    threads = []

    processed_comments = [start_index]  # Start from where we left off
    progress_lock = threading.Lock()

    def progress_worker():
        last_reported = start_index
        while True:
            with progress_lock:
                done = processed_comments[0]
            if done >= total_comments:
                break
            if done != last_reported:
                percent = (done / total_comments) * 100
                logger.info(f"Progress: {done}/{total_comments} ({percent:.2f}%)")
                print(f"Progress: {done}/{total_comments} ({percent:.2f}%)")
                last_reported = done
            time.sleep(2)

    # Start progress reporter
    progress_thread = threading.Thread(target=progress_worker)
    progress_thread.start()

    logger.info(f"Starting {num_threads} threads with {len(API_KEYS)} API keys and {len(MODELS)} models")
    for idx in range(num_threads):
        t = threading.Thread(target=worker, args=(batch_queue, result_queue, API_KEYS, MODELS, key_lock, RATE_LIMIT_DELAY, idx))
        t.start()
        threads.append(t)

    batch_queue.join()
    for t in threads:
        t.join()
    with progress_lock:
        processed_comments[0] = total_comments  # Ensure 100% at end
    progress_thread.join()

    # Load existing results if any
    existing_results = []
    try:
        with open('labeled_comments_openrouter.json', 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing_results = existing_data.get('comments', [])
            logger.info(f"Loaded {len(existing_results)} existing results")
    except (FileNotFoundError, json.JSONDecodeError):
        logger.info("No existing results found, starting fresh")

    # Collect new results
    new_results = []
    while not result_queue.empty():
        new_results.extend(result_queue.get()['comments'])
    
    # Combine and sort all results
    all_results = existing_results + new_results
    all_results.sort(key=lambda x: x['index'])

    with open('labeled_comments_openrouter.json', 'w', encoding='utf-8') as f:
        json.dump({'comments': all_results}, f, indent=2, ensure_ascii=False)
    logger.info(f"Parallel processing complete! Total results: {len(all_results)}")

def save_results(results: Dict, output_file: str, is_first_batch: bool):
    """Save or append results to the output file."""
    if is_first_batch:
        # For first batch, create new file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        # For subsequent batches, read existing, append, and save
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # Append new comments
            existing_data["comments"].extend(results["comments"])
            
            # Save updated data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
        except (FileNotFoundError, json.JSONDecodeError):
            # If there's an error with the existing file, save as new
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

def estimate_time_remaining(start_time: datetime, current_index: int, total_comments: int) -> str:
    """Estimate remaining time based on current progress."""
    elapsed_time = datetime.now() - start_time
    if current_index == 0:
        return "Calculating..."
    
    comments_per_second = current_index / elapsed_time.total_seconds()
    remaining_comments = total_comments - current_index
    if comments_per_second > 0:
        seconds_remaining = remaining_comments / comments_per_second
        remaining_time = timedelta(seconds=int(seconds_remaining))
        return str(remaining_time)
    return "Calculating..."

if __name__ == "__main__":
    main_parallel() 