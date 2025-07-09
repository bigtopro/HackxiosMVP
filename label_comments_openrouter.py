import json
import requests
import os
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

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

def process_batch(comments: List[str], base_index: int, key_manager: APIKeyManager, model_manager: ModelManager) -> Optional[Dict]:
    """Process a batch of comments using OpenRouter API with key and model rotation."""
    messages = create_prompt(comments)
    max_retries = len(key_manager.api_keys) * len(model_manager.models)
    
    logger.info(f"Processing batch of {len(comments)} comments (indices {base_index}-{base_index + len(comments) - 1})")
    logger.info(f"Available keys: {len(key_manager.api_keys)}, Available models: {len(model_manager.models)}")
    
    for attempt in range(max_retries):
        current_key = key_manager.get_current_key()
        current_model = model_manager.get_current_model()
        key_retry_count = 0
        max_key_retries = 3
        
        while key_retry_count < max_key_retries:
            key_manager.key_usage[current_key]["attempts"] += 1
            model_manager.model_usage[current_model]["attempts"] += 1
            
            logger.info(f"Attempt {key_retry_count + 1}/{max_key_retries}")
            logger.info(f"Using API key: ...{current_key[-10:]} with model: {current_model}")
            
            try:
                response = requests.post(
                    url=API_URL,
                    headers={
                        "Authorization": f"Bearer {current_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/yourusername/LUCY-1",
                        "X-Title": "LUCY-1 Comment Analyzer"
                    },
                    data=json.dumps({
                        "model": current_model,
                        "messages": messages
                    }, ensure_ascii=False).encode('utf-8'),
                    timeout=480
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Log token usage
                    if "usage" in result:
                        usage = result["usage"]
                        logger.info("Token Usage:")
                        logger.info(f"  Prompt tokens: {usage.get('prompt_tokens', 0)}")
                        logger.info(f"  Completion tokens: {usage.get('completion_tokens', 0)}")
                        logger.info(f"  Total tokens: {usage.get('total_tokens', 0)}")
                    
                    content = result["choices"][0]["message"]["content"]
                    content = content.replace("```json", "").replace("```", "").strip()
                    
                    try:
                        parsed = json.loads(content)
                        
                        if isinstance(parsed, dict) and "comments" in parsed and parsed["comments"]:
                            enriched_comments = []
                            for comment_result in parsed["comments"]:
                                if not isinstance(comment_result, dict) or "index" not in comment_result or "label" not in comment_result:
                                    raise ValueError("Invalid comment format in response")
                                
                                original_index = base_index + comment_result["index"]
                                if 0 <= comment_result["index"] < len(comments):
                                    enriched_comments.append({
                                        "index": original_index,
                                        "label": comment_result["label"],
                                        "text": comments[comment_result["index"]]
                                    })
                            
                            if enriched_comments:
                                key_manager.key_usage[current_key]["successes"] += 1
                                model_manager.model_usage[current_model]["successes"] += 1
                                logger.info(f"Successfully processed {len(enriched_comments)} comments")
                                return {"comments": enriched_comments}
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Error processing response: {str(e)}")
                
                if response.status_code == 429 or "quota exceeded" in response.text.lower():
                    logger.warning(f"Model quota exceeded for {current_model}")
                    model_manager.mark_model_failed(current_model)
                    if model_manager.models:
                        model_manager.rotate_model()
                        break
                    else:
                        logger.critical("All models have failed!")
                        return None
                
                key_retry_count += 1
                if key_retry_count < max_key_retries:
                    logger.info(f"Retrying with same key and model (Attempt {key_retry_count + 1}/{max_key_retries})")
                    time.sleep(5)
                else:
                    logger.warning(f"Key failed after {max_key_retries} attempts")
                    key_manager.mark_key_failed(current_key)
                    key_manager.rotate_key()
                    break
                
            except Exception as e:
                logger.error(f"Error with current key and model: {str(e)}")
                key_retry_count += 1
                if key_retry_count < max_key_retries:
                    logger.info(f"Retrying with same key and model (Attempt {key_retry_count + 1}/{max_key_retries})")
                    time.sleep(5)
                else:
                    logger.warning(f"Key failed after {max_key_retries} attempts")
                    key_manager.mark_key_failed(current_key)
                    key_manager.rotate_key()
                    break
    
    logger.error("All attempts failed for this batch")
    return None

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

def main():
    """Main function to process comments."""
    logger.info("Starting comment processing script")
    
    # Initialize managers
    key_manager = APIKeyManager(API_KEYS)
    model_manager = ModelManager(MODELS)
    
    try:
        # Read all comments
        logger.info("Reading comments from file...")
        all_comments = read_all_comments('comments.json')
        total_comments = len(all_comments)
        logger.info(f"Found {total_comments} comments to process")
        
        # Get the last processed index
        start_index = get_last_processed_index()
        logger.info(f"Resuming from index: {start_index}")
        
        # Initialize results structure
        if start_index == 0:
            results = {"comments": []}
            logger.info("Starting fresh processing")
        else:
            try:
                with open('labeled_comments_openrouter.json', 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    logger.info(f"Loaded {len(results['comments'])} existing results")
            except (FileNotFoundError, json.JSONDecodeError):
                results = {"comments": []}
                logger.warning("Could not load existing results, starting fresh")
        
        start_time = datetime.now()
        logger.info(f"Processing started at: {start_time}")
        
        # Process comments in batches
        for i in range(start_index, total_comments, BATCH_SIZE):
            batch = all_comments[i:i + BATCH_SIZE]
            
            # Log batch information
            logger.info(f"\nProcessing batch {i//BATCH_SIZE + 1} of {(total_comments + BATCH_SIZE - 1)//BATCH_SIZE}")
            logger.info(f"Batch size: {len(batch)} comments")
            
            # Process the batch
            batch_results = process_batch(batch, i, key_manager, model_manager)
            
            if batch_results is None:
                logger.error("Failed to process batch after all retries")
                save_checkpoint(i)
                break
            
            # Save results
            save_results(batch_results, 'labeled_comments_openrouter.json', i == 0)
            save_checkpoint(i + len(batch))
            
            # Calculate and display progress
            progress = (i + len(batch)) / total_comments * 100
            time_remaining = estimate_time_remaining(start_time, i + len(batch), total_comments)
            
            logger.info(f"Progress: {progress:.2f}%")
            logger.info(f"Time remaining: {time_remaining}")
            logger.info(f"Available API keys: {len(key_manager.api_keys)}")
            logger.info(f"Available models: {len(model_manager.models)}")
            
            # Log usage statistics
            logger.info("\nCurrent Statistics:")
            for key in key_manager.key_usage:
                if key_manager.key_usage[key]["attempts"] > 0:
                    logger.info(f"Key ...{key[-10:]}: {key_manager.key_usage[key]}")
            for model in model_manager.model_usage:
                if model_manager.model_usage[model]["attempts"] > 0:
                    logger.info(f"Model {model}: {model_manager.model_usage[model]}")
            
            # Rate limiting delay
            if i + BATCH_SIZE < total_comments:
                logger.info(f"Waiting {RATE_LIMIT_DELAY} seconds before next batch...")
                time.sleep(RATE_LIMIT_DELAY)
        
        logger.info("\nProcessing complete!")
        logger.info(f"Total time elapsed: {datetime.now() - start_time}")
        
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user")
        logger.info("Saving progress...")
        save_checkpoint(i)
        logger.info(f"Saved checkpoint at index {i}")
    
    except Exception as e:
        logger.critical(f"\nUnexpected error: {str(e)}")
        logger.info("Saving progress...")
        save_checkpoint(i)
        logger.info(f"Saved checkpoint at index {i}")
        raise

if __name__ == "__main__":
    main() 