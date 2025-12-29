#!/usr/bin/env python3
"""
Auto-Embedding Comments from Google Drive
This script monitors a specific folder in Google Drive for new comment files
and automatically generates embeddings for them using the same logic as embedComments (1).ipynb
"""

import os
import json
import torch
import numpy as np
import shutil
import time
import logging
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any
import gc
import psutil
from contextlib import contextmanager
import threading
import glob
import sys
from packaging import version

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def install_and_check_packages():
    """Install required packages and check for compatibility issues"""
    logger.info("Installing required packages...")

    # Install packages - updated to fix dependency issues
    os.system('pip install -U transformers accelerate sentencepiece packaging')
    os.system('pip install "numpy<2.0"')

    # Simple version check without exiting
    try:
        from transformers import __version__ as transformers_version
        logger.info(f"Using transformers {transformers_version} with numpy {np.__version__}")
    except (ImportError, ModuleNotFoundError):
        # If transformers isn't installed, the script will fail on the next import anyway.
        # This check is specifically for the version mismatch.
        pass

    logger.info("✅ Required packages installed and checked")

def mount_drive():
    """Mount Google Drive and handle Colab environment"""
    logger.info("Attempting to mount Google Drive...")
    
    try:
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive')
        logger.info("✅ Google Drive mounted successfully")
        return True
    except ModuleNotFoundError:
        logger.warning("⚠️  Not in Google Colab environment - Drive mounting skipped")
        return False

class BatchOptimizer:
    """Dynamic batch size optimizer that adjusts based on GPU memory usage"""

    def __init__(self, initial_batch_size: int = 8, max_batch_size: int = 256,
                 target_memory_utilization: float = 0.85, device: str = "cuda"):
        """
        Initialize the batch optimizer

        Args:
            initial_batch_size: Starting batch size
            max_batch_size: Maximum allowed batch size
            target_memory_utilization: Target GPU memory utilization (0.0-1.0)
            device: Device type ("cuda", "cpu", "mps")
        """
        self.current_batch_size = initial_batch_size
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.target_memory_utilization = target_memory_utilization
        self.device = device

        # Tracking variables
        self.successful_batches = 0
        self.memory_usage_history = []
        self.batch_size_history = []
        self.last_oom = False
        self.adjustment_cooldown = 0

        logger.info(f"BatchOptimizer initialized - Initial batch size: {self.current_batch_size}")

    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage information"""
        if self.device != "cuda" or not torch.cuda.is_available():
            return {"used": 0.0, "total": 1.0, "utilization": 0.0}

        try:
            # Get memory info for the current device
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            utilization = memory_used / memory_total if memory_total > 0 else 0.0

            return {
                "used": memory_used,
                "total": memory_total,
                "utilization": utilization
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
            return {"used": 0.0, "total": 1.0, "utilization": 0.0}

    def should_increase_batch_size(self, memory_info: Dict[str, float]) -> bool:
        """Determine if batch size should be increased"""
        if self.last_oom or self.adjustment_cooldown > 0:
            return False

        # Only increase if we have successful batches and low memory usage
        return (self.successful_batches >= 3 and
                memory_info["utilization"] < self.target_memory_utilization - 0.1 and
                self.current_batch_size < self.max_batch_size)

    def should_decrease_batch_size(self, memory_info: Dict[str, float]) -> bool:
        """Determine if batch size should be decreased"""
        # Decrease if memory usage is too high
        return memory_info["utilization"] > self.target_memory_utilization + 0.05

    def adjust_batch_size(self, memory_info: Dict[str, float]) -> int:
        """Adjust batch size based on current memory usage"""
        old_batch_size = self.current_batch_size

        if self.adjustment_cooldown > 0:
            self.adjustment_cooldown -= 1
            return self.current_batch_size

        if self.should_increase_batch_size(memory_info):
            # Increase batch size gradually
            increase_factor = 1.5 if memory_info["utilization"] < 0.6 else 1.2
            new_batch_size = min(int(self.current_batch_size * increase_factor), self.max_batch_size)
            self.current_batch_size = new_batch_size
            self.adjustment_cooldown = 2  # Wait 2 batches before next adjustment
            logger.info(f"📈 Increased batch size: {old_batch_size} → {new_batch_size} (GPU: {memory_info['utilization']:.1%})")

        elif self.should_decrease_batch_size(memory_info):
            # Decrease batch size more aggressively
            decrease_factor = 0.7 if memory_info["utilization"] > 0.95 else 0.8
            new_batch_size = max(int(self.current_batch_size * decrease_factor), 1)
            self.current_batch_size = new_batch_size
            self.adjustment_cooldown = 3  # Wait longer after decreasing
            logger.info(f"📉 Decreased batch size: {old_batch_size} → {new_batch_size} (GPU: {memory_info['utilization']:.1%})")

        return self.current_batch_size

    def handle_oom_error(self):
        """Handle out-of-memory error by reducing batch size"""
        old_batch_size = self.current_batch_size
        self.current_batch_size = max(1, self.current_batch_size // 2)
        self.last_oom = True
        self.adjustment_cooldown = 5  # Wait longer after OOM
        self.successful_batches = 0 # Reset confidence after OOM
        logger.error(f"💥 OOM Error! Reduced batch size: {old_batch_size} → {self.current_batch_size}")

        # Clear GPU cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def record_successful_batch(self, memory_info: Dict[str, float]):
        """Record a successful batch processing"""
        self.successful_batches += 1
        self.last_oom = False
        self.memory_usage_history.append(memory_info["utilization"])
        self.batch_size_history.append(self.current_batch_size)

        # Keep only recent history
        if len(self.memory_usage_history) > 50:
            self.memory_usage_history = self.memory_usage_history[-50:]
            self.batch_size_history = self.batch_size_history[-50:]

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not self.memory_usage_history:
            return {"status": "No data available"}

        return {
            "current_batch_size": self.current_batch_size,
            "initial_batch_size": self.initial_batch_size,
            "successful_batches": self.successful_batches,
            "avg_memory_utilization": np.mean(self.memory_usage_history),
            "max_memory_utilization": np.max(self.memory_usage_history),
            "batch_size_range": f"{min(self.batch_size_history)}-{max(self.batch_size_history)}"
        }

class CommentEmbedder:
    def __init__(self,
                 model_name: str = "intfloat/multilingual-e5-small",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 8,
                 use_fp16: bool = True,
                 compile_model: bool = False,  # Changed to False for Colab compatibility
                 optimize_batch_size: bool = True,
                 max_batch_size: int = 256):
        """Initialise tokenizer/model with optional half-precision and Torch compile.

        Args:
            model_name: HuggingFace model id
            device: "cuda" | "cpu" | "mps"
            batch_size: initial batch size for embedding
            use_fp16: Cast model to float16 when running on GPU for speed & memory
            compile_model: Run `torch.compile` (PyTorch ≥2.0) for kernel fusion
            optimize_batch_size: Enable dynamic batch size optimization
            max_batch_size: Maximum allowed batch size for optimization
        """

        self.device = device
        self.model_name = model_name

        # Initialize batch optimizer
        self.optimize_batch_size = optimize_batch_size and device == "cuda"
        if self.optimize_batch_size:
            self.batch_optimizer = BatchOptimizer(
                initial_batch_size=batch_size,
                max_batch_size=max_batch_size,
                device=device
            )
            self.batch_size = self.batch_optimizer.current_batch_size
        else:
            self.batch_size = batch_size
            self.batch_optimizer = None

        # Load tokenizer / model with multiple fallback strategies
        try:
            logger.info(f"Loading tokenizer and model: {model_name}")
            # First try with standard parameters
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=False)
        except Exception as e:
            logger.warning(f"Standard loading failed: {e}")
            logger.info("Trying with trust_remote_code=True...")
            try:
                # If standard loading fails, try with trust_remote_code=True
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            except Exception as e2:
                logger.warning(f"Loading with trust_remote_code=True also failed: {e2}")
                logger.info("Trying with additional parameters...")
                try:
                    # If that also fails, try with additional parameters
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        trust_remote_code=True,
                        local_files_only=False
                    )
                    self.model = AutoModel.from_pretrained(
                        model_name, 
                        trust_remote_code=True,
                        local_files_only=False
                    )
                except Exception as e3:
                    logger.error(f"All loading attempts failed: {e3}")
                    logger.info("Attempting to clear cache and retry...")
                    # Clear cache and try again
                    os.system("rm -rf ~/.cache/huggingface/transformers/")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                    except Exception as e4:
                        logger.error(f"Final loading attempt failed: {e4}")
                        raise

        # Move to device & precision
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True  # potentially faster
            if use_fp16:
                logger.info("Converting model to half precision (fp16)")
                self.model = self.model.half()

        self.model = self.model.to(self.device)

        # Optionally compile (PyTorch 2.x) - made optional for Colab compatibility
        if compile_model and version.parse(torch.__version__) >= version.parse("2.0") and self.device == "cuda":
            try:
                logger.info("Compiling model with torch.compile() for optimization")
                self.model = torch.compile(self.model)
                logger.info("✅ Model compiled with torch.compile()")
            except Exception as compile_err:  # pragma: no cover
                logger.warning(f"torch.compile failed: {compile_err}. Continuing without compilation.")

        logger.info(f"✅ Model loaded on {self.device} (fp16={use_fp16})")
        if self.device == "cuda":
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU Memory: {total_mem:.2f} GB")
            if self.optimize_batch_size:
                logger.info("🚀 Dynamic batch size optimization enabled")

    def prepare_comment(self, text: str) -> str:
        """Prepare comment text following E5 format"""
        # Clean and format text for the E5 model (prefix with 'query: ')
        text = str(text).strip()
        if not text:
            return "query: empty comment"
        return f"query: {text}"

    @torch.no_grad()
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a single batch of texts."""
        # This method will now raise torch.cuda.OutOfMemoryError on failure,
        # to be handled by the calling function.

        # Prepare texts for the model
        prepared_texts = [self.prepare_comment(text) for text in texts]

        # Tokenize the batch of texts
        encoded = self.tokenizer(
            prepared_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # Get model outputs (embeddings)
        outputs = self.model(**encoded)
        # Use the CLS token embedding as the sentence embedding
        embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        # Normalize embeddings to unit length
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Record successful batch if optimizing
        if self.batch_optimizer:
            memory_info_after = self.batch_optimizer.get_gpu_memory_info()
            self.batch_optimizer.record_successful_batch(memory_info_after)

        return embeddings.cpu().numpy()

    def process_file(self, input_path: Path, output_path: Path = None) -> Dict[str, Any]:
        """Process a single JSON file with dynamic batch optimization and checkpointing."""
        logger.info(f"🔄 Starting to process file: {input_path.name}")
        logger.info(f"   📁 Input path: {input_path}")
        logger.info(f"   💾 Output path: {output_path if output_path else input_path.parent / f'{input_path.stem}_embeddings.npz'}")

        # Set output path if not provided
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_embeddings.npz"

        # Temporary directory for checkpointing batches
        tmp_dir = output_path.parent / f".tmp_{input_path.stem}"
        tmp_dir.mkdir(exist_ok=True)
        logger.debug(f"   📂 Created temporary directory: {tmp_dir}")

        # Load comments from JSON file
        logger.info(f"   📖 Loading comments from JSON file...")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle both array of strings and array of objects
        comments = []
        comment_ids = []

        # Extract comments and their IDs (if present)
        for idx, item in enumerate(data):
            if isinstance(item, dict) and 'comment' in item:
                # Handle object format
                comments.append(item['comment'])
                comment_ids.append(item.get('id', idx))
            elif isinstance(item, str):
                # Handle string format
                comments.append(item)
                comment_ids.append(idx)

        if not comments:
            # Clean up temp dir if no comments are found
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            raise ValueError(f"No valid comments found in {input_path}")

        logger.info(f"✅ Found {len(comments)} comments to process from {input_path.name}")

        # --- Resume logic ---
        all_embeddings = []
        processed_count = 0
        batch_counter = 0

        # Discover and load existing batches to resume
        try:
            # Sort by batch number to ensure correct order
            existing_batches = sorted(
                tmp_dir.glob("batch_*.npz"),
                key=lambda p: int(p.stem.split('_')[1])
            )

            if existing_batches:
                logger.info(f"🔄 Resuming from {len(existing_batches)} completed batches...")
                for batch_file in existing_batches:
                    with np.load(batch_file) as batch_data:
                        all_embeddings.append(batch_data['embeddings'])
                        processed_count += len(batch_data['ids'])
                batch_counter = len(existing_batches)
                logger.info(f"🔄 Resuming from comment #{processed_count}")

        except (ValueError, IndexError) as e:
            logger.warning(f"⚠️  Warning: Could not parse batch filenames in {tmp_dir}. Starting from scratch. Error: {e}")
            shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(exist_ok=True)

        # Process comments in batches with dynamic sizing
        i = processed_count
        logger.info(f"⚙️  Starting embedding process for {len(comments)} comments...")

        with tqdm(total=len(comments), initial=i, desc=f"Embedding {input_path.name}") as pbar:
            while i < len(comments):
                # Get current batch size (may change dynamically)
                current_batch_size = self.batch_optimizer.current_batch_size if self.batch_optimizer else self.batch_size

                # Get batch of comments
                end_idx = min(i + current_batch_size, len(comments))
                batch_comments = comments[i:end_idx]
                batch_ids = comment_ids[i:end_idx]

                if not batch_comments:
                    break

                try:
                    # Process the batch
                    embeddings = self.embed_batch(batch_comments)

                    # Save the current batch as a checkpoint
                    np.savez_compressed(
                        tmp_dir / f"batch_{batch_counter}.npz",
                        embeddings=embeddings,
                        ids=batch_ids
                    )
                    all_embeddings.append(embeddings)

                    # Update progress
                    processed_in_batch = end_idx - i
                    i = end_idx
                    pbar.update(processed_in_batch)
                    batch_counter += 1

                    # Optimize batch size periodically
                    if self.batch_optimizer and batch_counter % 5 == 0:
                        memory_info = self.batch_optimizer.get_gpu_memory_info()
                        self.batch_optimizer.adjust_batch_size(memory_info)

                    # Clear CUDA cache periodically
                    if self.device == "cuda" and batch_counter % 10 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                except torch.cuda.OutOfMemoryError:
                    logger.error(f"💥 Caught OOM error while processing {input_path.name}.")
                    if self.batch_optimizer:
                        self.batch_optimizer.handle_oom_error()
                        logger.info("🔄 Retrying batch with a smaller size...")
                        # The loop will continue and retry the same batch with the new, smaller size.
                        # `i` is not incremented, so we retry the same slice.
                        continue
                    else:
                        # If not optimizing, we cannot recover, so we raise.
                        logger.error("❌ Cannot recover from OOM without batch optimizer. Exiting file processing.")
                        raise
                except Exception as e:
                    logger.error(f"❌ An unexpected error occurred while processing {input_path.name}: {e}")
                    # For other errors, we should probably stop processing this file.
                    raise

        # Print optimization stats
        if self.batch_optimizer:
            stats = self.batch_optimizer.get_stats()
            logger.info(f"📊 Optimization Stats:")
            logger.info(f"   📈 Final batch size: {stats['current_batch_size']}")
            logger.info(f"   🔄 Successful batches: {stats['successful_batches']}")
            logger.info(f"   💻 Avg GPU utilization: {stats.get('avg_memory_utilization', 0):.1%}")
            logger.info(f"   📏 Batch size range: {stats.get('batch_size_range', 'N/A')}")

        # Concatenate all batch embeddings into a single array
        embeddings_array = np.vstack(all_embeddings)
        logger.info(f"✅ Embedding process completed. Array shape: {embeddings_array.shape}")

        # Save embeddings and IDs to a compressed .npz file
        logger.info(f"💾 Saving embeddings to: {output_path}")
        np.savez_compressed(
            output_path,
            embeddings=embeddings_array,
            ids=comment_ids
        )
        logger.info(f"✅ Embeddings saved successfully")

        # Clean up temporary directory on success
        shutil.rmtree(tmp_dir)
        logger.debug(f"🗑️  Cleaned up temporary directory: {tmp_dir}")

        result = {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "num_comments": len(comments),
            "embedding_dim": embeddings_array.shape[1]
        }

        # Add optimization stats to result
        if self.batch_optimizer:
            result["optimization_stats"] = self.batch_optimizer.get_stats()

        logger.info(f"🎉 Successfully processed {input_path.name} -> {output_path.name}")
        logger.info(f"   📊 Comments: {result['num_comments']:,}")
        logger.info(f"   📐 Embedding dimension: {result['embedding_dim']}")
        logger.info(f"   💾 Output file size: {output_path.stat().st_size / (1024*1024):.2f} MB")

        return result

class AutoEmbeddingMonitor:
    """Monitors a directory for new comment files and automatically embeds them"""

    def __init__(self, comments_dir: str = "/content/drive/My Drive/youtubeComments",
                 embeddings_dir: str = "/content/drive/My Drive/youtubeComments/embed"):
        self.comments_dir = Path(comments_dir)
        self.embeddings_dir = Path(embeddings_dir)
        
        # Create embeddings directory if it doesn't exist
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Track processed files to avoid reprocessing
        self.processed_files = set()
        
        # Initialize the embedder
        logger.info("Initializing Comment Embedder...")
        self.embedder = CommentEmbedder(
            model_name="intfloat/multilingual-e5-small",
            batch_size=8,
            max_batch_size=256,
            use_fp16=True,
            compile_model=False,  # Changed to False for Colab compatibility
            optimize_batch_size=True
        )
        
        # Thread control
        self.monitoring_thread = None
        self.monitoring_active = False
        
        logger.info(f"✅ AutoEmbeddingMonitor initialized")
        logger.info(f"📁 Monitoring directory: {self.comments_dir}")
        logger.info(f"💾 Output directory: {self.embeddings_dir}")
        
    def get_new_files(self) -> List[Path]:
        """Get list of new JSON files that haven't been processed yet"""
        logger.debug(f"🔍 Checking for new files in: {self.comments_dir}")

        if not self.comments_dir.exists():
            logger.warning(f"Comments directory does not exist: {self.comments_dir}")
            return []

        # Get all JSON files in the comments directory
        all_json_files = list(self.comments_dir.glob("*.json"))
        logger.debug(f"📁 Found {len(all_json_files)} JSON files in directory")

        # Filter out files that have already been processed
        new_files = [f for f in all_json_files if f not in self.processed_files]
        logger.info(f"🆕 Found {len(new_files)} new files to process")

        # Log the names of new files if any are found
        if new_files:
            for file_path in new_files:
                logger.info(f"   📄 New file: {file_path.name}")

        return new_files
    
    def process_new_files(self):
        """Process all new comment files"""
        logger.info("🔄 Starting file processing cycle...")
        new_files = self.get_new_files()

        if not new_files:
            logger.info("✅ No new files to process in this cycle")
            return 0  # Return count of processed files

        logger.info(f"📁 Found {len(new_files)} new files to process:")
        for file_path in new_files:
            logger.info(f"   - {file_path.name}")

        processed_count = 0
        total_files = len(new_files)

        for i, file_path in enumerate(new_files, 1):
            try:
                logger.info(f"\n🔄 Processing file {i}/{total_files}: {file_path.name}")
                logger.info(f"   📁 Input: {file_path}")
                logger.info(f"   💾 Output will be: {self.embeddings_dir / f'{file_path.stem}_embeddings.npz'}")

                # Create output path in embeddings directory
                output_path = self.embeddings_dir / f"{file_path.stem}_embeddings.npz"

                # Check if embedding file already exists
                if output_path.exists():
                    logger.warning(f"⚠️  Embedding file already exists: {output_path.name}")
                    # Add to processed files to avoid reprocessing
                    self.processed_files.add(file_path)
                    continue

                # Process the file
                logger.info(f"   ⚙️  Starting embedding process...")
                result = self.embedder.process_file(file_path, output_path)
                logger.info(f"   ✅ Embedding process completed successfully")

                logger.info(f"✅ Successfully processed: {file_path.name}")
                logger.info(f"   📊 Comments processed: {result['num_comments']:,}")
                logger.info(f"   📐 Embedding dimension: {result['embedding_dim']}")
                logger.info(f"   💾 Output saved to: {output_path.name}")

                # Mark as processed
                self.processed_files.add(file_path)
                processed_count += 1
                logger.info(f"   📈 Progress: {processed_count}/{total_files} files processed in this cycle")

            except Exception as e:
                logger.error(f"❌ Error processing {file_path.name}: {str(e)}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                continue

        logger.info(f"✅ Completed processing cycle - {processed_count}/{total_files} files processed")
        return processed_count
    
    def monitoring_loop(self, check_interval: int = 5):
        """The main monitoring loop that runs in a separate thread"""
        logger.info(f"👀 Monitoring started (checking every {check_interval} seconds)")
        logger.info(f"📁 Monitoring directory: {self.comments_dir}")
        logger.info(f"💾 Output directory: {self.embeddings_dir}")

        while self.monitoring_active:
            try:
                logger.debug(f"🔍 Checking for new files at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                processed_count = self.process_new_files()
                if processed_count > 0:
                    logger.info(f"✅ Processed {processed_count} new file(s)")
                else:
                    logger.debug(f"No new files found in the last check at {time.strftime('%Y-%m-%d %H:%M:%S')}")

                # Sleep for the specified interval, but wake up if monitoring stops
                slept = 0
                while slept < check_interval and self.monitoring_active:
                    time.sleep(1)
                    slept += 1

            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {str(e)}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                time.sleep(check_interval)

        logger.info("🛑 Monitoring stopped")
    
    def start_monitoring(self, check_interval: int = 5):
        """Start monitoring for new files in a separate thread"""
        if self.monitoring_active:
            logger.warning("⚠️  Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self.monitoring_loop, 
            args=(check_interval,),
            daemon=True  # Dies when main thread dies
        )
        self.monitoring_thread.start()
        logger.info(f"✅ Monitoring started in background (checking every {check_interval} seconds)")
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=2)  # Wait up to 2 seconds for thread to finish
            logger.info("🛑 Monitoring stopped")
        else:
            logger.warning("⚠️  Monitoring is not active")
    
    def process_once(self):
        """Process new files once without continuous monitoring"""
        logger.info("🔄 Processing new files once...")
        processed_count = self.process_new_files()
        logger.info(f"✅ Processing complete ({processed_count} files processed)")

def main():
    """Main function to run the auto-embedding process"""
    logger.info("🚀 Starting Auto-Embedding Comments from Google Drive")
    logger.info("   Version: 1.0")
    logger.info("   Monitoring: Every 5 seconds")
    logger.info("   Output directory: /content/drive/My Drive/youtubeComments/embed")

    # Install packages and check compatibility
    logger.info("📦 Installing required packages...")
    install_and_check_packages()

    # Mount Google Drive
    logger.info("💾 Mounting Google Drive...")
    drive_mounted = mount_drive()

    if not drive_mounted:
        logger.error("❌ Could not mount Google Drive. Please run in Google Colab environment.")
        return

    logger.info("✅ Google Drive mounted successfully")

    # Initialize the auto-embedding monitor
    logger.info("⚙️  Initializing Auto-Embedding Monitor...")
    monitor = AutoEmbeddingMonitor(
        comments_dir="/content/drive/My Drive/youtubeComments",
        embeddings_dir="/content/drive/My Drive/youtubeComments/embed"
    )

    # Process any new files that are currently in the directory
    logger.info("🔍 Processing any existing files in the directory...")
    monitor.process_once()

    # Start continuous monitoring in background
    logger.info("\n" + "="*70)
    logger.info("🎯 STARTING CONTINUOUS MONITORING")
    logger.info("="*70)
    logger.info("📁 Monitoring directory: /content/drive/My Drive/youtubeComments")
    logger.info("💾 Output directory: /content/drive/My Drive/youtubeComments/embed")
    logger.info("⏰ Check interval: Every 5 seconds")
    logger.info("🔄 New files will be automatically processed when uploaded")
    logger.info("🛑 To stop monitoring, use Ctrl+C or call monitor.stop_monitoring()")
    logger.info("="*70)

    logger.info("🚀 Starting monitoring service...")
    monitor.start_monitoring(check_interval=5)  # Check every 5 seconds

    # Keep the main thread alive
    try:
        logger.info("✅ Monitoring is now running in the background!")
        logger.info("💡 You can continue using this notebook while monitoring runs.")
        logger.info("   Files uploaded to the monitored directory will be processed automatically.")

        while monitor.monitoring_active:
            time.sleep(10)  # Sleep in small intervals to allow for interruption
    except KeyboardInterrupt:
        logger.info("\n🛑 Received interrupt signal (Ctrl+C), stopping monitoring...")
        monitor.stop_monitoring()
        logger.info("✅ Monitoring has been stopped successfully.")

if __name__ == "__main__":
    main()