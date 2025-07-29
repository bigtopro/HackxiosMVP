# Imports
import os
import json
import torch
import numpy as np
import shutil

# Pre-flight check for common numpy/transformers compatibility issues
import sys
from packaging import version
try:
    from transformers import __version__ as transformers_version
    if version.parse(np.__version__) >= version.parse("2.0.0") and \
       version.parse(transformers_version) < version.parse("4.41.0"):
        print("="*80, file=sys.stderr)
        print("ERROR: Incompatible library versions detected!", file=sys.stderr)
        print(f"Numpy version {np.__version__} is not compatible with Transformers version {transformers_version}.", file=sys.stderr)
        print("\nTo fix, run ONE of the following commands in your notebook and RESTART the runtime:", file=sys.stderr)
        print("1. Downgrade NumPy: !pip install --upgrade \"numpy<2.0\"", file=sys.stderr)
        print("2. Upgrade Transformers: !pip install --upgrade transformers accelerate", file=sys.stderr)
        print("="*80, file=sys.stderr)
        sys.exit(1)
except (ImportError, ModuleNotFoundError):
    # If transformers isn't installed, the script will fail on the next import anyway.
    # This check is specifically for the version mismatch.
    pass

from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any
import gc
import time
import psutil
# Optional: speed/precision helpers
from contextlib import contextmanager

# Skip Drive import when not in Colab; this keeps the script runnable locally
try:
    from google.colab import drive  # type: ignore
except ModuleNotFoundError:
    drive = None  # noqa: F401

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
        
        print(f"BatchOptimizer initialized - Initial batch size: {self.current_batch_size}")
        
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
            print(f"Error getting GPU memory info: {e}")
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
            print(f"📈 Increased batch size: {old_batch_size} → {new_batch_size} (GPU: {memory_info['utilization']:.1%})")
            
        elif self.should_decrease_batch_size(memory_info):
            # Decrease batch size more aggressively
            decrease_factor = 0.7 if memory_info["utilization"] > 0.95 else 0.8
            new_batch_size = max(int(self.current_batch_size * decrease_factor), 1)
            self.current_batch_size = new_batch_size
            self.adjustment_cooldown = 3  # Wait longer after decreasing
            print(f"📉 Decreased batch size: {old_batch_size} → {new_batch_size} (GPU: {memory_info['utilization']:.1%})")
        
        return self.current_batch_size
    
    def handle_oom_error(self):
        """Handle out-of-memory error by reducing batch size"""
        old_batch_size = self.current_batch_size
        self.current_batch_size = max(1, self.current_batch_size // 2)
        self.last_oom = True
        self.adjustment_cooldown = 5  # Wait longer after OOM
        self.successful_batches = 0 # Reset confidence after OOM
        print(f"💥 OOM Error! Reduced batch size: {old_batch_size} → {self.current_batch_size}")
        
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

# This class handles loading a transformer model and embedding comment texts
class CommentEmbedder:
    def __init__(self,
                 model_name: str = "intfloat/multilingual-e5-small",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 8,
                 use_fp16: bool = True,
                 compile_model: bool = True,
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

        # Load tokenizer / model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Move to device & precision
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True  # potentially faster
            if use_fp16:
                self.model = self.model.half()

        self.model = self.model.to(self.device)

        # Optionally compile (PyTorch 2.x)
        if compile_model and version.parse(torch.__version__) >= version.parse("2.0") and self.device == "cuda":
            try:
                self.model = torch.compile(self.model)
                print("Model compiled with torch.compile()")
            except Exception as compile_err:  # pragma: no cover
                print(f"torch.compile failed: {compile_err}. Continuing without compilation.")

        print(f"Model loaded on {self.device} (fp16={use_fp16})")
        if self.device == "cuda":
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory: {total_mem:.2f} GB")
            if self.optimize_batch_size:
                print("🚀 Dynamic batch size optimization enabled")
    
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
        # Set output path if not provided
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_embeddings.npz"

        # Temporary directory for checkpointing batches
        tmp_dir = output_path.parent / f".tmp_{input_path.stem}"
        tmp_dir.mkdir(exist_ok=True)
            
        # Load comments from JSON file
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
            
        print(f"\nProcessing {len(comments)} comments from {input_path.name}")
        
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
                print(f"Resuming from {len(existing_batches)} completed batches...")
                for batch_file in existing_batches:
                    with np.load(batch_file) as batch_data:
                        all_embeddings.append(batch_data['embeddings'])
                        processed_count += len(batch_data['ids'])
                batch_counter = len(existing_batches)
                print(f"Resuming from comment #{processed_count}")

        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse batch filenames in {tmp_dir}. Starting from scratch. Error: {e}")
            shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(exist_ok=True)

        # Process comments in batches with dynamic sizing
        i = processed_count
        
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
                    print(f"\nCaught OOM error while processing {input_path.name}.")
                    if self.batch_optimizer:
                        self.batch_optimizer.handle_oom_error()
                        print("Retrying batch with a smaller size...")
                        # The loop will continue and retry the same batch with the new, smaller size.
                        # `i` is not incremented, so we retry the same slice.
                        continue
                    else:
                        # If not optimizing, we cannot recover, so we raise.
                        print("Cannot recover from OOM without batch optimizer. Exiting file processing.")
                        raise
                except Exception as e:
                    print(f"\nAn unexpected error occurred while processing {input_path.name}: {e}")
                    # For other errors, we should probably stop processing this file.
                    raise
        
        # Print optimization stats
        if self.batch_optimizer:
            stats = self.batch_optimizer.get_stats()
            print(f"🎯 Optimization Stats:")
            print(f"   Final batch size: {stats['current_batch_size']}")
            print(f"   Successful batches: {stats['successful_batches']}")
            print(f"   Avg GPU utilization: {stats.get('avg_memory_utilization', 0):.1%}")
            print(f"   Batch size range: {stats.get('batch_size_range', 'N/A')}")
        
        # Concatenate all batch embeddings into a single array
        embeddings_array = np.vstack(all_embeddings)
        
        # Save embeddings and IDs to a compressed .npz file
        np.savez_compressed(
            output_path,
            embeddings=embeddings_array,
            ids=comment_ids
        )
        
        # Clean up temporary directory on success
        shutil.rmtree(tmp_dir)

        result = {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "num_comments": len(comments),
            "embedding_dim": embeddings_array.shape[1]
        }
        
        # Add optimization stats to result
        if self.batch_optimizer:
            result["optimization_stats"] = self.batch_optimizer.get_stats()
            
        return result

# Main function to process all comment files in a directory

def main():
    # Mount Google Drive (for Colab usage)
    if drive:
        drive.mount('/content/drive')
        # Colab environment - use Google Drive path
        base_dir = Path('/content/drive/My Drive/youtube_embeddings_project')
        comments_dir = base_dir / 'comments'
        output_dir = base_dir / 'embeddings'
    else:
        # Local environment - use current working directory
        base_dir = Path('.')
        comments_dir = base_dir / 'comments'
        output_dir = base_dir / 'embeddings'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Reading comments from: {comments_dir}")
    print(f"Saving embeddings to: {output_dir}")
    
    # Initialize the embedder with model and batch size optimization
    embedder = CommentEmbedder(
        model_name="intfloat/multilingual-e5-small",
        batch_size=8,  # Starting batch size (will be optimized)
        max_batch_size=256,  # Maximum batch size for optimization
        use_fp16=True, # Enable half-precision for faster inference
        compile_model=True, # Enable Torch compile for faster inference
        optimize_batch_size=True  # Enable dynamic batch size optimization
    )
    
    # Prepare to process all JSON files in the comments directory
    results = []
    
    # List all JSON files to process, sorted for deterministic order
    json_files = sorted(list(comments_dir.glob("*.json")))
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each JSON file and save embeddings
    for json_file in tqdm(json_files, desc="Overall Progress"):
        output_path = output_dir / f"{json_file.stem}_embeddings.npz"

        # Checkpoint: Skip if the final output file already exists
        if output_path.exists():
            print(f"\n✔️ Skipping already completed file: {json_file.name}")
            continue
        
        try:
            result = embedder.process_file(json_file, output_path)
            results.append(result)
            print(f"✅ Saved {result['num_comments']} embeddings to {result['output_file']}")
            print(f"   Embedding dimension: {result['embedding_dim']}")
        except Exception as e:
            print(f"\n❌ Error processing {json_file.name}: {str(e)}")
            print("   Moving to the next file. This file's progress is saved and will be resumed on the next run.")
            continue
    
    # Save a summary of the processing results
    summary_path = output_dir / "processing_summary.json"
    print(f"\nProcessing complete. Saving summary to {summary_path}")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

# Run main if this script is executed directly
if __name__ == "__main__":
    main()