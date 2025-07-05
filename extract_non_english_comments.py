import json
import fasttext
import multiprocessing as mp
import os

MODEL_PATH = 'lid.176.bin'
INPUT_FILE = 'sampled_comments.json'
OUTPUT_FILE = 'non_english_comments.json'
CHUNK_SIZE = 10000  # Number of comments per chunk

# Load fastText model once per process
def load_model():
    return fasttext.load_model(MODEL_PATH)

def detect_non_english(comments):
    model = load_model()
    non_english = []
    for comment in comments:
        if not comment.strip():
            continue
        lang = model.predict(comment.replace('\n', ' '))[0][0].replace('__label__', '')
        if lang != 'en':
            non_english.append(comment)
    return non_english

def chunked_iterator(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        comments = json.load(f)

    pool = mp.Pool(mp.cpu_count())
    results = []
    for chunk in chunked_iterator(comments, CHUNK_SIZE):
        results.append(pool.apply_async(detect_non_english, (chunk,)))
    pool.close()
    pool.join()

    non_english_comments = []
    for r in results:
        non_english_comments.extend(r.get())

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(non_english_comments, f, ensure_ascii=False, indent=2)
    print(f"Extracted {len(non_english_comments)} non-English comments to {OUTPUT_FILE}")

if __name__ == '__main__':
    main() 