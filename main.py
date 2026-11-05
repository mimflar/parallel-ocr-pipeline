import os
import csv
import time
import multiprocessing as mp
import pytesseract

os.environ["OMP_THREAD_LIMIT"] = "1"

def load_config(config_path="config.txt"):
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                config[key.strip()] = value.strip()
    return config['image_dir'], int(config['n_workers'])

def loader_worker(image_dir, input_queue, n_workers):
    # map labels to a dictionary for O(1) lookups
    labels = {}
    labels_path = os.path.join(image_dir, "labels.txt")
    if os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    labels[parts[0].strip()] = parts[1].strip()

    current_id = 1
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            filepath = os.path.join(image_dir, filename)
            expected_text = labels.get(filename, "")
            
            # append the expected text to the queue payload
            input_queue.put((current_id, filename, filepath, expected_text))
            current_id += 1
            
    for _ in range(n_workers):
        input_queue.put(None)

"""measure the difference between two strings"""
def calculate_levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_accuracy(expected, extracted):
    if not expected and not extracted:
        return 100.0
    if not expected or not extracted:
        return 0.0
        
    distance = calculate_levenshtein_distance(expected, extracted)
    max_len = max(len(expected), len(extracted))
    
    return ((max_len - distance) / max_len) * 100

def ocr_worker(input_queue, output_queue):
    # this config works on most photos
    custom_config = r'--tessdata-dir ./tessdata --oem 1 --psm 8 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' 
    
    while True:
        task = input_queue.get()
        if task is None:
            break 
            
        task_id, filename, filepath, expected_text = task
        start_time = time.time()
        
        extracted_text = pytesseract.image_to_string(filepath, config=custom_config).strip()
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # calculate character recognition percentage
        accuracy = calculate_accuracy(expected_text, extracted_text)
        
        output_queue.put((task_id, filename, expected_text, extracted_text, f"{accuracy:.2f}%", processing_time_ms))

def writer_worker(output_queue, output_file="ocr_results.csv"):
    busy = 0.0
    total_accuracy = 0.0
    count = 0

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Filename', 'Expected Text', 'Extracted Text', 'Accuracy', 'Processing Time (ms)'])

        while True:
            result = output_queue.get()
            if result is None:
                break
            t = time.perf_counter()
            writer.writerow(result)
            busy += time.perf_counter() - t
            total_accuracy += float(result[4].rstrip('%'))
            count += 1

    avg_accuracy = total_accuracy / count if count else 0.0
    print(f"Writer busy time: {busy * 1000:.2f} ms")
    print(f"Average accuracy: {avg_accuracy:.2f}% over {count} images")

def run_pipeline(image_dir, n_workers, output_file="ocr_results.csv"):
    input_queue = mp.Queue(maxsize=100) 
    output_queue = mp.Queue(maxsize=100)
    
    loader = mp.Process(target=loader_worker, args=(image_dir, input_queue, n_workers))
    workers = [mp.Process(target=ocr_worker, args=(input_queue, output_queue)) for _ in range(n_workers)]
    writer = mp.Process(target=writer_worker, args=(output_queue, output_file))
    
    start_time = time.time()
    
    writer.start()
    for w in workers:
        w.start()
    loader.start()
    
    loader.join()
    for w in workers:
        w.join()
        
    output_queue.put(None)
    writer.join()
    
    return time.time() - start_time

if __name__ == '__main__':
    # load configuration from config.txt
    image_dir, n_workers = load_config("config.txt")
    
    print(f"Starting parallel OCR Pipeline on directory: {image_dir}")
    print(f"Configured for {n_workers} OCR worker(s)...")
    
    # execute exactly once using the required filename
    elapsed = run_pipeline(image_dir, n_workers, output_file="ocr_results.csv")
    
    print(f"Pipeline completed in {elapsed:.2f} seconds.")
    print("Results successfully saved to ocr_results.csv.")