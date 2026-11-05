## Setup
 
### 1. Install Tesseract
 
Install Tesseract OCR and make sure it is on your system PATH.
 
- **Windows:** use the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki), then add the install folder (usually `C:\Program Files\Tesseract-OCR`) to PATH.
- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`
 
### 2. Install Python packages
 
```bash
pip install -r requirements.txt
```
 
### 3. Configure
 
Edit `config.txt`:
 
```
image_dir = ./images
n_workers = 2
```
 
### Single run
 
```bash
python main.py
```
 
Runs the pipeline once using the settings in `config.txt`. Results are saved to `ocr_results.csv`, and the total time, writer busy time, and average accuracy are printed.
 
### Benchmark
 
```bash
python benchmark.py
```
 
Runs the pipeline with 1, 2, 4, and 8 workers, repeated several times (set `N_RUNS` at the top of the file), after one untimed warm-up run. It prints the average time and speedup for each worker count and saves a graph to `benchmark.png`.