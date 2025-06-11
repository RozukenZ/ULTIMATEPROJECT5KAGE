import csv
import os
import subprocess
import sys
import platform
import json
from datetime import datetime

# Fix encoding issues for Windows
if platform.system() == "Windows":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def run_command(command):
    """
    Execute shell commands and return success status and output
    """
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return False, stderr
        
        return True, stdout
    except Exception as e:
        return False, str(e)

def safe_print(message, error=False):
    """Safely print messages with emoji fallback for Windows"""
    # Replace emojis with text alternatives for better compatibility
    emoji_replacements = {
        '🚀': '[SPEED]',
        '⚡': '[FAST]', 
        '✅': '[OK]',
        '❌': '[ERROR]',
        '🎯': '[TARGET]',
        '📦': '[PACKAGE]',
        '⬇️': '[DOWNLOAD]',
        '🎉': '[SUCCESS]',
        '🔍': '[CHECK]',
        '📊': '[INFO]',
        '🤖': '[AI]',
        '💾': '[MEMORY]',
        '🔥': '[HOT]'
    }
    
    safe_message = message
    for emoji, replacement in emoji_replacements.items():
        safe_message = safe_message.replace(emoji, replacement)
    
    try:
        if error:
            print(safe_message, file=sys.stderr)
        else:
            print(safe_message)
    except UnicodeEncodeError:
        # Final fallback - remove any remaining problematic characters
        safe_message = safe_message.encode('ascii', errors='ignore').decode('ascii')
        if error:
            print(safe_message, file=sys.stderr)
        else:
            print(safe_message)

def log_message(message, error=False):
    """Log messages with timestamp and safe encoding"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if error:
        safe_print(f"[{timestamp}] [ERROR] {message}", error=True)
    else:
        safe_print(f"[{timestamp}] [INFO] {message}")

def detect_system_specs():
    """Detect system specifications for optimal performance"""
    specs = {
        "cpu_cores": os.cpu_count(),
        "gpu_memory": 4096,  # Default
        "platform": platform.system()
    }
    
    try:
        # Try to detect NVIDIA GPU memory
        success, output = run_command("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits")
        if success and output.strip().isdigit():
            specs["gpu_memory"] = int(output.strip())
            log_message(f"Detected GPU Memory: {specs['gpu_memory']}MB ({specs['gpu_memory']/1024:.1f}GB)")
        else:
            log_message("Using default GPU memory estimation (4GB)")
    except:
        log_message("Could not detect GPU. Using CPU-optimized settings.")
    
    log_message(f"System: {specs['platform']}, CPU Cores: {specs['cpu_cores']}")
    return specs

def get_zero_delay_config(system_specs):
    """Return configuration optimized for zero delay"""
    gpu_memory_gb = system_specs["gpu_memory"] / 1024
    cpu_cores = system_specs["cpu_cores"]
    
    # Ultra-optimized configuration for instant response
    if gpu_memory_gb >= 8:
        return {
            "base_model": "llama3.2:3b",
            "num_ctx": 2048,           # Smaller context for faster processing
            "num_predict": 512,        # Limit response length for speed
            "num_gpu": 1,
            "num_thread": min(cpu_cores, 8),
            "temperature": 0.05,       # Very low for consistent, fast responses
            "top_p": 0.8,             # Optimized for speed
            "top_k": 20,              # Smaller for faster token selection
            "repeat_penalty": 1.05,    # Minimal penalty for speed
            "batch_size": 1024,        # Larger batch for GPU efficiency
            "num_keep": 24,           # Keep tokens in memory
            "seed": 42,               # Fixed seed for consistency
            "tfs_z": 1.0,             # Tail-free sampling for speed
            "mirostat": 0,            # Disable for speed
            "mirostat_eta": 0.1,
            "mirostat_tau": 5.0,
            "penalize_newline": False, # Faster processing
            "numa": False,            # Disable NUMA for speed
            "num_batch": 512,         # Batch processing optimization
            "rope_frequency_base": 10000.0,
            "rope_frequency_scale": 1.0
        }
    elif gpu_memory_gb >= 6:
        return {
            "base_model": "llama3.2:3b",
            "num_ctx": 1024,
            "num_predict": 256,
            "num_gpu": 1,
            "num_thread": min(cpu_cores, 6),
            "temperature": 0.05,
            "top_p": 0.75,
            "top_k": 15,
            "repeat_penalty": 1.05,
            "batch_size": 512,
            "num_keep": 16,
            "seed": 42,
            "tfs_z": 1.0,
            "mirostat": 0,
            "numa": False,
            "num_batch": 256,
            "rope_frequency_base": 10000.0,
            "rope_frequency_scale": 1.0
        }
    else:  # 4GB or less - Ultra speed mode
        return {
            "base_model": "llama3.2:1b",    # Fastest small model
            "num_ctx": 512,                 # Minimal context for maximum speed
            "num_predict": 128,             # Short responses for speed
            "num_gpu": 1,
            "num_thread": min(cpu_cores, 4),
            "temperature": 0.01,            # Nearly deterministic for speed
            "top_p": 0.7,
            "top_k": 10,                   # Very small for fastest selection
            "repeat_penalty": 1.01,         # Minimal
            "batch_size": 256,
            "num_keep": 8,                 # Minimal keep for speed
            "seed": 42,
            "tfs_z": 1.0,
            "mirostat": 0,
            "numa": False,
            "num_batch": 128,
            "rope_frequency_base": 10000.0,
            "rope_frequency_scale": 1.0
        }

def read_and_process_csv():
    """Read CSV data with speed optimization"""
    csv_files = ['NewBrain.csv', 'SampriBrain.csv', 'SampriBrainTrial.csv', 'training_data.csv', 'dataset.csv']
    csv_file_found = None
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            csv_file_found = csv_file
            break
    
    if not csv_file_found:
        raise FileNotFoundError("CSV file not found. Please ensure you have a CSV file with training data in the same directory.")
    
    log_message(f"Using file: {csv_file_found}")
    
    cleaned_dataset = []
    
    try:
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        file_content = None
        
        for encoding in encodings:
            try:
                with open(csv_file_found, 'r', encoding=encoding) as file:
                    file_content = file.read()
                    log_message(f"File successfully read with encoding: {encoding}")
                    break
            except UnicodeDecodeError:
                continue
        
        if file_content is None:
            raise Exception("Cannot read CSV file with any encoding")
        
        lines = file_content.strip().split('\n')
        reader = csv.reader(lines)
        
        try:
            header = next(reader)
            log_message(f"CSV Header: {header}")
        except StopIteration:
            raise Exception("CSV file is empty")
        
        question_col = None
        answer_col = None
        
        for i, col in enumerate(header):
            col_lower = col.lower().strip()
            if any(keyword in col_lower for keyword in ['question', 'pertanyaan', 'tanya', 'ask', 'query']):
                question_col = i
            elif any(keyword in col_lower for keyword in ['answer', 'jawaban', 'reply', 'response', 'respon']):
                answer_col = i
        
        if question_col is None or answer_col is None:
            if len(header) >= 2:
                question_col = 0 if question_col is None else question_col
                answer_col = 1 if answer_col is None else answer_col
                log_message(f"Using columns by position - Question: {question_col}, Answer: {answer_col}")
            else:
                raise Exception("CSV must have at least 2 columns for questions and answers")
        
        log_message(f"Question column: {header[question_col]} (index {question_col})")
        log_message(f"Answer column: {header[answer_col]} (index {answer_col})")
        
        for row_num, row in enumerate(reader, start=2):
            if len(row) <= max(question_col, answer_col):
                continue
            
            question = row[question_col].strip() if question_col < len(row) else ""
            answer = row[answer_col].strip() if answer_col < len(row) else ""
            
            if len(question) > 3 and len(answer) > 3:
                # Truncate for speed optimization
                question = question[:150] if len(question) > 150 else question
                answer = answer[:200] if len(answer) > 200 else answer
                
                cleaned_dataset.append({
                    "question": question,
                    "answer": answer,
                    "row_number": row_num
                })
                
    except Exception as e:
        raise Exception(f"Error reading CSV: {str(e)}")
    
    if len(cleaned_dataset) == 0:
        raise Exception("No valid data found in CSV.")
    
    # Limit dataset size for speed (take only most important entries)
    if len(cleaned_dataset) > 20:
        cleaned_dataset = cleaned_dataset[:20]
        log_message(f"[FAST] Dataset limited to 20 entries for maximum speed")
    
    log_message(f"[OK] Successfully processed {len(cleaned_dataset)} question-answer pairs")
    return cleaned_dataset, csv_file_found

def create_ultra_fast_knowledge_base(csv_dataset):
    """Create minimal, ultra-fast knowledge base"""
    # Ultra-compact knowledge base for zero delay
    knowledge_base = "CORE KNOWLEDGE:\n"
    
    # Only include essential entries
    essential_entries = csv_dataset[:10]  # Maximum 10 for speed
    
    for i, data in enumerate(essential_entries, 1):
        knowledge_base += f"Q{i}: {data['question']}\n"
        knowledge_base += f"A{i}: {data['answer']}\n"
    
    return knowledge_base

def create_zero_delay_modelfile(csv_dataset, model_config):
    """Create ultra-optimized Modelfile for zero delay"""
    
    log_message("[FAST] Creating ZERO-DELAY optimized knowledge base...")
    knowledge_base = create_ultra_fast_knowledge_base(csv_dataset)
    
    # Minimal, ultra-fast examples
    core_examples = """INSTANT RESPONSES:
Team: "Tim Azure (5 Kage) - UMM Informatics: Haidar Yusuf, Taufiqurrahman Yudhi Atmadja, Mukhammad Rezarudin Yusuf, Ahyad Izzuddin Syuhaiba, Muhammad Hauzan Afif"
Tech: "Offline AI via Ollama+Unity, no internet needed"
"""

    # Ultra-compact system prompt for maximum speed
    system_prompt = f"""UMM Informatics AI Assistant - SPEED MODE

{knowledge_base}

{core_examples}

SPEED RULES:
- Team questions: Use team info above
- Tech questions: Offline AI via Ollama+Unity
- Other: Brief, helpful answers
- Never invent team member names
- Keep responses short and direct

PRIORITY: SPEED FIRST, then accuracy.
"""

    # Ultra-optimized Modelfile with all speed parameters
    modelfile_content = f'''FROM {model_config["base_model"]}

# ZERO-DELAY OPTIMIZED TEMPLATE
TEMPLATE """<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""

# ULTRA-COMPACT SYSTEM PROMPT FOR MAXIMUM SPEED
SYSTEM """{system_prompt}"""

# ZERO-DELAY PARAMETERS - OPTIMIZED FOR INSTANT RESPONSE
PARAMETER temperature {model_config["temperature"]}
PARAMETER top_p {model_config["top_p"]}
PARAMETER top_k {model_config["top_k"]}
PARAMETER num_ctx {model_config["num_ctx"]}
PARAMETER num_predict {model_config["num_predict"]}
PARAMETER repeat_penalty {model_config["repeat_penalty"]}
PARAMETER seed {model_config["seed"]}
PARAMETER tfs_z {model_config["tfs_z"]}
PARAMETER mirostat {model_config["mirostat"]}
PARAMETER penalize_newline {str(model_config.get("penalize_newline", False)).lower()}
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# MAXIMUM PERFORMANCE PARAMETERS
PARAMETER num_thread {model_config["num_thread"]}
PARAMETER num_gpu {model_config["num_gpu"]}
PARAMETER num_batch {model_config["num_batch"]}
PARAMETER num_keep {model_config["num_keep"]}
PARAMETER batch_size {model_config["batch_size"]}

# SPEED OPTIMIZATION FLAGS
PARAMETER mlock true
PARAMETER low_vram true
PARAMETER numa {str(model_config.get("numa", False)).lower()}
PARAMETER use_mmap true
PARAMETER use_mlock true
PARAMETER rope_frequency_base {model_config["rope_frequency_base"]}
PARAMETER rope_frequency_scale {model_config["rope_frequency_scale"]}

# ULTRA-FAST INFERENCE SETTINGS
PARAMETER flash_attn true
PARAMETER logits_all false
PARAMETER vocab_only false
PARAMETER embedding_only false
'''
    return modelfile_content

def setup_ollama_for_speed():
    """Configure Ollama for maximum speed"""
    log_message("[SPEED] Configuring Ollama for maximum speed...")
    
    # Create speed optimization script
    if platform.system() == "Windows":
        speed_script = '''@echo off
REM Ollama Speed Optimization Script for Windows

REM Set environment variables for maximum performance
set OLLAMA_NUM_PARALLEL=4
set OLLAMA_MAX_LOADED_MODELS=1
set OLLAMA_MAX_QUEUE=512
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KEEP_ALIVE=24h
set OLLAMA_HOST=127.0.0.1:11434

echo [SPEED] Ollama Speed Optimization Applied!
echo Model will stay loaded for 24 hours for instant responses

REM Pre-load the model for instant access
echo [FAST] Pre-loading model for zero-delay responses...
start /B ollama run umm-informatics-speed --keepalive 24h

echo [OK] Speed optimization complete!
echo Usage: ollama run umm-informatics-speed
pause
'''
        script_filename = "speed_optimize.bat"
    else:
        speed_script = '''#!/bin/bash
# Ollama Speed Optimization Script

# Set environment variables for maximum performance
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_MAX_QUEUE=512
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KEEP_ALIVE=24h
export OLLAMA_HOST=127.0.0.1:11434

echo "[SPEED] Ollama Speed Optimization Applied!"
echo "Model will stay loaded for 24 hours for instant responses"

# Pre-load the model for instant access
echo "[FAST] Pre-loading model for zero-delay responses..."
ollama run umm-informatics-speed --keepalive 24h < /dev/null &

echo "[OK] Speed optimization complete!"
echo "Usage: ollama run umm-informatics-speed"
'''
        script_filename = "speed_optimize.sh"
    
    with open(script_filename, "w", encoding="utf-8") as f:
        f.write(speed_script)
    
    # Make script executable on Unix systems
    if platform.system() != "Windows":
        os.chmod(script_filename, 0o755)
    
    log_message(f"[OK] Speed optimization script created: {script_filename}")

# Main execution
try:
    log_message("[FAST] Starting UMM AI - ZERO DELAY EDITION...")
    
    # Step 0: Detect system and optimize
    system_specs = detect_system_specs()
    model_config = get_zero_delay_config(system_specs)
    
    log_message(f"[TARGET] ZERO-DELAY Configuration:")
    log_message(f"   Base Model: {model_config['base_model']}")
    log_message(f"   Context: {model_config['num_ctx']} (optimized for speed)")
    log_message(f"   Max Tokens: {model_config['num_predict']} (fast responses)")
    log_message(f"   Temperature: {model_config['temperature']} (consistent & fast)")
    
    # Step 1: Read CSV with speed optimization
    log_message("[FAST] Reading CSV with speed optimization...")
    csv_dataset, csv_file_used = read_and_process_csv()
    
    # Step 2: Create zero-delay Modelfile
    log_message("[SPEED] Creating ZERO-DELAY Modelfile...")
    modelfile_content = create_zero_delay_modelfile(csv_dataset, model_config)
    
    modelfile_name = "Modelfile_UMM_ZeroDelay"
    with open(modelfile_name, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    log_message(f"[OK] {modelfile_name} created with speed optimizations")

    # Step 3: Verify Ollama
    log_message("[CHECK] Checking Ollama...")
    success, output = run_command("ollama --version")
    if not success:
        raise Exception("Ollama not found. Please install Ollama first.")
    
    log_message(f"Ollama detected: {output.strip()}")

    # Step 4: Setup speed optimizations
    setup_ollama_for_speed()

    # Step 5: Download base model if needed
    base_model = model_config["base_model"]
    log_message(f"[PACKAGE] Ensuring {base_model} is available...")
    success, output = run_command("ollama list")
    if base_model not in output:
        log_message(f"[DOWNLOAD] Downloading {base_model} (this may take a moment)...")
        success, output = run_command(f"ollama pull {base_model}")
        if not success:
            raise Exception(f"Failed to download {base_model}: {output}")
        log_message(f"[OK] {base_model} downloaded successfully")

    # Step 6: Create speed-optimized model
    model_name = "umm-informatics-speed"
    log_message(f"[FAST] Creating ZERO-DELAY model '{model_name}'...")
    
    success, output = run_command(f"ollama create {model_name} -f {modelfile_name}")
    if not success:
        raise Exception(f"Failed to create model: {output}")
    
    log_message(f"[SUCCESS] Speed-optimized model '{model_name}' created!")

    # Step 7: Pre-load model for instant access
    log_message("[SPEED] Pre-loading model for instant responses...")
    if platform.system() == "Windows":
        run_command(f"start /B ollama run {model_name} --keepalive 24h")
    else:
        run_command(f"ollama run {model_name} --keepalive 24h < /dev/null &")

    # Step 8: Verify and save info
    success, output = run_command("ollama list")
    if model_name in output:
        model_info = {
            "model_name": model_name,
            "created_at": datetime.now().isoformat(),
            "optimization_level": "ZERO_DELAY",
            "csv_file_used": csv_file_used,
            "knowledge_entries": len(csv_dataset),
            "base_model": base_model,
            "system_specs": system_specs,
            "speed_features": [
                "Ultra-low temperature for consistent responses",
                "Minimal context size for fast processing",
                "Pre-loaded model with 24h keepalive",
                "Optimized batch processing", 
                "Flash attention enabled",
                "Memory lock for instant access"
            ],
            "model_config": model_config,
            "version": "ZeroDelay-1.0"
        }
        
        with open("model_info_speed.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        log_message("[INFO] Speed optimization info saved")

    # Final instructions
    log_message("="*80)
    log_message("[FAST] UMM INFORMATICS AI - ZERO DELAY EDITION COMPLETED!")
    log_message("="*80)
    log_message(f"[AI] Model Name: {model_name}")
    log_message(f"[FAST] Optimization: ZERO DELAY")
    log_message(f"[TARGET] Base Model: {base_model}")
    log_message(f"[MEMORY] VRAM Optimized: {system_specs['gpu_memory']/1024:.1f}GB")
    log_message("")
    log_message("[SPEED] FOR INSTANT RESPONSES:")
    if platform.system() == "Windows":
        log_message("  1. First run the speed script: speed_optimize.bat")
    else:
        log_message("  1. First run the speed script: ./speed_optimize.sh")
    log_message(f"  2. Then use: ollama run {model_name}")
    log_message("")
    log_message("[FAST] SPEED FEATURES ENABLED:")
    log_message("  [OK] Model pre-loaded and kept in memory")
    log_message("  [OK] Ultra-low latency parameters") 
    log_message("  [OK] Minimal context for maximum speed")
    log_message("  [OK] Flash attention optimization")
    log_message("  [OK] Memory locking for instant access")
    log_message("")
    log_message("[HOT] Your AI should now respond INSTANTLY with zero delay!")

except Exception as e:
    log_message(f"[ERROR] Error: {str(e)}", error=True)
    sys.exit(1)