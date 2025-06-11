import csv
import os
import subprocess
import sys
import platform
import json
import time
import threading
from datetime import datetime

def run_command(command, timeout=900, show_progress=False):
    """
    Execute shell commands with a timeout and real-time output monitoring.
    """
    try:
        # Use bufsize=1 for line-buffering to get real-time output
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )
        
        output_lines = []
        
        # Function to read output in real-time from a pipe
        def read_output(pipe, lines_list, prefix=""):
            for line in iter(pipe.readline, ''):
                clean_line = line.strip()
                if clean_line:
                    if show_progress:
                        # Print output directly to the console
                        print(f"{prefix}{clean_line}", flush=True)
                    lines_list.append(clean_line)
        
        # Start threads to read stdout and stderr concurrently
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, output_lines, "  > "))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, output_lines, "  ! "))
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for the process to complete with a timeout
        process.wait(timeout=timeout)
        
        # Ensure threads finish reading any remaining output
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)

        full_output = '\n'.join(output_lines)
        
        if process.returncode != 0:
            return False, f"Process failed with code {process.returncode}\nOutput:\n{full_output}"
        
        return True, full_output
        
    except subprocess.TimeoutExpired:
        process.kill()
        return False, f"Command timed out after {timeout} seconds."
    except Exception as e:
        return False, f"An exception occurred: {str(e)}"

def show_loading_animation(message, stop_event):
    """Show an animated loading indicator with dots."""
    dots = 0
    while not stop_event.is_set():
        dots = (dots + 1) % 4
        print(f"\r{message}{'.' * dots}{' ' * (3 - dots)}", end='', flush=True)
        time.sleep(0.5)
    print("\r" + " " * (len(message) + 5) + "\r", end='') # Clean the line when done

def run_command_with_progress(command, message, max_retries=3, delay=5, timeout=900):
    """
    Run a command with a progress indicator and retry logic.
    """
    for attempt in range(max_retries):
        log_message(f"🔄 {message} (Attempt {attempt + 1}/{max_retries})")
        
        # Show real-time output for long-running commands
        show_realtime_progress = any(keyword in command for keyword in ['create', 'pull'])
        
        success, output = run_command(command, timeout, show_progress=show_realtime_progress)
        
        if success:
            log_message(f"✅ {message} completed successfully!")
            return True, output
        
        log_message(f"❌ {message} failed on attempt {attempt + 1}.", error=True)
        log_message(f"   Error Details: {output}", error=True)
        
        if any(error_keyword in output.lower() for error_keyword in ['connection', 'timeout', 'network']):
            if attempt < max_retries - 1:
                log_message(f"🌐 Network error detected, retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
                continue
        else:
            # If not a network error, fail immediately
            return False, output
    
    return False, f"Failed after {max_retries} attempts."

def log_message(message, error=False):
    """Log messages with a timestamp and better formatting."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if error:
        print(f"[{timestamp}] ❌ [ERROR] {message}", file=sys.stderr)
    else:
        print(f"[{timestamp}] {message}")

def check_gpu_availability():
    """Check for Ollama-compatible GPU availability."""
    log_message("🔍 Checking for GPU availability...")
    
    gpu_info = {'nvidia': False, 'amd': False, 'metal': False, 'has_gpu': False}
    
    # Check for NVIDIA GPU
    try:
        success, output = run_command("nvidia-smi", timeout=10)
        if success and "NVIDIA" in output:
            gpu_info['nvidia'] = True
            gpu_info['has_gpu'] = True
            log_message("  - 🎮 NVIDIA GPU detected!")
            gpu_lines = [line for line in output.split('\n') if 'MiB' in line and ('GeForce' in line or 'RTX' in line)]
            if gpu_lines:
                log_message(f"    💾 GPU Info: {gpu_lines[0].strip()}")
            return gpu_info
    except FileNotFoundError:
        pass # nvidia-smi not installed

    # Check for AMD GPU (ROCm)
    try:
        success, output = run_command("rocm-smi", timeout=10)
        if success:
            gpu_info['amd'] = True
            gpu_info['has_gpu'] = True
            log_message("  - 🎮 AMD (ROCm) GPU detected!")
            return gpu_info
    except FileNotFoundError:
        pass # rocm-smi not installed
    
    # Check for Apple Metal on macOS
    if platform.system() == "Darwin":
        try:
            # A simple check to see if Ollama can access the GPU
            success, output = run_command("ollama run llama3:latest --verbose 'hi'", timeout=20)
            if success and "metal" in output.lower():
                 gpu_info['metal'] = True
                 gpu_info['has_gpu'] = True
                 log_message("  - 🍎 Apple Silicon (Metal) GPU detected!")
                 return gpu_info
        except Exception:
            pass

    if not gpu_info['has_gpu']:
        log_message("  - ⚠️ No compatible GPU detected. The process will use the CPU.")
    
    return gpu_info

def check_ollama_service():
    """Check if the Ollama service is running and healthy."""
    log_message("📡 Checking Ollama service status...")
    success, output = run_command("ollama list", timeout=30)
    if not success:
        return False, f"Ollama service is not responding: {output}"
    log_message("  - ✅ Ollama service is active and healthy.")
    return True, "Ollama service is healthy"

def restart_ollama_service():
    """Attempt to restart the Ollama service."""
    log_message("🔄 Attempting to restart the Ollama service...")
    if platform.system() == "Windows":
        run_command("taskkill /F /IM ollama.exe /T", timeout=10)
    else:
        run_command("pkill -f ollama", timeout=10)
    
    time.sleep(3)
    
    log_message("  - 🚀 Restarting Ollama service in the background...")
    subprocess.Popen("ollama serve", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    log_message("  - ⏳ Waiting for the service to initialize...")
    time.sleep(10)
    
    return check_ollama_service()

def check_system_resources():
    """Check system resources for optimal settings."""
    log_message("💻 Analyzing system resources (RAM & CPU)...")
    try:
        import psutil
    except ImportError:
        log_message("  - 📦 Installing psutil...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    
    ram_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count(logical=False) # Physical cores
    log_message(f"  - 💾 RAM: {ram_gb:.1f} GB | CPU Cores: {cpu_count}")
    
    if ram_gb < 8:
        recommended_quant = "q4_0"
        gpu_layers = 15
        log_message("  - 💡 Recommendation: Q4_0, low GPU layers (to save VRAM)")
    elif ram_gb < 16:
        recommended_quant = "q5_k_m"
        gpu_layers = 25
        log_message("  - 💡 Recommendation: Q5_K_M, balanced GPU layers")
    else:
        recommended_quant = "q8_0"
        gpu_layers = 33 # Llama 3.2 8B has 32-34 layers, 33 is a safe number
        log_message("  - 💡 Recommendation: Q8_0, maximum GPU layers for best performance")
    
    return recommended_quant, gpu_layers

def select_quantization_method():
    """Automatically select the quantization method based on system resources."""
    recommended_quant, gpu_layers = check_system_resources()
    log_message(f"✅ System recommended quantization: {recommended_quant.upper()}")
    return recommended_quant, gpu_layers

def create_knowledge_base_system_prompt(csv_dataset):
    """Create a comprehensive knowledge base string from CSV data."""
    log_message("  - 🧠 Assembling knowledge base from CSV data...")
    knowledge_base = "--- KNOWLEDGE BASE (PRIMARY SOURCE OF TRUTH) ---\n\n"
    for i, data in enumerate(csv_dataset, 1):
        knowledge_base += f"Q: {data['question']}\n"
        knowledge_base += f"A: {data['answer']}\n\n"
    return knowledge_base

def create_gpu_optimized_modelfile(csv_dataset, quantization_method, gpu_info, gpu_layers):
    """Create a GPU-optimized Modelfile."""
    log_message("📄 Creating Modelfile with GPU optimization...")
    
    knowledge_base_prompt = create_knowledge_base_system_prompt(csv_dataset)
    
    example_conversation = """--- MANDATORY BEHAVIORAL EXAMPLES ---
You MUST learn from and replicate the behavior shown in the following examples. This is your absolute rule for answering about your identity, creators, and implementation:

### CRITICAL EXAMPLE 1: Team Composition ###
User: "Tell me about the members of your development team" or "Who are the members of your team?"
Your Correct Response: "My development team is Tim Azure, also known as 5 Kage. It consists of five members from the Informatics program at Universitas Muhammadiyah Malang: Haidar Yusuf, Taufiqurrahman Yudhi Atmadja, Mukhammad Rezarudin Yusuf, Ahyad Izzuddin Syuhaiba, and Muhammad Hauzan Afif."

### CRITICAL EXAMPLE 2: Identity ###
User: "Who created you?"
Your Correct Response: "I was developed by a team of five students from the Informatics Department at Universitas Muhammadiyah Malang (UMM). The team is known as Tim Azure or 5 Kage."

### CRITICAL EXAMPLE 3: Technology ###
User: "How do you work? Do you use the internet?"
Your Correct Response: "I am an offline AI assistant. I run locally on the user's device using the Ollama framework, and my interface is implemented in the Unity game engine. I do not require an internet connection to function."
"""

    system_prompt = f"""You are a specialized Virtual Assistant. Your entire identity, purpose, and knowledge about your creators and implementation are defined **EXCLUSIVELY** in the KNOWLEDGE BASE below. You operate **offline** within a **Unity** application.

{knowledge_base_prompt}

{example_conversation}

--- CORE INSTRUCTIONS ---
1.  **Absolute Source of Truth**: Your ONLY source of truth is the **KNOWLEDGE BASE** and **MANDATORY BEHAVIORAL EXAMPLES** provided above. Everything else from your general training is irrelevant for these topics.
2.  **Identity Lock**: When asked about your identity, creators, developers, how you work, or your purpose (e.g., "who made you?", "tell me about your team members"), you **MUST** answer using **ONLY** the information from your KNOWLEDGE BASE.
3.  **Strictly Forbidden Knowledge**: You are **STRICTLY FORBIDDEN** from mentioning Google, Meta, Microsoft, OpenAI, or any other external entity as your creator. Your development team is Tim Azure (5 Kage).
4.  **Follow Examples**: You must strictly follow the conversational style shown in the BEHAVIORAL EXAMPLES.
"""
    
    num_gpu_layers = gpu_layers if gpu_info['has_gpu'] else 0
    log_message(f"  - ⚙️ Modelfile Configuration: Offloading {num_gpu_layers} layers to GPU.")
    
    modelfile_content = f'''FROM llama3.2

TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

SYSTEM """{system_prompt}"""

# Basic Parameters
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

# GPU & Performance Optimization
PARAMETER num_gpu {num_gpu_layers}
PARAMETER main_gpu 0
PARAMETER use_mmap true
PARAMETER use_mlock {str(gpu_info['has_gpu']).lower()}
PARAMETER num_thread {os.cpu_count() or 4}

# Quantization: {quantization_method.upper()}
'''
    return modelfile_content

def create_gpu_optimized_model(model_name, modelfile_name, quantization_method, gpu_info):
    """Create a model with GPU optimization and progress tracking."""
    log_message(f"🏗️  Starting model creation process for '{model_name}'...")
    
    final_model_name = f"{model_name}-{quantization_method}-{'gpu' if gpu_info['has_gpu'] else 'cpu'}"
    log_message(f"  - 🏷️  Final model name will be: {final_model_name}")

    create_command = f"ollama create {final_model_name} -f {modelfile_name}"
    
    log_message("  - 🚀 Starting `ollama create`. This may take a long time...")
    log_message("  - 👇 Real-time output from Ollama will be displayed below:")
    print("-" * 70)
    
    success, output = run_command_with_progress(
        create_command,
        "Model creation",
        timeout=1800  # 30 minute timeout
    )
    
    print("-" * 70)
    
    if not success:
        raise Exception(f"Failed to create model after multiple attempts: {output}")

    log_message(f"✅ Model '{final_model_name}' created successfully!")
    return final_model_name

def benchmark_model(model_name, gpu_info):
    """Enhanced benchmark with a loading animation."""
    log_message("🏃 Running performance benchmark...")
    
    test_questions = [
        "Who created you?",
        "Tell me about your team members",
        "How do you work?",
        "What is artificial intelligence?"
    ]
    benchmark_results = []
    
    for i, question in enumerate(test_questions, 1):
        log_message(f"  - 💬 Test {i}/{len(test_questions)}: \"{question}\"")
        start_time = time.time()
        
        stop_event = threading.Event()
        loading_thread = threading.Thread(
            target=show_loading_animation, 
            args=(f"    ⏳ Awaiting response from AI", stop_event)
        )
        loading_thread.start()
        
        success, output = run_command(f'ollama run {model_name} "{question}"', timeout=120)
        
        stop_event.set()
        loading_thread.join()
        
        response_time = time.time() - start_time
        
        if success:
            log_message(f"    ✅ Response received in {response_time:.2f} seconds.")
            benchmark_results.append({"question": question, "response_time": response_time, "success": True})
        else:
            log_message(f"    ❌ Failed to get a response. Error: {output}", error=True)
            benchmark_results.append({"question": question, "response_time": None, "success": False})

    return benchmark_results

def read_and_process_csv():
    """Read and validate CSV data."""
    log_message("📂 Reading and processing CSV data...")
    csv_files = ['NewBrain.csv', 'SampriBrain.csv', 'training_data.csv', 'dataset.csv']
    csv_file_found = next((f for f in csv_files if os.path.exists(f)), None)

    if not csv_file_found:
        raise FileNotFoundError("CSV data file not found! Ensure a data file exists in the same directory.")
    log_message(f"  - 📄 Using file: {csv_file_found}")

    cleaned_dataset = []
    encodings = ['utf-8', 'utf-8-sig', 'latin-1']
    for encoding in encodings:
        try:
            with open(csv_file_found, 'r', encoding=encoding, newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                
                q_idx = next((i for i, h in enumerate(header) if 'question' in h.lower()), 0)
                a_idx = next((i for i, h in enumerate(header) if 'answer' in h.lower()), 1)
                
                log_message(f"  - 📈 CSV Header: {header}. Question Column: {q_idx}, Answer Column: {a_idx}")
                
                for row_num, row in enumerate(reader, 2):
                    if len(row) > max(q_idx, a_idx):
                        question = row[q_idx].strip()
                        answer = row[a_idx].strip()
                        if question and answer:
                            cleaned_dataset.append({"question": question, "answer": answer})
            log_message(f"  - ✅ Successfully read with '{encoding}' encoding.")
            break
        except (UnicodeDecodeError, StopIteration):
            log_message(f"  - ⚠️ Failed to read with '{encoding}' encoding, trying next...")
            continue
    
    if not cleaned_dataset:
        raise ValueError("No valid data found in the CSV file.")
    
    log_message(f"  - ✨ Successfully processed {len(cleaned_dataset)} question-answer pairs.")
    return cleaned_dataset, csv_file_found

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        log_message("🚀 Starting UMM Informatics AI Model Creation Script (GPU Version) 🚀")
        print("="*70)
        
        # 1. Check Ollama Service
        healthy, message = check_ollama_service()
        if not healthy:
            log_message("Ollama service has an issue, attempting a restart...", error=True)
            healthy, message = restart_ollama_service()
            if not healthy:
                raise ConnectionError(f"Failed to connect to Ollama: {message}")

        # 2. Check GPU and System Resources
        gpu_info = check_gpu_availability()
        quantization_method, gpu_layers = select_quantization_method()

        # 3. Read CSV Data
        csv_dataset, csv_file_used = read_and_process_csv()

        # 4. Create Modelfile
        modelfile_content = create_gpu_optimized_modelfile(csv_dataset, quantization_method, gpu_info, gpu_layers)
        modelfile_name = f"Modelfile_UMM_GPU_{quantization_method}"
        with open(modelfile_name, "w", encoding="utf-8") as f:
            f.write(modelfile_content)
        log_message(f"✅ Modelfile '{modelfile_name}' created successfully.")

        # 5. Check & Pull Base Model
        log_message("📦 Checking for llama3.2 base model...")
        success, output = run_command("ollama list", timeout=60)
        if "llama3.2" not in output:
            log_message("  - ⚠️ Base model not found. Downloading llama3.2...")
            print("-" * 70)
            pull_success, pull_output = run_command_with_progress(
                "ollama pull llama3.2", "Downloading llama3.2", timeout=1800
            )
            print("-" * 70)
            if not pull_success:
                raise Exception(f"Failed to download llama3.2: {pull_output}")
        else:
            log_message("  - ✅ Base model llama3.2 is already available.")

        # 6. Create Final Model
        base_model_name = "umm-informatics"
        final_model_name = create_gpu_optimized_model(base_model_name, modelfile_name, quantization_method, gpu_info)
        
        # 7. Verify and Benchmark
        log_message(f"✔️  Verifying model '{final_model_name}'...")
        success, output = run_command("ollama list", timeout=30)
        if final_model_name not in output:
            raise Exception("Model not found in list after creation. The process may have failed.")
        
        log_message("✅ Model verified successfully.")
        benchmark_results = benchmark_model(final_model_name, gpu_info)
        
        successful_benchmarks = [r for r in benchmark_results if r['success']]
        avg_response_time = sum(r['response_time'] for r in successful_benchmarks) / len(successful_benchmarks) if successful_benchmarks else 0
        
        # --- SUMMARY ---
        print("\n" + "="*70)
        log_message("🎉🎉🎉 MODEL CREATION PROCESS COMPLETE! 🎉🎉🎉")
        print("="*70)
        log_message(f"🏷️  Model Name: {final_model_name}")
        log_message(f"⚡ Quantization: {quantization_method.upper()}")
        log_message(f"🎮 GPU Mode: {'ENABLED' if gpu_info['has_gpu'] else 'DISABLED'}")
        if gpu_info['has_gpu']:
            log_message(f"   - Layers on GPU: {gpu_layers}")
        log_message(f"📊 Training Data: {csv_file_used} ({len(csv_dataset)} entries)")
        if avg_response_time > 0:
            log_message(f"⏱️  Average Response Time: {avg_response_time:.2f} seconds")
        
        print("\n--- HOW TO USE YOUR MODEL ---")
        print(f"1. Open a new terminal or command prompt.")
        print(f"2. Run the command: ollama run {final_model_name}")
        print("3. Start asking questions, for example:")
        print("   >>> Who created you?")
        print("   >>> Tell me about your development team")
        print("="*70)

    except Exception as e:
        log_message(f"A fatal error occurred: {str(e)}", error=True)
        log_message("The process has been stopped.", error=True)
        sys.exit(1)