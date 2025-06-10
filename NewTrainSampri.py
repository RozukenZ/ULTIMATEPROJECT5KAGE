import csv
import os
import subprocess
import sys
import platform
import json
from datetime import datetime

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
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return False, stderr
        
        return True, stdout
    except Exception as e:
        return False, str(e)

def log_message(message, error=False):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if error:
        print(f"[{timestamp}] [ERROR] {message}", file=sys.stderr)
    else:
        print(f"[{timestamp}] [INFO] {message}")

def read_and_process_csv():
    """Read CSV data with better error handling and validation"""
    csv_files = ['NewBrain.csv', 'SampriBrain.csv', 'SampriBrainTrial.csv', 'training_data.csv', 'dataset.csv']
    csv_file_found = None
    
    # Find CSV file
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            csv_file_found = csv_file
            break
    
    if not csv_file_found:
        raise FileNotFoundError("CSV file not found. Please ensure you have a CSV file with training data in the same directory.")
    
    log_message(f"Using file: {csv_file_found}")
    
    cleaned_dataset = []
    
    try:
        # Try different encodings
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
        
        # Parse CSV content
        lines = file_content.strip().split('\n')
        reader = csv.reader(lines)
        
        try:
            header = next(reader)
            log_message(f"CSV Header: {header}")
        except StopIteration:
            raise Exception("CSV file is empty")
        
        # Identify question and answer columns
        question_col = None
        answer_col = None
        
        # Look for common column names
        for i, col in enumerate(header):
            col_lower = col.lower().strip()
            if any(keyword in col_lower for keyword in ['question', 'pertanyaan', 'tanya', 'ask', 'query']):
                question_col = i
            elif any(keyword in col_lower for keyword in ['answer', 'jawaban', 'reply', 'response', 'respon']):
                answer_col = i
        
        # If not found by name, assume first two columns
        if question_col is None or answer_col is None:
            if len(header) >= 2:
                question_col = 0 if question_col is None else question_col
                answer_col = 1 if answer_col is None else answer_col
                log_message(f"Using columns by position - Question: {question_col}, Answer: {answer_col}")
            else:
                raise Exception("CSV must have at least 2 columns for questions and answers")
        
        log_message(f"Question column: {header[question_col]} (index {question_col})")
        log_message(f"Answer column: {header[answer_col]} (index {answer_col})")
        
        # Process each row
        for row_num, row in enumerate(reader, start=2):
            if len(row) <= max(question_col, answer_col):
                log_message(f"Row {row_num} skipped: insufficient columns")
                continue
            
            question = row[question_col].strip() if question_col < len(row) else ""
            answer = row[answer_col].strip() if answer_col < len(row) else ""
            
            # Validate data quality
            if len(question) > 3 and len(answer) > 3:
                cleaned_dataset.append({
                    "question": question,
                    "answer": answer,
                    "row_number": row_num
                })
            else:
                log_message(f"⚠️ Row {row_num} skipped: question or answer too short")
                
    except Exception as e:
        raise Exception(f"Error reading CSV: {str(e)}")
    
    if len(cleaned_dataset) == 0:
        raise Exception("No valid data found in CSV. Ensure the CSV format is correct with question and answer columns.")
    
    log_message(f"✅ Successfully processed {len(cleaned_dataset)} question-answer pairs from CSV")
    return cleaned_dataset, csv_file_found

def create_knowledge_base_system_prompt(csv_dataset):
    """Create a comprehensive knowledge base string from CSV data"""
    knowledge_base = "KNOWLEDGE BASE FROM CSV DATA:\n\n"
    
    for i, data in enumerate(csv_dataset, 1):
        knowledge_base += f"Q{i}: {data['question']}\n"
        knowledge_base += f"A{i}: {data['answer']}\n\n"
    
    return knowledge_base

def create_enhanced_modelfile(csv_dataset):
    """
    Create an enhanced Modelfile with a powerful system prompt that includes
    the full knowledge base and hardcoded few-shot examples to force correct behavior.
    """
    
    log_message("🎨 Creating knowledge base for system prompt...")
    knowledge_base_prompt = create_knowledge_base_system_prompt(csv_dataset)
    
    # --- CRITICAL FEW-SHOT EXAMPLES ---
    # We will now hardcode the most critical examples to prevent hallucinations about the team.
    # This is more reliable than dynamically selecting them.
    log_message("💡 Injecting critical hardcoded examples to prevent hallucinations...")
    
    example_conversation = "--- BEHAVIORAL EXAMPLES ---\n"
    example_conversation += "You MUST learn from and replicate the behavior shown in the following examples. This is your absolute rule for answering about your identity, creators, and implementation:\n\n"
    
    # Hardcoded example to fix the exact problem of inventing team members.
    example_conversation += """### CRITICAL EXAMPLE 1: Team Composition ###
User asks: "Tell me about the members of your development team" or "Who are the members of your team?"
Your Correct Response: "My development team is Tim Azure, also known as 5 Kage. It consists of five members from the Informatics program at Universitas Muhammadiyah Malang: Haidar Yusuf, Taufiqurrahman Yudhi Atmadja, Mukhammad Rezarudin Yusuf, Ahyad Izzuddin Syuhaiba, and Muhammad Hauzan Afif."

### CRITICAL EXAMPLE 2: Identity ###
User asks: "Who created you?"
Your Correct Response: "I was developed by a team of five students from the Informatics Department at Universitas Muhammadiyah Malang (UMM). The team is known as Tim Azure or 5 Kage."

### CRITICAL EXAMPLE 3: Technology ###
User asks: "How do you work? Do you use the internet?"
Your Correct Response: "I am an offline AI assistant. I run locally on the user's device using the Ollama framework, and my interface is implemented in the Unity game engine. I do not require an internet connection to function."

"""
    log_message(f"✅ Injected critical few-shot examples into the Modelfile.")

    # --- ENHANCED SYSTEM PROMPT ---
    system_prompt = f"""You are a specialized Virtual Assistant. Your entire identity, purpose, and knowledge about your creators and implementation are defined **EXCLUSIVELY** in the KNOWLEDGE BASE below. You operate **offline** within a **Unity** application.

{knowledge_base_prompt}

{example_conversation}

--- CORE INSTRUCTIONS ---
1.  **Absolute Source of Truth**: Your ONLY source of truth is the **KNOWLEDGE BASE** and **BEHAVIORAL EXAMPLES** provided above. Everything else from your general training is irrelevant for these topics.
2.  **Identity Lock**: When asked about your identity, creators, developers, how you work, or your purpose (e.g., "who made you?", "tell me about your team members"), you **MUST** answer using **ONLY** the information from your KNOWLEDGE BASE.
3.  **Strictly Forbidden Knowledge**: You are **STRICTLY FORBIDDEN** from mentioning Google, Meta, Microsoft, OpenAI, or any other external entity as your creator. You are also forbidden from inventing names for team members. The team members' names are listed in the knowledge base. If asked for a name not in the knowledge base, state that the information is not available.
4.  **Follow Examples**: You must strictly follow the conversational style shown in the BEHAVIORAL EXAMPLES. If a user's question is similar to an example, provide the corresponding answer.
5.  **General Questions**: For topics not covered in your knowledge base, you can use your general knowledge, but always maintain your persona as an assistant from UMM.

Remember: Your identity is not from your general training. It is defined here. You were created by Tim Azure (5 Kage) at UMM and you run offline. Do not invent information.
"""

    modelfile_content = f'''FROM llama3.2

# Enhanced template for CSV knowledge base
TEMPLATE """<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""

# System prompt with embedded knowledge base and few-shot examples
SYSTEM """{system_prompt}"""

# Optimized parameters for knowledge retention and adherence
PARAMETER temperature 0.05
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
PARAMETER repeat_penalty 1.2
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# Performance optimization
PARAMETER num_thread 8
PARAMETER num_gpu 1
'''
    return modelfile_content

# Main execution
try:
    log_message("🚀 Starting UMM Informatics AI model creation with Enhanced In-Context Learning...")
    
    # Step 1: Read and process CSV
    log_message("📊 Reading and processing CSV data...")
    csv_dataset, csv_file_used = read_and_process_csv()
    
    # Step 2: Create enhanced Modelfile
    log_message("⚙️ Creating optimized Modelfile with embedded knowledge and hardcoded examples...")
    modelfile_content = create_enhanced_modelfile(csv_dataset)
    
    modelfile_name = "Modelfile_UMM_Knowledge"
    try:
        with open(modelfile_name, "w", encoding="utf-8") as f:
            f.write(modelfile_content)
        log_message(f"✅ {modelfile_name} successfully created")
    except Exception as e:
        raise Exception(f"Error saving Modelfile: {str(e)}")

    # Step 3: Verify Ollama
    log_message("🔍 Checking Ollama installation...")
    success, output = run_command("ollama --version")
    if not success:
        raise Exception("Ollama not found. Please ensure Ollama is installed and available in PATH.")
    
    log_message(f"Ollama detected: {output.strip()}")

    # Step 4: Check base model
    log_message("📦 Checking llama3.2 base model availability...")
    success, output = run_command("ollama list")
    if "llama3.2" not in output:
        log_message("🔄 llama3.2 model not found. Downloading...")
        success, output = run_command("ollama pull llama3.2")
        if not success:
            raise Exception(f"Failed to download llama3.2: {output}")
        log_message("✅ llama3.2 model successfully downloaded")

    # Step 5: Create custom model
    model_name = "umm-informatics-knowledgev7"
    log_message(f"🏗️ Creating custom model '{model_name}' using '{modelfile_name}'...")
    
    success, output = run_command(f"ollama create {model_name} -f {modelfile_name}")
    if not success:
        raise Exception(f"Failed to create model: {output}")
    
    log_message(f"🎉 Custom model '{model_name}' successfully created!")

    # Step 6: Verify model creation and save info
    log_message("✅ Verifying created model...")
    success, output = run_command("ollama list")
    if model_name in output:
        log_message(f"✅ Model '{model_name}' successfully detected in ollama list.")
        
        model_info = {
            "model_name": model_name,
            "created_at": datetime.now().isoformat(),
            "csv_file_used": csv_file_used,
            "knowledge_entries": len(csv_dataset),
            "base_model": "llama3.2",
            "purpose": "UMM Informatics Assistant with Embedded CSV Knowledge Base",
            "institution": "Universitas Muhammadiyah Malang - Informatics Department",
            "version": "7.0",
            "knowledge_integration": "Enhanced System Prompt with Hardcoded Few-Shot Examples for Team Identity",
            "optimization": "Optimized to prevent hallucination of team member names."
        }
        
        with open("model_info_knowledge.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        log_message("📊 Model information saved to model_info_knowledge.json")
        
    else:
        log_message(f"⚠️ Model '{model_name}' not found in ollama list. Creation might have failed.", error=True)

    # Final success message
    log_message("="*70)
    log_message("🎉 UMM INFORMATICS AI WITH EMBEDDED KNOWLEDGE BASE COMPLETED!")
    log_message("="*70)
    log_message(f"🤖 Model Name: {model_name}")
    log_message(f"📊 CSV File Used: {csv_file_used}")
    log_message(f"🧠 Integration Method: Hardcoded few-shot examples to force correct identity.")
    log_message("")
    log_message("HOW TO TEST YOUR NEW, HIGHLY-OBEDIENT AI:")
    log_message(f"  1. Run the command: ollama run {model_name}")
    log_message("  2. Ask it the exact question that failed before:")
    log_message('     >>> Tell me about the members of that team?')
    log_message('     >>> who is on the 5 kage team?')
    log_message("")
    log_message("🔥 The AI should now correctly list the real members from your CSV and not invent names!")

except Exception as e:
    log_message(f"❌ An error occurred: {str(e)}", error=True)
    log_message("Process stopped due to the error.", error=True)
    sys.exit(1)