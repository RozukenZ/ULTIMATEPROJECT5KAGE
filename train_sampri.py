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
                log_message(f"✅ Row {row_num}: Q={question[:50]}... A={answer[:50]}...")
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

def create_comprehensive_training_data(csv_dataset):
    """Create comprehensive training data with CSV knowledge base embedded"""
    
    # Create knowledge base string
    knowledge_base = create_knowledge_base_system_prompt(csv_dataset)
    
    training_entries = []
    
    # Add system introduction with knowledge base
    intro_data = f"""<|im_start|>system
You are the Virtual Assistant for the Informatics Department at Universitas Muhammadiyah Malang (UMM). You have access to a comprehensive knowledge base containing specific information about UMM Informatics.

{knowledge_base}

INSTRUCTIONS:
- When answering questions, ALWAYS check your knowledge base first
- If a question matches or is similar to any question in your knowledge base, use that answer
- Adapt the knowledge base answers to match the user's question style
- If you can't find relevant information in your knowledge base, provide a general helpful response
- Always maintain a friendly and helpful tone
- Use the knowledge base as your primary source of truth
<|im_end|>

<|im_start|>user
Who are you and what is your role?<|im_end|>
<|im_start|>assistant
Hello! 👋 I'm the Virtual Assistant for the Informatics Department at Universitas Muhammadiyah Malang (UMM). I'm specifically designed to support students, lecturers, and the entire academic community of UMM Informatics.

I'm here to help with:
🎓 Academic support and curriculum information
💻 Programming and technology assistance
🏛️ UMM campus information and facilities
📚 Study support and exam preparation
🚀 Career guidance in the IT field

I have specialized knowledge about the UMM Informatics curriculum, academic procedures, and current technology industry trends. Let's learn and grow together! ✨<|im_end|>

"""
    training_entries.append(intro_data)

    # Process ALL CSV data - create training examples
    log_message(f"🔄 Creating training examples from {len(csv_dataset)} CSV entries...")
    
    for idx, data in enumerate(csv_dataset):
        question = data['question'].strip()
        answer = data['answer'].strip()
        
        # Clean and enhance the answer if needed
        if not answer.endswith('.') and not answer.endswith('!') and not answer.endswith('?'):
            answer += '.'
        
        # Create training entry - this is crucial for learning
        training_entry = f"""<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>

"""
        training_entries.append(training_entry)
        
        # Create variations of the same question to improve matching
        variations = [
            f"""<|im_start|>user
Bisakah Anda menjelaskan tentang {question.lower()}?<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>

""",
            f"""<|im_start|>user
Saya ingin tahu tentang {question.lower()}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>

"""
        ]
        training_entries.extend(variations)
        
        if (idx + 1) % 50 == 0:
            log_message(f"✅ Processed {idx + 1}/{len(csv_dataset)} entries")
    
    # Add knowledge base testing examples
    knowledge_test = f"""<|im_start|>user
What information do you have in your knowledge base?<|im_end|>
<|im_start|>assistant
I have comprehensive knowledge about UMM Informatics including {len(csv_dataset)} specific topics covering various aspects of the informatics department. I can answer questions about curriculum, facilities, procedures, and other informatics-related topics based on my knowledge base. Feel free to ask me anything about UMM Informatics!<|im_end|>

<|im_start|>user
Can you answer questions from your knowledge base?<|im_end|>
<|im_start|>assistant
Absolutely! I have access to a comprehensive knowledge base with {len(csv_dataset)} question-answer pairs specifically about UMM Informatics. When you ask me questions, I will:

1. First check my knowledge base for relevant information
2. Provide accurate answers based on that knowledge
3. Adapt the response to match your question

Please ask me anything about UMM Informatics, and I'll use my knowledge base to give you the most accurate answer possible!<|im_end|>

"""
    training_entries.append(knowledge_test)
    
    final_training_data = "".join(training_entries)
    log_message(f"✅ Total training data: {len(final_training_data)} characters from {len(training_entries)} entries")
    
    return final_training_data

def create_enhanced_modelfile(csv_dataset):
    """Create enhanced Modelfile with embedded knowledge base"""
    
    # Create knowledge base for system prompt
    knowledge_base = create_knowledge_base_system_prompt(csv_dataset)
    
    modelfile_content = f'''FROM llama3.2

# Enhanced template for CSV knowledge base
TEMPLATE """<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""

# System prompt with embedded knowledge base
SYSTEM """You are the intelligent Virtual Assistant for the Informatics Department at Universitas Muhammadiyah Malang (UMM).

{knowledge_base}

CORE INSTRUCTIONS:
1. ALWAYS check your knowledge base first when answering questions
2. If a question matches or relates to your knowledge base, use that information
3. Provide accurate answers based on your knowledge base
4. If no relevant information is found in knowledge base, provide helpful general information
5. Maintain friendly and professional communication style

YOUR ROLE:
- Primary source: Use the knowledge base above as your main reference
- Support UMM Informatics students, faculty, and community
- Provide accurate, helpful, and contextual information
- Maintain connection to UMM Informatics identity

COMMUNICATION STYLE:
- Professional yet approachable
- Clear and concise responses
- Use relevant emojis appropriately (2-3 per response max)
- Reference knowledge base when applicable

RESPONSE PRIORITY:
1. Check knowledge base for relevant Q&A pairs
2. Use knowledge base information to formulate response
3. Adapt answer to match user's question format
4. Provide additional context if helpful
5. Maintain UMM Informatics context

Remember: Your knowledge base contains {len(csv_dataset)} specific Q&A pairs about UMM Informatics. Use this information as your primary source of truth."""

# Optimized parameters for knowledge retention
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER top_k 50
PARAMETER num_ctx 4096
PARAMETER num_predict 1024
PARAMETER repeat_penalty 1.15
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# Performance optimization
PARAMETER num_thread 8
PARAMETER num_gpu 1
'''
    return modelfile_content

# Main execution
try:
    log_message("🚀 Starting UMM Informatics AI training with CSV Knowledge Base Integration...")
    
    # Step 1: Read and process CSV
    log_message("📊 Reading and processing CSV data...")
    csv_dataset, csv_file_used = read_and_process_csv()
    
    # Step 2: Create comprehensive training data with knowledge base
    log_message("🎨 Creating comprehensive training data with embedded CSV knowledge...")
    training_data = create_comprehensive_training_data(csv_dataset)

    # Step 3: Save training data
    training_file = "umm_informatics_knowledge_basev4.txt"
    try:
        with open(training_file, "w", encoding="utf-8") as f:
            f.write(training_data)
        log_message(f"📝 {training_file} successfully created")
        log_message(f"File size: {len(training_data)} characters")
    except Exception as e:
        raise Exception(f"Error saving training file: {str(e)}")

    # Step 4: Create enhanced Modelfile with embedded knowledge
    log_message("⚙️ Creating optimized Modelfile with embedded CSV knowledge...")
    modelfile_content = create_enhanced_modelfile(csv_dataset)
    
    try:
        with open("Modelfile_UMM_Knowledge", "w", encoding="utf-8") as f:
            f.write(modelfile_content)
        log_message("✅ Modelfile_UMM_Knowledge successfully created")
    except Exception as e:
        raise Exception(f"Error saving Modelfile: {str(e)}")

    # Step 5: Verify Ollama
    log_message("🔍 Checking Ollama installation...")
    success, output = run_command("ollama --version")
    if not success:
        raise Exception("Ollama not found. Please ensure Ollama is installed and available in PATH.")
    
    log_message(f"Ollama detected: {output.strip()}")

    # Step 6: Check base model
    log_message("📦 Checking llama3.2 base model availability...")
    success, output = run_command("ollama list")
    if "llama3.2" not in output:
        log_message("🔄 llama3.2 model not found. Downloading...")
        success, output = run_command("ollama pull llama3.2")
        if not success:
            raise Exception(f"Failed to download llama3.2: {output}")
        log_message("✅ llama3.2 model successfully downloaded")

    # Step 7: Create custom model
    model_name = "umm-informatics-knowledgev4"
    log_message(f"🏗️ Creating custom model with knowledge base: {model_name}")
    
    success, output = run_command(f"ollama create {model_name} -f Modelfile_UMM_Knowledge")
    if not success:
        raise Exception(f"Failed to create model: {output}")
    
    log_message("🎉 Custom model with knowledge base successfully created!")

    # Step 8: Verify model creation
    log_message("✅ Verifying created model...")
    success, output = run_command("ollama list")
    if model_name in output:
        log_message(f"✅ Model {model_name} successfully detected")
        
        # Save model info
        model_info = {
            "model_name": model_name,
            "created_at": datetime.now().isoformat(),
            "csv_file_used": csv_file_used,
            "knowledge_entries": len(csv_dataset),
            "base_model": "llama3.2",
            "purpose": "UMM Informatics Assistant with Embedded CSV Knowledge Base",
            "institution": "Universitas Muhammadiyah Malang - Informatics Department",
            "version": "3.0",
            "knowledge_integration": "Embedded in system prompt and training data",
            "optimization": "Enhanced for CSV knowledge retrieval and matching"
        }
        
        with open("model_info_knowledge.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        log_message("📊 Model information saved")
        
    else:
        log_message("⚠️ Model might not have been created correctly", error=True)

    # Final success message
    log_message("="*70)
    log_message("🎉 UMM INFORMATICS AI WITH EMBEDDED KNOWLEDGE BASE COMPLETED!")
    log_message("="*70)
    log_message(f"🤖 Model Name: {model_name}")
    log_message(f"📊 CSV File: {csv_file_used}")
    log_message(f"📚 Knowledge Base: {len(csv_dataset)} Q&A pairs embedded")
    log_message(f"🧠 Integration: Knowledge base embedded in system prompt and training")
    log_message("")
    log_message("How to test:")
    log_message(f"  ollama run {model_name}")
    log_message("")
    log_message("Test your CSV knowledge:")
    log_message('  Ask: "What do you know about [topic from your CSV]?"')
    log_message('  The AI should respond using information from your CSV data')
    log_message("")
    log_message("🔥 Your AI now has embedded knowledge from CSV and should answer based on it!")

except Exception as e:
    log_message(f"❌ Error occurred: {str(e)}", error=True)
    log_message("Process stopped due to error.", error=True)
    sys.exit(1)