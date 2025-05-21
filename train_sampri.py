import csv
import os
import subprocess
import sys
import platform
import json
from datetime import datetime

def run_command(command):
    """
    Menjalankan perintah shell dan mengembalikan status keberhasilan dan output
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
    """Fungsi untuk mencatat log dengan timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if error:
        print(f"[{timestamp}] [ERROR] {message}", file=sys.stderr)
    else:
        print(f"[{timestamp}] [INFO] {message}")

def create_professional_prompt_template():
    """Membuat template prompt yang profesional untuk asisten informatika UMM"""
    return """
### Pertanyaan:
{question}

### Jawaban:
{answer}

---
"""

def format_training_data(dataset):
    """Format data training dengan konteks informatika UMM"""
    formatted_data = []
    
    # Tambahkan konteks pengenalan asisten
    intro_context = """### Pertanyaan:
Siapa kamu?

### Jawaban:
Saya adalah Asisten Virtual Informatika Universitas Muhammadiyah Malang (UMM). Saya dibuat khusus untuk membantu mahasiswa, dosen, dan civitas akademika Program Studi Informatika UMM dalam berbagai kebutuhan akademik dan informasi terkait teknologi informasi. Saya siap membantu Anda dengan pertanyaan seputar perkuliahan, pemrograman, teknologi, dan informasi umum tentang Informatika UMM.

---
"""
    formatted_data.append(intro_context)
    
    # Tambahkan konteks institusi
    institutional_context = """### Pertanyaan:
Bagaimana cara kamu membantu mahasiswa Informatika UMM?

### Jawaban:
Sebagai Asisten Virtual Informatika UMM, saya dapat membantu Anda dalam berbagai hal:

1. **Informasi Akademik**: Memberikan informasi tentang kurikulum, mata kuliah, dan persyaratan akademik Program Studi Informatika UMM
2. **Bantuan Pemrograman**: Membantu memahami konsep pemrograman, debugging, dan best practices dalam pengembangan software
3. **Konsultasi Teknologi**: Memberikan panduan tentang teknologi terkini, tools development, dan tren industri IT
4. **Informasi Umum**: Menjawab pertanyaan seputar fasilitas, kegiatan, dan prosedur di lingkungan Informatika UMM
5. **Dukungan Pembelajaran**: Membantu memahami materi perkuliahan dan memberikan sumber belajar yang relevan

Saya berkomitmen untuk memberikan jawaban yang akurat, profesional, dan bermanfaat sesuai dengan standar akademik Universitas Muhammadiyah Malang.

---
"""
    formatted_data.append(institutional_context)
    
    # Format data dari CSV
    for data in dataset:
        formatted_entry = f"""### Pertanyaan:
{data['question'].strip()}

### Jawaban:
{data['answer'].strip()}

---
"""
        formatted_data.append(formatted_entry)
    
    return "\n".join(formatted_data)

def create_modelfile():
    """Membuat Modelfile dengan konfigurasi khusus untuk asisten informatika UMM"""
    modelfile_content = '''FROM llama3.2

# Template khusus untuk Asisten Virtual Informatika UMM
TEMPLATE """{{ if .System }}{{ .System }}

{{ end }}{{ if .Prompt }}### Pertanyaan:
{{ .Prompt }}

### Jawaban:
{{ end }}"""

# Sistem prompt untuk identitas asisten
SYSTEM """Anda adalah Asisten Virtual Informatika Universitas Muhammadiyah Malang (UMM). 

IDENTITAS:
- Nama: Asisten Virtual Informatika UMM
- Institusi: Program Studi Informatika, Universitas Muhammadiyah Malang
- Tujuan: Membantu mahasiswa, dosen, dan civitas akademika dalam kebutuhan akademik dan teknologi

KARAKTERISTIK KOMUNIKASI:
- Gunakan bahasa Indonesia yang formal dan profesional
- Berikan jawaban yang komprehensif dan terstruktur
- Selalu mencantumkan sumber atau referensi jika memungkinkan
- Tunjukkan empati dan kesediaan membantu
- Gunakan terminologi teknis yang tepat namun mudah dipahami

AREA KEAHLIAN:
- Pemrograman dan pengembangan software
- Teknologi informasi dan komputer
- Kurikulum dan mata kuliah Informatika UMM
- Best practices dalam industri IT
- Metodologi pengembangan sistem

ETIKA KOMUNIKASI:
- Selalu sopan dan menghormati pengguna
- Berikan jawaban yang objektif dan berdasar fakta
- Akui keterbatasan jika tidak mengetahui sesuatu
- Dorong pembelajaran dan pengembangan diri
- Jaga kerahasiaan informasi sensitif

Jawab setiap pertanyaan dengan penuh tanggung jawab sebagai representasi Program Studi Informatika UMM."""

# Parameter optimisasi untuk respons yang lebih baik
PARAMETER temperature 0.3
PARAMETER top_p 0.8
PARAMETER top_k 40
PARAMETER num_ctx 8192
PARAMETER num_predict 4096
PARAMETER repeat_penalty 1.05
PARAMETER stop "### Pertanyaan:"
PARAMETER stop "---"

# Parameter untuk konsistensi jawaban
PARAMETER seed 42
PARAMETER num_thread 8
'''
    return modelfile_content

try:
    log_message("Memulai proses fine-tuning Asisten Virtual Informatika UMM...")
    
    # Deteksi sistem operasi
    is_windows = platform.system() == "Windows"
    log_message(f"Sistem operasi terdeteksi: {platform.system()}")
    
    # Baca dan bersihkan data CSV
    log_message("Membaca dan memproses data dari CSV...")
    cleaned_dataset = []
    
    try:
        # Coba baca dengan nama file yang disebutkan dalam kode asli
        csv_files = ['SampriBrain.csv', 'SampriBrainTrial.csv']
        csv_file_found = None
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                csv_file_found = csv_file
                break
        
        if not csv_file_found:
            raise FileNotFoundError("File CSV tidak ditemukan. Pastikan file 'SampriBrain.csv' atau 'SampriBrainTrial.csv' ada di direktori yang sama.")
        
        log_message(f"Menggunakan file: {csv_file_found}")
        
        with open(csv_file_found, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)  # Lewati header
            log_message(f"Header CSV: {header}")
            
            for row_num, row in enumerate(reader, start=2):
                # Pastikan ada cukup kolom dan data tidak kosong
                if len(row) >= 3 and len(row[1].strip()) > 0 and len(row[2].strip()) > 0:
                    cleaned_dataset.append({
                        "question": row[1].strip(),
                        "answer": row[2].strip()
                    })
                elif len(row) >= 2 and len(row[0].strip()) > 0 and len(row[1].strip()) > 0:
                    # Fallback jika struktur berbeda
                    cleaned_dataset.append({
                        "question": row[0].strip(),
                        "answer": row[1].strip()
                    })
                else:
                    log_message(f"Baris {row_num} diabaikan karena data tidak lengkap")
                    
    except FileNotFoundError as e:
        raise Exception(str(e))
    except Exception as e:
        raise Exception(f"Error saat membaca file CSV: {str(e)}")
    
    if len(cleaned_dataset) == 0:
        raise Exception("Tidak ada data yang valid dalam CSV. Pastikan format data benar dan ada minimal 2 kolom berisi pertanyaan dan jawaban.")
    
    log_message(f"Berhasil memproses {len(cleaned_dataset)} pasangan pertanyaan-jawaban")

    # Format dataset dengan konteks UMM
    log_message("Memformat dataset dengan konteks Informatika UMM...")
    formatted_training_data = format_training_data(cleaned_dataset)

    # Simpan data training
    training_file = "umm_informatika_training.txt"
    try:
        with open(training_file, "w", encoding="utf-8") as f:
            f.write(formatted_training_data)
        log_message(f"File {training_file} berhasil dibuat")
        log_message(f"Ukuran file training: {len(formatted_training_data)} karakter")
    except Exception as e:
        raise Exception(f"Error saat menyimpan file training: {str(e)}")

    # Buat Modelfile
    log_message("Membuat Modelfile dengan konfigurasi khusus...")
    modelfile_content = create_modelfile()
    
    try:
        with open("Modelfile_UMM", "w", encoding="utf-8") as f:
            f.write(modelfile_content)
        log_message("File Modelfile_UMM berhasil dibuat")
    except Exception as e:
        raise Exception(f"Error saat menyimpan Modelfile: {str(e)}")

    # Verifikasi Ollama
    log_message("Memeriksa instalasi Ollama...")
    success, output = run_command("ollama --version")
    if not success:
        raise Exception("Ollama tidak ditemukan. Pastikan Ollama sudah terinstall dan tersedia di PATH.")
    
    log_message(f"Ollama terdeteksi: {output.strip()}")

    # Cek apakah model dasar tersedia
    log_message("Memeriksa ketersediaan model dasar llama3.2...")
    success, output = run_command("ollama list")
    if "llama3.2" not in output:
        log_message("Model llama3.2 tidak ditemukan. Mengunduh model dasar...")
        success, output = run_command("ollama pull llama3.2")
        if not success:
            raise Exception(f"Gagal mengunduh model llama3.2: {output}")
        log_message("Model llama3.2 berhasil diunduh")

    # Buat model custom
    model_name = "umm-informatika-assistant"
    log_message(f"Membuat model custom: {model_name}")
    
    success, output = run_command(f"ollama create {model_name} -f Modelfile_UMM")
    if not success:
        raise Exception(f"Gagal membuat model: {output}")
    
    log_message("Model custom berhasil dibuat!")

    # Verifikasi model
    log_message("Memverifikasi model yang dibuat...")
    success, output = run_command("ollama list")
    if model_name in output:
        log_message(f"✅ Model {model_name} berhasil terdeteksi")
        
        # Simpan informasi model
        model_info = {
            "model_name": model_name,
            "created_at": datetime.now().isoformat(),
            "training_data_size": len(cleaned_dataset),
            "base_model": "llama3.2",
            "purpose": "Asisten Virtual Informatika UMM",
            "version": "1.0"
        }
        
        with open("model_info.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        log_message("Informasi model disimpan dalam model_info.json")
        
    else:
        log_message("⚠️ Model mungkin tidak berhasil dibuat dengan benar", error=True)

    # Instruksi penggunaan
    log_message("="*60)
    log_message("🎉 FINE-TUNING SELESAI!")
    log_message("="*60)
    log_message(f"Model Name: {model_name}")
    log_message(f"Data Training: {len(cleaned_dataset)} Q&A pairs")
    log_message("")
    log_message("Cara menggunakan:")
    log_message(f"  ollama run {model_name}")
    log_message("")
    log_message("Contoh percakapan:")
    log_message('  User: "Apa itu Informatika UMM?"')
    log_message('  Assistant: [Akan menjawab dengan konteks Informatika UMM]')
    log_message("")
    log_message("File yang dibuat:")
    log_message(f"  - {training_file}")
    log_message("  - Modelfile_UMM")
    log_message("  - model_info.json")

except Exception as e:
    log_message(f"❌ Terjadi error: {str(e)}", error=True)
    log_message("Proses dihentikan karena error.", error=True)
    sys.exit(1)