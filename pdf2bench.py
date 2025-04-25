import pdfplumber
import json
import re
from pathlib import Path
from PIL import Image
import io

def clean_text(text):
    # Remove extra whitespace and normalize line endings
    return ' '.join(text.split())

def extract_sections_from_pdf(pdf_path):
    all_image_data = []
    current_section = None
    
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    
    # Split into main sections more reliably
    sections = []
    current_text = ""
    current_section = None
    
    for line in text.split('\n'):
        if re.match(r'^(REAL|ANIMATED|AI_GENERATED)$', line.strip()):
            if current_section:
                sections.append((current_section, current_text))
            current_section = line.strip()
            current_text = ""
        else:
            current_text += line + "\n"
    
    if current_section:
        sections.append((current_section, current_text))
    
    for section_type, section_content in sections:
        # Split section content into individual image entries
        image_entries = re.split(r'Image \d+', section_content)
        
        for idx, entry in enumerate(image_entries[1:], 1):  # Skip first empty split
            # Extract source
            source_match = re.search(r'\((.*?)\)', entry)
            source = source_match.group(1) if source_match else "unknown"
            
            # Extract questions and answers
            questions = []
            entry_lines = entry.split('\n')
            i = 0
            while i < len(entry_lines):
                line = entry_lines[i].strip()
                if re.match(r'^\d+\.', line):
                    q_num = int(line[0])
                    full_question = line
                    
                    # Check if question continues on next lines
                    while i + 1 < len(entry_lines) and not re.match(r'^\d+\.', entry_lines[i + 1].strip()):
                        i += 1
                        full_question += ' ' + entry_lines[i].strip()
                    
                    # Extract question components with new pattern
                    # Handle both simple answers and answers with verification lists
                    question_pattern = r'(\d+)\.\s+(.*?)\?\s*(\d+)\s*(?:\((.*?)\))?(?:\s*\((.*?)\))?'
                    parts = re.match(question_pattern, clean_text(full_question))
                    
                    if parts:
                        question = parts.group(2) + "?"
                        answer = parts.group(3)
                        category = parts.group(4)
                        verification = parts.group(5)  # This will capture any verification list
                        
                        question_entry = {
                            "id": f"Q{q_num}",
                            "question": question,
                            "answer": answer,
                            "property_category": category.lower() if category else None
                        }
                        
                        # Only add verification if it exists
                        if verification:
                            question_entry["answer_verification"] = verification
                        
                        questions.append(question_entry)
                i += 1
            
            # Create image entry
            image_entry = {
                "image_id": f"image{idx:02d}",
                "image_type": section_type,
                "source": source,
                "path": f"data/{section_type}/image{idx:02d}.jpg",
                "questions": questions
            }
            all_image_data.append(image_entry)
    
    return all_image_data

def update_benchmark_json(json_path, new_data):
    # Create directory structure
    for section in ["REAL", "ANIMATED", "AI_GENERATED"]:
        Path(f"data/{section}").mkdir(parents=True, exist_ok=True)
    
    benchmark = {
        "benchmark": {
            "image_types": ["REAL", "ANIMATED", "AI_GENERATED"],
            "question_types": {
                "Q1": "direct_recognition",
                "Q2": "property_inference",
                "Q3": "counterfactual_reasoning"
            },
            "property_categories": {
                "Q1": ["physical", "taxonomic"],
                "Q2": ["functional", "relational"],
                "Q3": ["physical", "taxonomic", "functional", "relational"]
            },
            "images": new_data
        }
    }
    
    # Save updated benchmark
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark, f, indent=4)

# Usage
if __name__ == "__main__":
    pdf_path = "C:/Users/abkol/Desktop/VU_Courses/Masters_Thesis/ObjectProp_Bench.pdf"
    json_path = "benchmark.json"
    
    # Extract data from PDF
    image_data = extract_sections_from_pdf(pdf_path)
    
    # Update benchmark JSON
    update_benchmark_json(json_path, image_data)