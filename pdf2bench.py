import pdfplumber
import json
import re
import logging
from pathlib import Path
from PIL import Image
import io
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_to_json_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Remove extra whitespace and normalize line endings"""
    return ' '.join(text.split())

def extract_sections_from_pdf(pdf_path):
    """Extract image data from PDF with comprehensive logging"""
    logger.info(f"Starting PDF extraction from: {pdf_path}")
    all_image_data = []
    current_section = None
    section_stats = defaultdict(int)
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"PDF opened successfully. Total pages: {len(pdf.pages)}")
            text = ""
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    logger.debug(f"Extracted text from page {page_num}")
                else:
                    logger.warning(f"No text found on page {page_num}")
    except Exception as e:
        logger.error(f"Error opening PDF: {e}")
        return []
    
    # Split into main sections more reliably
    sections = []
    current_text = ""
    current_section = None
    
    logger.info("Parsing PDF content into sections...")
    
    for line in text.split('\n'):
        if re.match(r'^(REAL|ANIMATED|AI_GENERATED)$', line.strip()):
            if current_section:
                sections.append((current_section, current_text))
                logger.info(f"Found section: {current_section}")
            current_section = line.strip()
            current_text = ""
        else:
            current_text += line + "\n"
    
    if current_section:
        sections.append((current_section, current_text))
        logger.info(f"Found section: {current_section}")
    
    logger.info(f"Total sections found: {len(sections)}")
    
    for section_type, section_content in sections:
        logger.info(f"Processing section: {section_type}")
        
        # Split section content into individual image entries
        image_entries = re.split(r'Image \d+', section_content)
        
        images_in_section = 0
        for idx, entry in enumerate(image_entries[1:], 1):  # Skip first empty split
            # Extract theme (previously source)
            theme_match = re.search(r'\((.*?)\)', entry)
            theme = theme_match.group(1) if theme_match else "unknown"
            
            if theme == "unknown":
                logger.warning(f"No theme found for {section_type} Image {idx}")
            
            # Extract questions and answers
            questions = []
            entry_lines = entry.split('\n')
            i = 0
            questions_found = 0
            
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
                        questions_found += 1
                        logger.debug(f"Parsed question {q_num} for {section_type} Image {idx}")
                    else:
                        logger.warning(f"Could not parse question format: {clean_text(full_question)}")
                i += 1
            
            # Create image entry
            image_entry = {
                "image_id": f"image{idx:02d}",
                "image_type": section_type,
                "theme": theme,  # Changed from "source" to "theme"
                "path": f"merged_data/{section_type}/image{idx:02d}.jpg",
                "questions": questions
            }
            all_image_data.append(image_entry)
            images_in_section += 1
            section_stats[section_type] += 1
            
            # Log details for each image
            logger.info(f"Processed {section_type} Image {idx:02d}: theme='{theme}', questions={questions_found}")
            
            if questions_found == 0:
                logger.warning(f"No questions found for {section_type} Image {idx}")
            elif questions_found != 3:
                logger.warning(f"Expected 3 questions but found {questions_found} for {section_type} Image {idx}")
        
        logger.info(f"Completed section {section_type}: {images_in_section} images processed")
    
    # Log final statistics
    logger.info("=== EXTRACTION SUMMARY ===")
    logger.info(f"Total images processed: {len(all_image_data)}")
    for section, count in section_stats.items():
        logger.info(f"{section}: {count} images")
    
    # Check for missing questions
    total_questions = sum(len(img['questions']) for img in all_image_data)
    expected_questions = len(all_image_data) * 3
    logger.info(f"Total questions extracted: {total_questions}")
    logger.info(f"Expected questions: {expected_questions}")
    
    if total_questions != expected_questions:
        logger.warning(f"Question count mismatch! Missing {expected_questions - total_questions} questions")
    
    return all_image_data

def validate_extracted_data(image_data):
    """Validate the extracted data and log any issues"""
    logger.info("Validating extracted data...")
    
    issues = []
    section_counts = defaultdict(int)
    question_categories = defaultdict(int)
    themes_per_section = defaultdict(set)
    
    for img in image_data:
        section_counts[img['image_type']] += 1
        themes_per_section[img['image_type']].add(img['theme'])
        
        # Check for required fields
        if not img.get('theme') or img['theme'] == 'unknown':
            issues.append(f"Missing or unknown theme for {img['image_id']}")
        
        # Check questions
        if len(img['questions']) != 3:
            issues.append(f"{img['image_id']} has {len(img['questions'])} questions (expected 3)")
        
        for q in img['questions']:
            if q.get('property_category'):
                question_categories[q['property_category']] += 1
            
            # Check for empty answers
            if not q.get('answer'):
                issues.append(f"Empty answer for {img['image_id']} {q['id']}")
    
    # Log validation results
    logger.info("=== VALIDATION RESULTS ===")
    logger.info(f"Section distribution: {dict(section_counts)}")
    logger.info(f"Property categories: {dict(question_categories)}")
    
    for section, themes in themes_per_section.items():
        logger.info(f"{section} themes ({len(themes)}): {sorted(themes)}")
    
    if issues:
        logger.warning(f"Found {len(issues)} validation issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Data validation passed - no issues found!")
    
    return len(issues) == 0

def update_benchmark_json(json_path, new_data):
    """Create benchmark JSON with comprehensive logging"""
    logger.info(f"Creating benchmark JSON file: {json_path}")
    
    # Create directory structure
    created_dirs = []
    for section in ["REAL", "ANIMATED", "AI_GENERATED"]:
        dir_path = Path(f"merged_data/{section}")
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(dir_path))
    
    if created_dirs:
        logger.info(f"Created directories: {created_dirs}")
    
    # Count data for logging
    section_counts = defaultdict(int)
    total_questions = 0
    
    for img in new_data:
        section_counts[img['image_type']] += 1
        total_questions += len(img['questions'])
    
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
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark, f, indent=4)
        
        logger.info(f"Successfully saved benchmark JSON to: {json_path}")
        logger.info("=== BENCHMARK SUMMARY ===")
        logger.info(f"Total images in benchmark: {len(new_data)}")
        logger.info(f"Total questions in benchmark: {total_questions}")
        logger.info(f"Images per section: {dict(section_counts)}")
        
        # Verify file was created and check size
        json_file = Path(json_path)
        if json_file.exists():
            file_size = json_file.stat().st_size
            logger.info(f"Benchmark file size: {file_size:,} bytes")
        else:
            logger.error("Benchmark file was not created!")
            return False
            
    except Exception as e:
        logger.error(f"Error saving benchmark JSON: {e}")
        return False
    
    return True

def main():
    """Main function with comprehensive logging"""
    logger.info("=== PDF TO JSON CONVERSION STARTED ===")
    
    # Configuration
    pdf_path = "/var/scratch/ave303/OP_bench/ORBIT_data.pdf"
    json_path = "benchmark.json"
    
    # Check if PDF exists
    if not Path(pdf_path).exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    logger.info(f"Input PDF: {pdf_path}")
    logger.info(f"Output JSON: {json_path}")
    
    # Extract data from PDF
    image_data = extract_sections_from_pdf(pdf_path)
    
    if not image_data:
        logger.error("No data extracted from PDF!")
        return
    
    # Validate extracted data
    validation_passed = validate_extracted_data(image_data)
    
    if not validation_passed:
        logger.warning("Data validation failed, but proceeding with JSON creation...")
    
    # Update benchmark JSON
    success = update_benchmark_json(json_path, image_data)
    
    if success:
        logger.info("=== PDF TO JSON CONVERSION COMPLETED SUCCESSFULLY ===")
    else:
        logger.error("=== PDF TO JSON CONVERSION FAILED ===")

# Usage
if __name__ == "__main__":
    main()