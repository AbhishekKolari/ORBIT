import os
import csv
import re
import logging
from docx import Document
import pandas as pd
from typing import List, Dict, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('doc_to_csv_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentToCSVConverter:
    def __init__(self, annotator_file_path: str = None):
        # GitHub base URL template
        self.github_base_url = "https://github.com/AbhishekKolari/OP_bench/blob/main/merged_data"
        
        # Load annotator assignments if file provided
        self.annotator_assignments = {}
        if annotator_file_path and os.path.exists(annotator_file_path):
            self.load_annotator_assignments(annotator_file_path)
        else:
            logger.warning("Annotator file not provided or not found")
    
    def load_annotator_assignments(self, file_path: str):
        """Load annotator assignments from Excel file"""
        try:
            # Try different engines to read Excel file
            df = None
            engines = ['openpyxl', 'xlrd', None]  # None lets pandas choose
            
            for engine in engines:
                try:
                    if engine:
                        df = pd.read_excel(file_path, engine=engine)
                    else:
                        df = pd.read_excel(file_path)
                    logger.info(f"Successfully loaded Excel file using engine: {engine or 'default'}")
                    break
                except Exception as engine_error:
                    logger.warning(f"Failed to read Excel with engine {engine}: {engine_error}")
                    continue
            
            if df is None:
                logger.error(f"Could not read Excel file {file_path} with any available engine")
                logger.error("Try installing: pip install openpyxl xlrd")
                return
            
            # Expected columns: Annotator, Image Type, Image Number, Image ID
            required_columns = ['Annotator', 'Image Type', 'Image Number']
            
            # Check if required columns exist
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Required columns not found in {file_path}. Expected: {required_columns}")
                logger.error(f"Found columns: {list(df.columns)}")
                return
            
            # Process each row
            for _, row in df.iterrows():
                annotator = row['Annotator']
                image_type = str(row['Image Type']).lower()  # Convert to lowercase for consistency
                image_number = str(row['Image Number']).zfill(2)  # Ensure 2-digit format
                
                # Map image types to our naming convention
                type_mapping = {
                    'real': 'Real',
                    'animated': 'Animated',
                    'ai': 'AI_Generated',
                    'ai_generated': 'AI_Generated'
                }
                
                # This line maps the image type correctly
                mapped_type = type_mapping.get(image_type, image_type.title())
                
                # Store in format: {image_type: {image_number: annotator}}
                if mapped_type not in self.annotator_assignments:
                    self.annotator_assignments[mapped_type] = {}
                
                self.annotator_assignments[mapped_type][image_number] = annotator
            
            # Log statistics
            for img_type, assignments in self.annotator_assignments.items():
                annotator_counts = {}
                for annotator in assignments.values():
                    annotator_counts[annotator] = annotator_counts.get(annotator, 0) + 1
                logger.info(f"{img_type}: {len(assignments)} images assigned - {annotator_counts}")
            
        except Exception as e:
            logger.error(f"Error loading annotator assignments: {e}")
            logger.error(f"Make sure the file exists and has the correct format with columns: Annotator, Image Type, Image Number")
    
    def generate_github_link(self, image_number: str, image_type: str) -> str:
        """Generate GitHub link for image"""
        # Format image number as 2-digit with leading zero
        formatted_number = str(image_number).zfill(2)
        
        # Create the GitHub URL
        github_url = f"{self.github_base_url}/{image_type}/image{formatted_number}.jpg"
        
        return github_url
    
    def get_annotator(self, image_number: str, image_type: str) -> str:
        """Get annotator for specific image"""
        formatted_number = str(image_number).zfill(2)
        return self.annotator_assignments.get(image_type, {}).get(formatted_number, "UNASSIGNED")
    
    def parse_document(self, doc_path: str) -> List[Dict]:
        """Parse a Word document and extract image data with improved regex patterns"""
        try:
            doc = Document(doc_path)
            image_data = []
            current_image = None
            
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                
                # Skip empty paragraphs
                if not text:
                    continue
                
                # More flexible regex for image headings - handles various spacing
                # Matches: "Image 1 (theme)", "Image  1   (theme)", etc.
                if re.match(r'^\s*[Ii]mage\s+\d+', text, re.IGNORECASE):
                    # Save previous image if exists
                    if current_image:
                        image_data.append(current_image)
                    
                    # More flexible parsing for heading
                    match = re.search(r'[Ii]mage\s+(\d+).*?\(([^)]+)\)', text)
                    if match:
                        image_number = match.group(1)
                        theme = match.group(2).strip()
                        
                        current_image = {
                            'image_number': image_number,
                            'image_theme': theme,
                            'questions': [],
                            'ground_truth': [],
                            'property_dimension': [],
                            'list_of_objects': []
                        }
                        logger.debug(f"Found image header: Image {image_number} ({theme})")
                
                # More flexible question parsing - handles various formatting issues
                elif current_image and re.match(r'^\s*\d+\.', text):
                    # Try multiple regex patterns to handle different formatting
                    patterns = [
                        # Standard format: "1. Question? Answer (dimension) (objects)"
                        r'^\s*\d+\.\s*(.+?)\s+(\d+)\s*\(([^)]+)\)\s*\(([^)]*)\)\s*$',
                        # Format without objects: "1. Question? Answer (dimension)"
                        r'^\s*\d+\.\s*(.+?)\s+(\d+)\s*\(([^)]+)\)\s*$',
                        # Format with extra spaces or formatting issues
                        r'^\s*\d+\.\s*(.+?)\s+(\d+)\s*\(([^)]+)\)(?:\s*\(([^)]*)\))?\s*',
                        # More lenient pattern for malformed entries
                        r'^\s*\d+\.\s*(.+?)(?:\s+(\d+))?(?:\s*\(([^)]+)\))?(?:\s*\(([^)]*)\))?\s*$'
                    ]
                    
                    parsed = False
                    for pattern in patterns:
                        question_match = re.match(pattern, text)
                        if question_match:
                            groups = question_match.groups()
                            question = groups[0].strip() if groups[0] else ""
                            count = groups[1].strip() if len(groups) > 1 and groups[1] else ""
                            property_dim = groups[2].strip() if len(groups) > 2 and groups[2] else ""
                            objects = groups[3].strip() if len(groups) > 3 and groups[3] else ""
                            
                            # Only add if we have at least a question
                            if question:
                                current_image['questions'].append(question)
                                current_image['ground_truth'].append(count)
                                current_image['property_dimension'].append(property_dim)
                                current_image['list_of_objects'].append(objects)
                                parsed = True
                                logger.debug(f"Parsed question: {question}")
                                break
                    
                    if not parsed:
                        logger.warning(f"Could not parse question line: {text}")
                
                # Handle cases where question might be split across multiple lines
                elif current_image and current_image['questions'] and not text.startswith('Image'):
                    # This might be a continuation of the previous question or additional info
                    logger.debug(f"Potential continuation line: {text}")
            
            # Don't forget the last image
            if current_image:
                image_data.append(current_image)
            
            logger.info(f"Parsed {len(image_data)} images from {doc_path}")
            
            # Log details about each image for debugging
            for img in image_data:
                logger.debug(f"Image {img['image_number']} ({img['image_theme']}): {len(img['questions'])} questions")
                
            return image_data
            
        except Exception as e:
            logger.error(f"Error parsing document {doc_path}: {e}")
            return []
    
    def convert_to_csv(self, doc_path: str, image_type: str, output_csv_path: str):
        """Convert document to CSV format with metadata appearing only once per image"""
        try:
            # Parse the document
            image_data = self.parse_document(doc_path)
            
            if not image_data:
                logger.warning(f"No data found in document: {doc_path}")
                return
            
            # Prepare CSV data with metadata only in first row per image
            csv_data = []
            
            for img_data in image_data:
                image_number = img_data['image_number']
                theme = img_data['image_theme']
                
                # Generate GitHub link
                github_link = self.generate_github_link(image_number, image_type)
                
                # Get annotator assignment
                annotator = self.get_annotator(image_number, image_type)
                
                # Get the number of questions for this image
                num_questions = len(img_data['questions'])
                
                if num_questions == 0:
                    logger.warning(f"No questions found for Image {image_number}")
                    # Still create one row with empty question data
                    row = {
                        'image_number': image_number,
                        'image_theme': theme,
                        'image_link': github_link,
                        'annotator': annotator,
                        'complexity_level': 1,
                        'question': '',
                        'ground_truth': '',
                        'property_dimension': '',
                        'list_of_objects': ''
                    }
                    csv_data.append(row)
                else:
                    # Create rows for each question - metadata only in first row
                    for i in range(num_questions):
                        if i == 0:  # First question row - include all metadata
                            row = {
                                'image_number': image_number,
                                'image_theme': theme,
                                'image_link': github_link,
                                'annotator': annotator,
                                'complexity_level': i + 1,
                                'question': img_data['questions'][i],
                                'ground_truth': img_data['ground_truth'][i] if i < len(img_data['ground_truth']) else '',
                                'property_dimension': img_data['property_dimension'][i] if i < len(img_data['property_dimension']) else '',
                                'list_of_objects': img_data['list_of_objects'][i] if i < len(img_data['list_of_objects']) else ''
                            }
                        else:  # Subsequent question rows - empty metadata fields
                            row = {
                                'image_number': '',
                                'image_theme': '',
                                'image_link': '',
                                'annotator': '',
                                'complexity_level': i + 1,
                                'question': img_data['questions'][i],
                                'ground_truth': img_data['ground_truth'][i] if i < len(img_data['ground_truth']) else '',
                                'property_dimension': img_data['property_dimension'][i] if i < len(img_data['property_dimension']) else '',
                                'list_of_objects': img_data['list_of_objects'][i] if i < len(img_data['list_of_objects']) else ''
                            }
                        csv_data.append(row)
            
            # Write to CSV
            if csv_data:
                df = pd.DataFrame(csv_data)
                # Column order with complexity_level added
                column_order = ['image_number', 'image_theme', 'image_link', 'annotator', 
                              'complexity_level', 'question', 'ground_truth', 'property_dimension', 'list_of_objects']
                df = df[column_order]
                df.to_csv(output_csv_path, index=False)
                logger.info(f"Successfully created CSV: {output_csv_path} with {len(csv_data)} rows")
                
                # Log annotator distribution for this CSV (only count non-empty annotator entries)
                annotator_counts = df[df['annotator'] != '']['annotator'].value_counts()
                logger.info(f"Annotator distribution in {output_csv_path}: {annotator_counts.to_dict()}")
                
                # Log question distribution per image
                questions_per_image = df[df['image_number'] != ''].groupby('image_number').size()
                logger.info(f"Images with questions: {len(questions_per_image)}")
                
            else:
                logger.warning("No data to write to CSV")
                
        except Exception as e:
            logger.error(f"Error converting document to CSV: {e}")
    
    def create_annotator_template(self, output_file: str):
        """Create a template Excel file for annotator assignments"""
        try:
            # Create sample data structure
            sample_data = []
            
            # Add sample entries for each type and annotator
            annotators = ['AK', 'HK', 'YJ', 'Fl', 'FdH']
            image_types = [('real', 'Real'), ('animated', 'Animated'), ('ai', 'AI_Generated')]
            
            sample_counter = 1
            for annotator in annotators:
                for img_type_code, img_type_name in image_types:
                    for i in range(1, 4):  # 3 sample images per type per annotator
                        sample_data.append({
                            'Annotator': annotator,
                            'Image Type': img_type_code,
                            'Image Number': sample_counter,
                            'Image ID': f"{img_type_code}_{str(sample_counter).zfill(3)}"
                        })
                        sample_counter += 1
            
            df = pd.DataFrame(sample_data)
            df.to_excel(output_file, index=False)
            logger.info(f"Created annotator template: {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating annotator template: {e}")

def main():
    # Configuration
    base_dir = "../merged_data"
    doc_files = {
        "Real": "/var/scratch/ave303/OP_bench/Real_questions_Annotation.docx",  # Adjust these paths as needed
        "Animated": "/var/scratch/ave303/OP_bench/Animated_questions_Annotation.docx",
        "AI_Generated": "/var/scratch/ave303/OP_bench/AI_Generated_questions_Annotation.docx"
    }
    
    # Path to your annotator assignments Excel file - UPDATE THIS PATH!
    annotator_file = "/var/scratch/ave303/OP_bench/updated_image_validation_assignments.xlsx"  # Make sure this file exists and has correct path
    
    # Check if annotator file exists
    if not os.path.exists(annotator_file):
        logger.error(f"Annotator file not found: {annotator_file}")
        logger.error("Please create an Excel file with columns: Annotator, Image Type, Image Number")
        logger.error("Example content:")
        logger.error("  Annotator | Image Type | Image Number")
        logger.error("  AK        | real       | 1")
        logger.error("  AK        | real       | 2")
        logger.error("  John      | animated   | 1")
        return
    
    # Initialize converter
    converter = DocumentToCSVConverter(annotator_file)
    
    # Check if annotator assignments were loaded successfully
    if not converter.annotator_assignments:
        logger.error("No annotator assignments loaded. Please check your Excel file format.")
        return
    
    # Convert documents to CSV
    for image_type, doc_file in doc_files.items():
        if os.path.exists(doc_file):
            output_csv = f"{image_type}_questions.csv"
            logger.info(f"Converting {doc_file} to {output_csv}")
            converter.convert_to_csv(doc_file, image_type, output_csv)
        else:
            logger.warning(f"Document file not found: {doc_file}")
    
    logger.info("Conversion process completed!")
    
    # Print summary of annotator assignments
    if converter.annotator_assignments:
        logger.info("\nAnnotator Assignment Summary:")
        for img_type, assignments in converter.annotator_assignments.items():
            annotator_counts = {}
            for annotator in assignments.values():
                annotator_counts[annotator] = annotator_counts.get(annotator, 0) + 1
            logger.info(f"{img_type}: {annotator_counts}")

if __name__ == "__main__":
    main()