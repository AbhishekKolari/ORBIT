import pandas as pd
import json

def convert_csv_to_json(csv_file_path, output_file_path, evaluator_name):
    """
    Convert CSV files to JSON format matching the existing results structure.
    
    Args:
        csv_file_path: Path to the input CSV file
        output_file_path: Path to the output JSON file
        evaluator_name: Name of the evaluator (for naming)
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    
    # Print column names for debugging
    print(f"Columns in {csv_file_path}: {list(df.columns)}")
    
    # Initialize the JSON data list
    json_data = []
    
    # Process each row
    for index, row in df.iterrows():
        # Skip empty rows
        if pd.isna(row['image number']) or pd.isna(row['questions']):
            continue
        
        # Convert image number to image_id format
        image_number = int(row['image number'])
        image_id = f"image{image_number:02d}"
        
        # Convert complexity to question_id format
        complexity = int(row['complexity'])
        question_id = f"Q{complexity}"
        
        # Convert image type to uppercase (matching existing format)
        image_type = row['image type'].upper()
        
        # Get the question text
        question = row['questions']
        
        # Get ground truth and count
        ground_truth = str(int(row['ground truth']))
        model_answer = str(int(row['count']))
        
        # Process comments as model_reasoning
        comments = row['comments (optional)']
        if pd.isna(comments) or comments == '':
            model_reasoning = [""]
        else:
            # Split comments by commas and clean up
            reasoning_items = [item.strip() for item in str(comments).split(',') if item.strip()]
            model_reasoning = reasoning_items if reasoning_items else [""]
        
        # Create raw_answer (blank as requested)
        raw_answer = ""
        
        # Get property category
        property_category = row['property dimension'].lower()
        
        # Create the JSON entry
        json_entry = {
            "image_id": image_id,
            "image_type": image_type,
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "model_reasoning": model_reasoning,
            "raw_answer": raw_answer,
            "property_category": property_category
        }
        
        json_data.append(json_entry)
    
    # Write to JSON file
    with open(output_file_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    print(f"✓ Converted {csv_file_path} to {output_file_path}")
    print(f"✓ Total entries: {len(json_data)}")
    
    return json_data

# Convert both CSV files
print("Converting CSV files to JSON format...")

# Convert Stefano's evaluation
stefano_data = convert_csv_to_json(
    'Human Evaluation (Stefano).csv',
    'ORBIT_results/human_evaluation_stefano.json',
    'Stefano'
)

# Convert Mehdi's evaluation
mehdi_data = convert_csv_to_json(
    'Human Evaluation (Mehdi).csv',
    'ORBIT_results/human_evaluation_mehdi.json',
    'Mehdi'
)

print("\nConversion complete!")
print("Generated files:")
print("- ORBIT_results/human_evaluation_stefano.json")
print("- ORBIT_results/human_evaluation_mehdi.json")

# Verify the conversion by loading and checking a few entries
def verify_conversion(json_file_path, evaluator_name):
    """Verify the conversion by loading and displaying a few sample entries."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    print(f"\nVerification for {evaluator_name}:")
    print(f"Total entries: {len(data)}")
    
    if len(data) > 0:
        print("\nSample entry:")
        sample = data[0]
        for key, value in sample.items():
            print(f"  {key}: {value}")
    
    return data

# Verify both conversions
stefano_verified = verify_conversion('ORBIT_results/human_evaluation_stefano.json', 'Stefano')
mehdi_verified = verify_conversion('ORBIT_results/human_evaluation_mehdi.json', 'Mehdi')