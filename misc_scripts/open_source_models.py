# import json
# import torch
# from PIL import Image
# from tqdm import tqdm
# from pathlib import Path

# class BenchmarkTester:
#     def __init__(self, benchmark_path, data_dir="data"):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         with open(benchmark_path, 'r') as f:
#             self.benchmark = json.load(f)
#         self.data_dir = data_dir
        
#     def format_question(self, question):
#         """Format a question for the model."""
#         return f"Answer the following question about the image and provide only the final count/number: {question['question']}"
    
#     def evaluate_model(self, model_name, model, processor, save_path):
#         results = []
#         print(f"\nEvaluating {model_name}...")
#         print(f"Using device: {self.device}")
        
#         total_images = len(self.benchmark['benchmark']['images'])
#         for idx, image_data in enumerate(tqdm(self.benchmark['benchmark']['images'], desc="Processing images")):
#             print(f"\nProcessing image {idx+1}/{total_images}: {image_data['image_id']}")
#             image_path = Path(self.data_dir) / image_data['path']
#             if not image_path.exists():
#                 print(f"Warning: Image not found at {image_path}")
#                 continue
                
#             image = Image.open(image_path).convert("RGB")
            
#             for question in image_data['questions']:
#                 prompt = self.format_question(question)
#                 print(f"Question: {question['question']}")
                
#                 # Process image and text
#                 inputs = processor(images=image, text=prompt, return_tensors="pt").to(self.device)
                
#                 # Generate answer
#                 with torch.no_grad():
#                     outputs = model.generate(**inputs, max_new_tokens=50)
#                 answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
#                 print(f"Model answer: {answer}")
#                 print(f"Ground truth: {question['answer']}")
                
#                 results.append({
#                     "image_id": image_data["image_id"],
#                     "image_type": image_data["image_type"],
#                     "question_id": question["id"],
#                     "question": question["question"],
#                     "ground_truth": question["answer"],
#                     "model_answer": answer,
#                     "property_category": question["property_category"]
#                 })
        
#         print(f"\nSaving results to {save_path}")
#         # Save results
#         with open(save_path, 'w') as f:
#             json.dump(results, f, indent=4)
        
#         return results

# # Example usage for different models:

# def test_fuyu():
#     from transformers import FuyuProcessor, FuyuForCausalLM
    
#     processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
#     model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b").to("cuda")
    
#     tester = BenchmarkTester("benchmark.json")
#     results = tester.evaluate_model("fuyu-8b", model, processor, "results_fuyu.json")

# def test_blip2():
#     from transformers import Blip2Processor, Blip2ForConditionalGeneration
    
#     processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
#     model = Blip2ForConditionalGeneration.from_pretrained(
#         "Salesforce/blip2-flan-t5-xxl",
#         torch_dtype=torch.float16
#     ).to("cuda")
    
#     tester = BenchmarkTester("benchmark.json")
#     results = tester.evaluate_model("blip2", model, processor, "results_blip2.json")

# if __name__ == "__main__":
#     print("Starting model evaluation...")
    
#     try:
#         print("Testing Fuyu-8b model...")
#         fuyu_results = test_fuyu()
#         print("Fuyu-8b testing completed")
        
#         print("\nTesting BLIP-2 model...")
#         blip2_results = test_blip2()
#         print("BLIP-2 testing completed")
        
#     except Exception as e:
#         print(f"An error occurred: {str(e)}") 






import json
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import gc  # for garbage collection

class BenchmarkTester:
    def __init__(self, benchmark_path, data_dir="."):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with open(benchmark_path, 'r') as f:
            self.benchmark = json.load(f)
        self.data_dir = data_dir
    
    def format_question(self, question):
        """Format a question for the model."""
        # return (
        #     f"Answer the following question about the image. You MUST respond with a number "
        #     f"followed by a list of objects in square brackets that justify your count. "
        #     f"Example format: '5 [chair, table, desk, shelf, cabinet]'\n\n"
        #     f"Question: {question['question']}"
        # )
        # return f"Answer the following question about the image with a number followed by a list of objects in square brackets that justify your count: {question['question']}"
        #return f"Answer the following question about the image with a number followed by the list of objects that led to this count within square brackets: {question['question']}"
        return f"{question['question']} \nPlease provide a number followed by the list of objects within square brackets as your answer."

    def clean_answer(self, answer):
        """Clean the model output to extract just the number."""
        # Remove any text that's not a number
        # import re
        # numbers = re.findall(r'\d+', answer)
        # if numbers:
        #     return numbers[0]  # Return the first number found
        # return answer
        """Extract number and reasoning from the model's answer."""
        # Try to extract number and reasoning using regex
        import re
        pattern = r'(\d+)\s*\[(.*?)\]'
        match = re.search(pattern, answer)
        
        if match:
            number = match.group(1)
            objects = [obj.strip() for obj in match.group(2).split(',')]
            return {
                "count": number,
                "reasoning": objects
            }
        else:
            # Fallback if format isn't matched
            numbers = re.findall(r'\d+', answer)
            return {
                "count": numbers[0] if numbers else "0",
                "reasoning": []
            }
    
    def evaluate_model(self, model_name, model, processor, save_path, start_idx=0, batch_size=5):
        results = []
        print(f"\nEvaluating {model_name}...")
        print(f"Using device: {self.device}")
        # Force garbage collection before starting
        gc.collect()
        torch.cuda.empty_cache()

        try:
            images = self.benchmark['benchmark']['images'][start_idx:start_idx + batch_size]
            total_images = len(images)
            
            for idx, image_data in enumerate(tqdm(images, desc="Processing images")):
                try:
                    print(f"\nProcessing image {idx+1}/{total_images}: {image_data['image_id']}")
                    image_path = Path(image_data['path'])
                    if not image_path.exists():
                        print(f"Warning: Image not found at {image_path}")
                        continue
                    
                    # Load and preprocess image
                    image = Image.open(image_path).convert("RGB")
                    image_results = []  # Store results for current image
                    
                    for question in image_data['questions']:
                        try:
                            prompt = self.format_question(question)
                            print(f"Question: {question['question']}")
                            
                            # Clear cache before processing each question
                            torch.cuda.empty_cache()
                            
                            # Process image and text
                            inputs = processor(images=image, text=prompt, return_tensors="pt").to(self.device)
                            
                            # Generate answer with better settings
                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=200,  # Increased from 10 to 200
                                    num_beams=1,        # Added beam search
                                    length_penalty=1.0,  # Encourage slightly longer outputs
                                    temperature=0.7,     # Add some randomness
                                    do_sample=True,      # Enable sampling
                                    pad_token_id=processor.tokenizer.eos_token_id
                                )
                            answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                            cleaned_answer = self.clean_answer(answer)

                            # Try to extract number and reasoning
                            # import re
                            # pattern = r'(\d+)\s*\[(.*?)\]'
                            # match = re.search(pattern, answer)
                            
                            # if match:
                            #     number = match.group(1)
                            #     objects = [obj.strip() for obj in match.group(2).split(',')]
                            #     clean_answer = {
                            #         "count": number,
                            #         "reasoning": objects
                            #     }
                            # else:
                            #     # Fallback if format isn't matched
                            #     numbers = re.findall(r'\d+', answer)
                            #     clean_answer = {
                            #         "count": numbers[0] if numbers else "0",
                            #         "reasoning": []
                            #     }
                            
                            image_results.append({
                                "image_id": image_data["image_id"],
                                "image_type": image_data["image_type"],
                                "question_id": question["id"],
                                "question": question["question"],
                                "ground_truth": question["answer"],
                                "model_answer": cleaned_answer["count"],
                                "model_reasoning": cleaned_answer["reasoning"],
                                "raw_answer": answer,  # Keep raw answer for debugging
                                "property_category": question["property_category"]
                            })
                            
                            # Clear memory
                            del outputs, inputs
                            torch.cuda.empty_cache()
                            
                        except Exception as e:
                            print(f"Error processing question: {str(e)}")
                            continue
                    
                    # Add results from this image
                    results.extend(image_results)
                    
                    # Save intermediate results only every 2 images or if it's the last image
                    if (idx + 1) % 2 == 0 or idx == total_images - 1:
                        with open(f"{save_path}_checkpoint.json", 'w') as f:
                            json.dump(results, f, indent=4)
                            
                except Exception as e:
                    print(f"Error processing image {image_data['image_id']}: {str(e)}")
                    continue
            
            # Save final results
            if results:
                with open(save_path, 'w') as f:
                    json.dump(results, f, indent=4)
            
        except Exception as e:
            print(f"An error occurred during evaluation: {str(e)}")
            if results:
                with open(f"{save_path}_error_state.json", 'w') as f:
                    json.dump(results, f, indent=4)
        
        return results

def test_fuyu(batch_size=5, start_idx=0):
    from transformers import FuyuProcessor, FuyuForCausalLM
    import torch
    
    print("Loading Fuyu-8b model and processor...")
    try:
        # Load with more aggressive memory optimization
        processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
        model = FuyuForCausalLM.from_pretrained(
            "adept/fuyu-8b",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        # Optional: Enable memory efficient attention
        if hasattr(model.config, 'use_memory_efficient_attention'):
            model.config.use_memory_efficient_attention = True
        
        tester = BenchmarkTester("benchmark.json")
        results = tester.evaluate_model(
            "fuyu-8b", 
            model, 
            processor, 
            "results_fuyu.json",
            start_idx=start_idx,
            batch_size=batch_size
        )
        
        # Clean up
        del model
        del processor
        torch.cuda.empty_cache()
        gc.collect()
        
        return results
    
    except Exception as e:
        print(f"Error in test_fuyu: {str(e)}")
        return None
    
def test_blip2(batch_size=5, start_idx=0):
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    print("Loading Blip2 model and processor...")
    try:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl", revision="main")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xxl",
            torch_dtype=torch.float16,
            revision="main"
        ).to("cuda")

         # Optional: Enable memory efficient attention
        if hasattr(model.config, 'use_memory_efficient_attention'):
            model.config.use_memory_efficient_attention = True
        
        # Ensure model is in eval mode
        model.eval()

        tester = BenchmarkTester("benchmark.json")
        results = tester.evaluate_model(
            "blip2", 
            model, 
            processor, 
            "results_blip2.json",
            start_idx=start_idx,
            batch_size=batch_size
        )
        
        # Clean up
        del model
        del processor
        torch.cuda.empty_cache()
        gc.collect()
        
        return results
    
    except Exception as e:
        print(f"Error in test_blip2: {str(e)}")
        return None

if __name__ == "__main__":
    print("Starting model evaluation...")
    
    try:
        # Try with even smaller batch first
        print("Testing Fuyu-8b/BLIP2 model with first 3 images...")
        # fuyu_results = test_fuyu(batch_size=1, start_idx=0)
        blip2_results = test_fuyu(batch_size=1, start_idx=0)
        
        if blip2_results is not None:
            print("Initial test successful!")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")