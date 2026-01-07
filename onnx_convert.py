from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_path = "./my_safety_model"
bigger_model = "distilbert-base-uncased"
output_path = "./onnx_output"

print(f"Loading trained model from {model_path}...")

model = ORTModelForSequenceClassification.from_pretrained(
    model_path,
    export=True
)

print(f"Fetching standard tokenizer from {bigger_model}...")
tokenizer = AutoTokenizer.from_pretrained(bigger_model)

print(f"Saving ONNX model to {output_path}...")
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("SUCCESS! Your model is now ONNX compatible.")