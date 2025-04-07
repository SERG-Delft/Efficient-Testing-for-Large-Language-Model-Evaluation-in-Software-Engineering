import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import os

# ----------------------
# Utility: Detect device
# ----------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# --------------------------------------
# Utility: Compute confidence & entropy
# --------------------------------------
def compute_confidence_and_entropy(probabilities):
    confidence = torch.max(probabilities).item()
    p = np.clip(confidence, 1e-12, 1.0)
    q = 1 - p
    entropy = -p * np.log2(p) - q * np.log2(q)
    return confidence, entropy

# ----------------------
# Inference Main Method
# ----------------------
def run_inference(args):
    device = get_device()
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    model.eval()

    # Load dataset
    df = pd.read_csv(args.input_file)
    is_clone_task = {"code1", "code2"}.issubset(df.columns)

    results = []
    for _, row in df.iterrows():
        if is_clone_task:
            # Clone Detection: use two input code columns
            inputs = tokenizer(row["code1"], row["code2"], return_tensors="pt", truncation=True, padding=True, max_length=512)
        else:
            # Vulnerability Detection: use one input code column
            inputs = tokenizer(row["code"], return_tensors="pt", truncation=True, padding=True, max_length=512)

        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze()

        pred = torch.argmax(probs).item()
        confidence, entropy = compute_confidence_and_entropy(probs.cpu().numpy())

        results.append({
            "id": row.get("id", None),
            "true_label": row.get("label", None),
            "predicted_label": pred,
            "confidence": confidence,
            "entropy": entropy
        })

    pd.DataFrame(results).to_csv(args.output_file, index=False)
    print(f"Output saved to {args.output_file}")

# ----------------------
# Script Entry Point
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--input_file", required=True, help="CSV input file path")
    parser.add_argument("--output_file", required=True, help="CSV output file path")

    args = parser.parse_args()
    run_inference(args)
