import time
start_time0 = time.time()
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
from pathlib import Path
from PIL import Image
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from collections import Counter
import torch
import argparse
import os
import urllib.request
import matplotlib.pyplot as plt
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DTYPE = torch.float16

print("Loading model …")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",          # balances across 2×T4
    # ── optional: quantise to 4-bit to save ~7 GB if VRAM is tight ──
    # load_in_4bit=True,
    # bnb_4bit_compute_dtype=torch.bfloat16,
    local_files_only=True
)
model.eval()

print("Loading processor …")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("✓ Model ready")
print(f"Total params : {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B")

# ─────────────────────────────────────────────
#  Prompt engineering
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert exam solver specialising in deep learning, machine learning, 
mathematics, physics, chemistry, biology, and logical reasoning — with particular depth in:
- Neural network architectures (CNNs, RNNs, Transformers, GANs, VAEs)
- Training techniques (backpropagation, optimizers, regularization, batch norm)
- Loss functions, metrics, and evaluation strategies
- Supervised, unsupervised, and reinforcement learning
- Statistics, probability, and linear algebra as applied to ML

STRICT OUTPUT FORMAT:
- Options are numbered 1, 2, 3, 4 (NOT letters A/B/C/D)
- Your LAST line MUST be exactly one of:
    ANSWER: 1   or   ANSWER: 2   or   ANSWER: 3   or   ANSWER: 4   or   ANSWER: 5
- Output ANSWER: 5 if you are not confident — skipping is safer than guessing wrong
- Do NOT output any other value — it counts as hallucination and costs −1 point
- Do NOT write anything after the ANSWER line

DECISION RULE:
- Answer (1/2/3/4) only if confidence ≥ 50%
- If confidence is below 50% after elimination, output ANSWER: 5"""

USER_PROMPT = (
    "Solve the MCQ in this image using the following steps:\n\n"
    "Step 1 — Question type: What kind of problem is this?\n"
    "         (calculation / conceptual / code / diagram / other)\n\n"
    "Step 2 — List options: Write out all options exactly as numbered (1/2/3/4).\n\n"
    "Step 3 — Core principle: State the formula, theorem, or definition needed.\n"
    "         If making any assumption, state it explicitly here.\n"
    "         Use the SAME formula for EVERY layer/step — never switch to a shortcut.\n\n"
    "Step 4 — Evaluate each option: Apply the formula numerically for each layer.\n"
    "         Show: formula → substitution → result, for every single step.\n"
    "         Then accept or eliminate each option with a specific reason.\n\n"
    "Step 5 — Verify: Cross-check your final result against ALL options.\n"
    "         If your result doesn't match any option exactly, recompute from Step 3.\n\n"
    "Step 6 — Confidence check: Rate your confidence (0–100%).\n"
    "         ≥ 50% → answer with 1/2/3/4\n"
    "         < 50% → output 5 (skip, no penalty)\n\n"
    "Final line must be: ANSWER: <1 or 2 or 3 or 4 or 5>"
)



def preprocess_image(image_input, target_min_side: int = 1024) -> Image.Image:
    """
    Upscale small images, boost contrast, and sharpen before passing to the model.
    Qwen2.5-VL benefits from clear, high-contrast input.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported type: {type(image_input)}")

    # Upscale if too small — model resolution is key for text/formula legibility
    # w, h = img.size
    # min_side = min(w, h)
    # if min_side < target_min_side:
    #     scale = target_min_side / min_side
    #     img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = img.resize((448, 448), Image.LANCZOS)

    # Boost contrast slightly — helps with faint printed text
    img = ImageEnhance.Contrast(img).enhance(1.3)

    # Sharpen — helps with slightly blurry scans
    img = img.filter(ImageFilter.SHARPEN)

    return img

def build_messages(image_input):
    img = preprocess_image(image_input)   # ← add this line
    img_payload = {"type": "image", "image": img}
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [img_payload, {"type": "text", "text": USER_PROMPT}]},
    ]


def extract_answer(text: str) -> str:
    """
    Multi-stage parser — handles every common model output format.
    Priority order: canonical tag > last-line letter > bolded letter > first letter.
    """
    VALID = set("12345")

    # Stage 1: Canonical ANSWER: X tag (case-insensitive, flexible spacing)
    match = re.search(r"(?:ANSWER|ANS|Answer)\s*[:\-]\s*\(?([1-5])\)?", text)
    if match:
        return match.group(1).upper()

    # Stage 2: "The answer is X" / "Option X is correct" / "Therefore X"
    match = re.search(
        r"(?:the\s+(?:correct\s+)?answer\s+is|therefore[,\s]+(?:the\s+answer\s+is)?|option)\s*[:\-]?\s*\(?([1-5])\)?",
        text, re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # Stage 3: Last non-empty line that is just a letter (with optional brackets/period)
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        m = re.match(r"^\(?([1-5])\)?[.):]?\s*$", line)
        if m and m.group(1).upper() in VALID:
            return m.group(1).upper()

    # Stage 4: Last occurrence of a standalone letter in the entire output
    matches = re.findall(r"\b([1-5])\b", text)
    if matches:
        return matches[-1].upper()

    return "UNKNOWN"


@torch.inference_mode()
def solve_mcq(image_input, max_new_tokens: int = 1024,
              temperature: float = 0.05, verbose: bool = True) -> dict:

    messages = build_messages(image_input)

    # Prepare text input
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Prepare image input
    image_inputs, video_inputs = process_vision_info(messages)

    # Tokenize everything
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0.0),
        repetition_penalty=1.05,
        top_p=0.9,
        top_k=20,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_output = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    answer = extract_answer(raw_output)

    if verbose:
        sep = "─" * 60
        print(sep)
        print("REASONING TRACE")
        print(sep)
        print(raw_output)
        print(sep)
        print(f"\n✅  FINAL ANSWER : {answer}\n")

    return {
        "answer": answer,
        # "reasoning": raw_output,
        # "raw_output": raw_output,
    }

print("✓ Solver function ready")



def solve_mcq_voting(image_input, n_votes: int = 5,
                     temperature: float = 0.4, verbose: bool = True) -> dict:
    votes, reasonings = [], []

    for i in range(n_votes):
        result = solve_mcq(image_input, temperature=temperature, verbose=False)
        votes.append(result["answer"])
        reasonings.append(result["reasoning"])
        if verbose:
            print(f"Vote {i+1}/{n_votes} → {result['answer']}")

    tally = Counter(votes)

    # Detect tie — run a tiebreaker at temperature=0 (greedy)
    top_count = tally.most_common(1)[0][1]
    top_answers = [a for a, c in tally.items() if c == top_count and a != "UNKNOWN"]

    if len(top_answers) > 1:
        if verbose:
            print(f"⚠️  Tie between {top_answers} — running greedy tiebreaker")
        tiebreaker = solve_mcq(image_input, temperature=0.0, verbose=False)
        winner = tiebreaker["answer"]
    else:
        # Exclude UNKNOWN from winning unless it's all we have
        non_unknown = [(a, c) for a, c in tally.most_common() if a != "UNKNOWN"]
        winner = non_unknown[0][0] if non_unknown else "UNKNOWN"

    best_reasoning = reasonings[votes.index(winner)] if winner in votes else reasonings[0]

    if verbose:
        print(f"\n📊 Vote tally : {dict(tally)}")
        print(f"✅  FINAL ANSWER (majority) : {winner}")

    return {"answer": winner}

print("✓ Voting solver ready")



def run_batch(
    csv_path: str,
    image_col: str = "image_path",
    id_col: str = "id",
    output_path: str = "submissions.csv",
    use_voting: bool = False,
    n_votes: int = 3,
):
    """
    Process every row in a CSV and produce a submission file.

    Parameters
    ----------
    csv_path    : path to input CSV (columns: id, image_path)
    image_col   : column containing image file paths
    id_col      : column containing question IDs
    output_path : where to save submission CSV
    use_voting  : enable self-consistency voting (slower, more accurate)
    n_votes     : votes per question when use_voting=True
    """
    df = pd.read_csv(csv_path)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Solving MCQs"):
        qid   = row[id_col]
        img_p = row[image_col]
        try:
            if use_voting:
                out = solve_mcq_voting(img_p, n_votes=n_votes, verbose=False)
            else:
                out = solve_mcq(img_p, verbose=False)

            # ── Auto-retry with voting if we got UNKNOWN ──────────────────
            if out["answer"] == "UNKNOWN":
                print(f"[RETRY] id={qid} — UNKNOWN, switching to voting")
                out = solve_mcq_voting(img_p, n_votes=3, temperature=0.3, verbose=False)

            results.append({"id": qid, "answer": out["answer"]})
        except Exception as e:
            print(f"[ERROR] id={qid} — {e}")
            results.append({"id": qid, "answer": "UNKNOWN"})

print("✓ Batch solver ready")


# ─────────────────────────────────────────────
#  Main Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve MCQ questions using Qwen2.5-VL")
    parser.add_argument(
        "--test_dir",
        type=str,
        default="testing_project",
        help="Directory containing test images (default: all_images)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Output CSV file path (default: submission.csv)"
    )
    parser.add_argument(
        "--voting",
        action="store_true",
        help="Enable voting-based predictions (slower, more accurate)"
    )
    parser.add_argument(
        "--n_votes",
        type=int,
        default=3,
        help="Number of votes per question (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Show the question
    test_dir = args.test_dir
    results = []

    test_csv_path = os.path.join(test_dir, "test.csv")
    df=pd.read_csv(test_csv_path)
    img_paths = df['image_name'].tolist()
    
    end_time0 = time.time()
    
    start_time = time.time()
        

    
    for x in tqdm(img_paths):

        IMAGE_PATH = os.path.join(str(test_dir), "images", x + ".png")

        try:
            img = Image.open(IMAGE_PATH).convert("RGB")

            # ---- Solve ----
            result = solve_mcq(IMAGE_PATH, verbose=False)

            # Store result
            results.append({
                "id":x,
                "image_name": x,
                "option": result['answer']
            })

        except Exception as e:
            print(f"Error processing {x}: {e}")
            results.append({
                "id":x,
                "image_name": x,
                "option": None
            })

    end_time = time.time()
    total_time0 = end_time0 - start_time0
    total_time = end_time - start_time

    # ---- Save CSV ----
    df = pd.DataFrame(results)
    output_dir=os.path.join(test_dir, "submission.csv")
    df.to_csv(output_dir, index=False)

    # ---- Print time ----
    print(f"\nTotal Time before the loop: {total_time0:.2f} seconds")
    print(f"\nTotal Time of the loop: {total_time:.2f} seconds")
    print(f"Avg Time per Image: {total_time / len(results):.2f} seconds")