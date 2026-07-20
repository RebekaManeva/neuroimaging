import os
import json
import torch
import random

random.seed(42)

for _attr in ["float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu"]:
    if not hasattr(torch, _attr):
        setattr(torch, _attr, torch.float32)

try:
    import transformers.masking_utils as _mu
    for _fn_name in ("create_sliding_window_causal_mask", "create_causal_mask"):
        if hasattr(_mu, _fn_name):
            _orig = getattr(_mu, _fn_name)
            def _make_patched(orig):
                def _patched(*args, **kwargs):
                    kwargs.pop("or_mask_function", None)
                    kwargs.pop("and_mask_function", None)
                    return orig(*args, **kwargs)
                return _patched
            setattr(_mu, _fn_name, _make_patched(_orig))
except ModuleNotFoundError:
    pass

import argparse
from PIL import Image
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)

LABEL_REGION_DESC = {
    "whole_tumor": "whole tumor (WT)"
}


def make_prompt(modality: str, label: str) -> str:
    seq_map = {"flair": "FLAIR", "t1": "T1", "t1ce": "T1C+", "t2": "T2"}
    seq = seq_map.get(modality, "MRI")
    region = LABEL_REGION_DESC.get(label, label)
    return f"""You are an expert neuroradiologist analyzing a brain MRI slice ({seq} sequence).
Locate the {region} in this image and return the tumor boundary as a polygon.

Return ONLY a valid JSON object inside <JSON> and </JSON> tags.

<JSON>
{{
  "has_tumor": true,
  "polygon_norm": [[x1,y1], [x2,y2], ...],
  "confidence": 0.0-1.0
}}
</JSON>

Rules:
- polygon_norm: list of [x,y] points, floats in [0,1], describing the tumor boundary in order
- has_tumor is always true for this dataset
- confidence: your certainty that the polygon covers the lesion (0.0-1.0)
- If no lesion is visible in this slice, return polygon_norm=[] and has_tumor=false
"""


def make_target(item: dict) -> str:
    has_tumor = item["has_tumor"]
    polygon = item.get("polygon_norm") if has_tumor else []
    obj = {
        "has_tumor": has_tumor,
        "polygon_norm": polygon if polygon else [],
        "confidence": round(random.uniform(0.75, 0.95), 2) if has_tumor else 0.0,
    }
    return f"<JSON>\n{json.dumps(obj, indent=2)}\n</JSON>"


def find_last_checkpoint(output_dir: str):
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    last = os.path.join(output_dir, checkpoints[-1])
    print(f"Found checkpoint for resume: {last}")
    return last


def train(args):
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = AutoConfig.from_pretrained(args.model_id, token=hf_token)
    config.use_sliding_window = False
    config._attn_implementation = "eager"
    if hasattr(config, "sliding_window"):
        config.sliding_window = None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model: {args.model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        config=config,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=hf_token,
    )

    processor = AutoProcessor.from_pretrained(args.model_id, token=hf_token)

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    print(f"\nLoading datasets (label={args.label_col})...")
    train_dataset = load_dataset("json", data_files=args.train_path, split="train")
    val_dataset   = load_dataset("json", data_files=args.val_path,   split="train")

    label_to_modality = {"whole_tumor": "flair"}
    target_modality = label_to_modality.get(args.label_col)

    train_dataset = train_dataset.filter(
        lambda x: x["label"] == args.label_col
            and x["modality"] == target_modality
            and (
                (x["has_tumor"] == False and random.random() < 0.30)
                or (
                    x["has_tumor"] == True and
                    x.get("polygon_norm") is not None and
                    len(x.get("polygon_norm", [])) >= 3 and
                    (x["bbox_xyxy_norm"][2] - x["bbox_xyxy_norm"][0]) *
                    (x["bbox_xyxy_norm"][3] - x["bbox_xyxy_norm"][1]) >= 0.005
                )
            )
    )
    val_dataset = val_dataset.filter(
        lambda x: x["label"] == args.label_col
            and x["modality"] == target_modality
            and (
                (x["has_tumor"] == False and random.random() < 0.30)
                or (
                    x["has_tumor"] == True and
                    x.get("polygon_norm") is not None and
                    len(x.get("polygon_norm", [])) >= 3 and
                    (x["bbox_xyxy_norm"][2] - x["bbox_xyxy_norm"][0]) *
                    (x["bbox_xyxy_norm"][3] - x["bbox_xyxy_norm"][1]) >= 0.005
                )
            )
    )
    print(f"  After filtering: {len(train_dataset)} train, {len(val_dataset)} val records")

    def collate_fn(batch):
        formatted_prompts = []
        assistant_responses = []
        images = []

        for item in batch:
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": make_prompt(item["modality"], item["label"])},
                    ],
                }
            ]
            full_messages = user_messages + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": make_target(item)}],
                }
            ]

            full_prompt = processor.apply_chat_template(full_messages, add_generation_prompt=False)
            prompt_only = processor.apply_chat_template(user_messages, add_generation_prompt=True)

            formatted_prompts.append(full_prompt)
            assistant_responses.append(prompt_only)

            file_name = os.path.basename(item["image_path"].replace("\\", "/"))
            patient_id = file_name.split("__")[0]
            cluster_images_dir = "/home/hpc/users/ml_models/elena.nikolovska/Medgemma_Lora/prepared_dataset/images"
            cluster_img_path = os.path.join(cluster_images_dir, patient_id, file_name)

            img = Image.open(cluster_img_path).convert("RGB")
            images.append(img)

        inputs = processor(
            text=formatted_prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        labels = inputs["input_ids"].clone()

        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        for idx, prompt_only_text in enumerate(assistant_responses):
            prompt_ids = processor.tokenizer(
                prompt_only_text,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0]
            prompt_len = prompt_ids.shape[0]
            mask_len = min(prompt_len, labels.shape[1])
            labels[idx, :mask_len] = -100

        inputs["labels"] = labels
        return inputs

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        per_device_eval_batch_size=1,
        num_train_epochs=args.epochs,
        logging_steps=10,
        eval_strategy="steps",         
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,                  
        load_best_model_at_end=True,     
        metric_for_best_model="eval_loss",  
        greater_is_better=False,         
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        save_total_limit=3,              
    )

    class LoraTrainer(Trainer):
        def save_model(self, output_dir=None, _internal_call=False):
            out = output_dir or self.args.output_dir
            os.makedirs(out, exist_ok=True)
            self.model.save_pretrained(out, safe_serialization=False)

    trainer = LoraTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  
    )

    last_checkpoint = find_last_checkpoint(args.output_dir)
    if last_checkpoint:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        print("No previous checkpoint found - starting training from scratch.")

    print("Starting training loop...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_dir = os.path.join(args.output_dir, "final_lora_weights")
    model.save_pretrained(final_dir, safe_serialization=False)
    processor.save_pretrained(final_dir)
    print(f"Training complete. Weights saved to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",    type=str,   default="google/medgemma-1.5-4b-it")
    parser.add_argument("--train_path",  type=str,   required=True)
    parser.add_argument("--val_path",    type=str,   required=True)
    parser.add_argument("--output_dir",  type=str,   default="./output_medgemma")
    parser.add_argument("--label_col",   type=str,   default="whole_tumor",
                        choices=["whole_tumor"])
    parser.add_argument("--batch_size",  type=int,   default=1)
    parser.add_argument("--grad_accum",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--epochs",      type=int,   default=2)  
    args = parser.parse_args()
    train(args)