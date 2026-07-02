import json
import os
import glob
from pathlib import Path
from tkinter.constants import TRUE

import typer
import wandb
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    output_dir: str = typer.Option(..., "--output_dir"),
    model_path: str = typer.Option(..., "--model_path"),
    dataset_path: str = typer.Option(..., "--dataset_path"),
    learning_rate: float = typer.Option(2e-4, "--learning_rate"),
    num_train_epochs: int = typer.Option(1, "--num_train_epochs"),
    per_device_train_batch_size: int = typer.Option(1, "--per_device_train_batch_size"),
    gradient_accumulation_steps: int = typer.Option(8, "--gradient_accumulation_steps"),
    max_seq_len: int = typer.Option(2048, "--max_seq_len"),
    warmup_ratio: float = typer.Option(0.03, "--warmup_ratio"),
    lora_r: int = typer.Option(16, "--lora_r"),
    lora_alpha: int = typer.Option(32, "--lora_alpha"),
    lora_dropout: float = typer.Option(0.05, "--lora_dropout"),
    report_to: str = typer.Option(None, "--report_to"),
    wandb_project: str = typer.Option("llama3-8b-lora", "--wandb_project"),
    save_steps: int = typer.Option(500, "--save_steps"),
    logging_steps: int = typer.Option(10, "--logging_steps"),
    bf16: bool = typer.Option(True, "--bf16", help="Enable bfloat16"),
    val_size: int = typer.Option(1024, "--val_size", help="Validation set size (approx)"),
):

    if report_to == "wandb":
        wandb.init(project=wandb_project, name=os.path.basename(output_dir))

    # Load tokenizer/model with 4-bit to keep memory low
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    # Will only be used if quantization_config is not set by the model itself.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    device_map = _get_device_map_with_logging()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        dtype="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    # DDP + LoRA stability
    try:
        model.config.use_cache = False
    except Exception:
        pass

    # Dataset: simple instruction -> prompt formatting
    cot_files = sorted(glob.glob(f"{dataset_path}/cot-*.parquet"))
    if not cot_files:
        raise FileNotFoundError(f"No files found matching {dataset_path}/cot-*.parquet")
    print(f"Loading parquet dataset from files {dataset_path}")
    raw = load_dataset("parquet", data_files={"train": cot_files})

    def format_example(ex):
        if "instruction" in ex and "output" in ex:
            prompt = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
        elif "text" in ex:
            prompt = ex["text"]
        else:
            prompt = str(ex)
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        return tokenized

    # Create small validation split
    # TODO: Remove after debugging
    base_train = raw["train"].select(range(5000))
    print(base_train)
    n_total = len(base_train)
    # Aim for a small subset: min(val_size, 1% of data but at least 1)
    desired_val = min(val_size, max(1, int(0.01 * n_total))) if n_total > 0 else 0
    split = base_train.train_test_split(test_size=desired_val, shuffle=True, seed=42)
    train_raw = split["train"]
    eval_raw = split["test"]

    train_ds = train_raw.map(format_example, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(format_example, remove_columns=eval_raw.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=bf16,
        report_to=[report_to] if report_to else [],
        optim="paged_adamw_8bit",
        eval_strategy="steps",
        eval_steps=logging_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

def _get_device_map_with_logging():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    is_dist = False
    try:
        is_dist = dist.is_available() and dist.is_initialized()
    except Exception:
        pass

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    # If distributed launch is used, LOCAL_RANK is set. Pin to that GPU.
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        current_device = local_rank
    else:
        current_device = torch.cuda.current_device()
        
    num_devices = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(current_device)
    device_map = {"": current_device}
    print(
        f"[RANK {rank} | LOCAL_RANK {local_rank} | WORLD_SIZE {world_size} | DIST {is_dist}] "
        f"CUDA_VISIBLE_DEVICES={cuda_visible} | cuda.count={num_devices} | current_device={current_device} ({device_name})\n"
        f"Using device_map={device_map} for quantized model loading."
    )
    return device_map

if __name__ == "__main__":
    app()

