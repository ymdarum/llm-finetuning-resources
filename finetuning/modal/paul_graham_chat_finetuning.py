# type: ignore
# Script I used for my finetuning experimentation on modal
# https://modal.com/docs

import modal
from typing import Any
from datasets import Dataset
import os

app = modal.App("pg_finetuning")

unsloth_image = modal.Image.from_dockerfile("finetuning/modal/Dockerfile")


def create_model(
    base_model: str, max_seq_length: int, train_embeddings: bool
) -> tuple[Any, Any]:

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    common_peft_params = {
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
        "use_rslora": False,
        "loftq_config": None,
    }

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    if train_embeddings:
        target_modules.extend(["embed_tokens", "lm_head"])

    model = FastLanguageModel.get_peft_model(
        model, target_modules=target_modules, **common_peft_params
    )

    return model, tokenizer


def prepare_dataset(hf_dataset_name: str, tokenizer: Any) -> Dataset:
    from datasets import load_dataset

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {
            "text": texts,
        }

    pass

    dataset = load_dataset(hf_dataset_name, split="train")
    return dataset.map(formatting_prompts_func, batched=True)


@app.function(
    gpu="T4",
    image=unsloth_image,
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("hugginface-secret")],
)
def train(
    base_model: str,
    hf_dataset_name: str,
    hf_save_gguf_name: str,
    epochs: int,
    train_embeddings: bool,
    max_seq_length: int,
    quantization_method: str,
) -> str:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    import torch

    # Create model and tokenizer
    model, tokenizer = create_model(
        base_model=base_model,
        max_seq_length=max_seq_length,
        train_embeddings=train_embeddings,
    )

    # Prepare dataset
    dataset = prepare_dataset(hf_dataset_name=hf_dataset_name, tokenizer=tokenizer)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,
            num_train_epochs=epochs,
            warmup_steps=5,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            dataset_text_field="text",
        ),
    )

    # Train and save
    trainer.train()

    model.push_to_hub_gguf(
        hf_save_gguf_name,
        tokenizer,
        quantization_method=quantization_method,
        token=os.environ["HF_ACCESS_TOKEN"],
    )

    return ""


@app.local_entrypoint()
def main():
    # PARAMS
    base_model = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    hf_dataset_name = "pookie3000/pg_chat"
    hf_save_gguf_name = "pookie3000/Paul-Graham-MODAL-Q4_K_M-GGUF"
    epochs = 3
    train_embeddings = False
    max_seq_length = 2048
    quantization_method = "q4_k_m"

    # TRAINING
    train.remote(
        base_model=base_model,
        hf_dataset_name=hf_dataset_name,
        hf_save_gguf_name=hf_save_gguf_name,
        epochs=epochs,
        train_embeddings=train_embeddings,
        max_seq_length=max_seq_length,
        quantization_method=quantization_method,
    )


# run using
# modal run finetuning/modal/paul_graham_chat_finetuning.py
