from datetime import datetime
import os
import sys

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset


dataset = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
train_dataset = dataset.train_test_split(test_size=0.1)["train"]
eval_dataset = dataset.train_test_split(test_size=0.1)["test"]

base_model = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")


tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""You are a powerful text-to-code model. Your job is to answer questions about a code. You are given a question regarding a code.

You must output the python code that answers the question.

### Input:
{data_point["instruction"]}

### Response:
{data_point["output"]}
"""
    return tokenize(full_prompt)


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

model.train() # put model back into training mode
model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)


if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True


batch_size = 128
per_device_train_batch_size = 32
gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = "assistant-code-llama"

training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        max_steps=400,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps", # if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=20,
        save_steps=20,
        output_dir=output_dir,
        # save_total_limit=3,
        load_best_model_at_end=False,
        # ddp_find_unused_parameters=False if ddp else None,
        group_by_length=True, # group sequences of roughly the same length together to speed up training
    )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)
if torch.__version__ >= "2" and sys.platform != "win32":
    print("compiling the model")
    model = torch.compile(model)


trainer.train()

