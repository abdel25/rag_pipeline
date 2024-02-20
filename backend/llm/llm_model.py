from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from llm.model_names import mistral_model
from utils.main import select_gpu_if_available


class LlmModel:
    def __init__(self, model_name=mistral_model):
        dtype = select_gpu_if_available()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            # device_map="auto",
        )
        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0,
            max_new_tokens=4096,
            return_full_text=False,
        )

    def generate_text(self, input_text: str):
        inputs = self.tokenizer.encode(
            text=input_text,
            return_tensors="pt",
        )
        output = self.model.generate(inputs)  # type: ignore
        generated_text = self.tokenizer.batch_decode(
            output, skip_special_tokens=True, device_map="auto"
        )
        return " ".join(generated_text)

    def cleanup(self):
        # If you're using a GPU, make sure to release GPU memory
        if self.model.device.type == "cuda":  # type: ignore
            self.model = self.model.to("cpu")  # Move model to CPU to release GPU memory

        # Delete references to the model
        del self.model
        del self.tokenizer

        # Perform garbage collection to release memory
        import gc

        gc.collect()
