import os
import sys
import torch


backend_folder = f"{os.getcwd()}/backend"

sys.path.append(backend_folder)


from llm.llm_model import LlmModel
from embedding.rag_wrapper import RagWrapper
from langchain_wrapper.lang_wrapper import LangWrapper
from llm.model_names import code_llama_model_13b_instruct, mistral_model
from utils.main import select_gpu_if_available


def main():
    try:
        # model
        model = LlmModel(model_name=code_llama_model_13b_instruct)

        # rag
        repo_url = "https://github.com/esphome/esphome"
        branch = "dev"
        file_type = ".py"
        ragWrapper = RagWrapper(repo_url=repo_url, branch=branch, file_type=file_type)

        # langchain
        langchain_wrapper = LangWrapper(model=model)
        langchain_wrapper.add_rag_wrapper(ragWrapper)
        langchain_wrapper.setup_rag_llm_chain()

        question = """
            Briefly tell me what the codegen.py file does
            """
        generated_text = langchain_wrapper.invoke_llm_chain(question)
        history = generated_text["chat_history"]  # type: ignore
        # gen_text = model.generate_text(question)
        print(generated_text["answer"])  # type: ignore

        question = """
            output EXATCLY the COMPLETE code of 'iter_components' function AS IS
            """
        generated_text = langchain_wrapper.invoke_llm_chain(question=question)

        # gen_text = model.generate_text(question)
        print(generated_text["answer"])  # type: ignore

        question = """
            what is the path of a02yyuw.cpp file in the repository ?
            """
        generated_text = langchain_wrapper.invoke_llm_chain(question=question)

        # gen_text = model.generate_text(question)
        print(generated_text["answer"])  # type: ignore

        langchain_wrapper.cleanup()

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
