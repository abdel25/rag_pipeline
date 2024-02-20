from git import Repo
from langchain_community.document_loaders import GitLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from embedding.utils import extension_to_language
from embedding.model_names import sentence_t5_base
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Any
from utils.main import select_gpu_if_available


class RagWrapper:
    repo_url: str
    default_branch = "main"
    repo_local_path: str
    retriever: Any  # type: ignore

    def __init__(self, repo_url: str, file_type: str, branch: str | None = None):
        self.downloadRepository(repo_url)
        self.loadSplitEmbedDocs(branch, file_type)

    def downloadRepository(
        self,
        repo_url: str,
    ):
        repo_name = repo_url.split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        local_path = os.path.join("remote_code", repo_name)
        if not os.path.isdir(local_path):
            print("Cloning repository")
            try:
                repo = Repo.clone_from(
                    url=repo_url,
                    to_path=local_path,
                    no_checkout=True,
                )
                print("Repository was successfully cloned")
            except Exception as e:
                print(e)
        else:
            print("Repository already exists")
        self.repo_local_path = local_path

    def loadSplitEmbedDocs(self, branch: str | None = None, file_type: str = ".py"):
        try:
            # Load
            loader = GitLoader(
                repo_path=self.repo_local_path,
                file_filter=lambda file_path: file_path.endswith(file_type),
                branch=branch or self.default_branch,
            )
            docs = loader.load()

            # Split
            code_splitter = RecursiveCharacterTextSplitter.from_language(
                language=extension_to_language.get(file_type, Language.PYTHON),
                chunk_size=1000,
                chunk_overlap=0,
            )
            texts = code_splitter.split_documents(docs)

            # embed and save in vector_store
            embeddings = HuggingFaceEmbeddings(
                model_name=sentence_t5_base,
                encode_kwargs={"normalize_embeddings": True},
                model_kwargs={"device": "cuda"},
            )
            db = Chroma.from_documents(texts, embeddings)
            self.retriever = db.as_retriever(
                search_type="mmr",  # Also test "similarity"
                search_kwargs={"k": 8},
            )
            print(self.retriever.get_relevant_documents("iter_components")[0])
            del embeddings

        except Exception as e:
            print(e)
