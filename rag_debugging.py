from datasets import load_dataset
from torch import cuda, bfloat16
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
import time
from pinecone import Pinecone
import transformers
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA


embed_model = 'hkunlp/instructor-xl'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)



# get API key from app.pinecone.io 
pinecone = Pinecone(api_key='1197c041-b95d-4aa1-b523-617a9224bff6')

# Initialize the index.
index_name = 'rag'

if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        index_name,
        dimension=len(embeddings[0]),
        metric='cosine'
    )
    # wait for index to finish initialization
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

# Connect to the index
index = pinecone.Index(index_name)
index.describe_index_stats()

data = load_dataset("alexjercan/bugnet", split='train')

# Embed and index the documents
data = data.to_pandas()

batch_size = 32

for i in range(0, len(data), batch_size):
    i_end = min(len(data), i+batch_size)
    batch = data.iloc[i:i_end]
    ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
    texts = [x['chunk'] for i, x in batch.iterrows()]
    embeds = embed_model.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {'buggy_code': x['fail'],
         'fixed_code': x['pass'],
         'change': x['change']} for i, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))

print(index.describe_index_stats())



# initialize the model and move it to CUDA-enabled GP
model_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")

# define a tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
    

# Initialize the HF pipeline
generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)



# implement this in LangChain
llm = HuggingFacePipeline(pipeline=generate_text)



# initializing the LangChain vector store
code = 'fixed_code'  # field in metadata that contains the fixed code

vectorstore = Pinecone(
    index, embed_model.embed_query, code
)



# create our RAG pipeline
rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=vectorstore.as_retriever()
)


#llm('find and fix the bug in this code:')
rag_pipeline('what is so special about llama 2?')