- opt models and dolly are too small and they do not generate anything (llm_output=null) when invoked
- you can set langchain.debug=True to enable debug in terminal
- when you get this message "", this means your propt has too much token and the model might not be able to handle it ===> increase max_new_tokens or reduce prompt content
- Orca 2 is not yet compatible with langchain
- how to avoid too much response drift
- vectorStore is good but multi query retriever might be better
- add indexing to avoid recomputing embeddings every time
- to remove model go to ~/.cache/huggingface/hub and rm -r models--
- the llm drifts when it does not find the good context ==> the retriever is not stable