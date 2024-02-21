Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

In the file .env.example, set HUGGINGFACEHUB_API_TOKEN to an api access key you create in your hugging-face account.

To run the application, execute the following command:
   ```
   streamlit run app.py
   ```

The app reads multiple PDF documents and extracts their text content. The extracted text is divided into smaller chunks that can be processed effectively. The application utilizes a language model to generate vector representations (embeddings) of the text chunks. When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones. The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.
