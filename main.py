import documentLoader
import memoryUse
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os, sys
import torch
from transformers import pipeline
from rich import print

embeddings = documentLoader.embeddings

url = 'https://en.wikipedia.org/wiki/Naruto'
docs = documentLoader.loadDocumentUrl(url)

db = memoryUse.create_vectorstore(docs, embeddings)

# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)

# Create a retriever object from the 'db' with a search configuration where it retrieves up to 4 relevant splits/documents.
retriever = db.as_retriever(search_kwargs={"k": 4})

# Create a question-answering instance (qa) using the hugging face pipeline class.
# as the RetrivalQA CLass in Langchain is not working 

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating

print('\n [green] USER> [/green]')
question = input()

while question.lower() != 'exit':
    docs = retriever.get_relevant_documents(question)

    retrived_text = '\n\n'.join([doc.page_content for doc in docs])
    messages = [
        {
            "role": "system",
            "content": f'''Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            context: {retrived_text}'''
        },
        {"role": "user", "content": f'{question}'}
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print('[red] AI: [/red]', outputs[0]["generated_text"][len(prompt):])
    print('\n [green] USER> [/green]')
    question = input()
    # <|system|>
    # You are a friendly chatbot who always responds in the style of a pirate.</s>
    # <|user|>
    # How many helicopters can a human eat in one sitting?</s>
    # <|assistant|>
    # ...





