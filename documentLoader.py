from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Define the path to the pre-trained model you want to use
modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

def embedThis(embeddingModel, text):
    query_result = embeddings.embed_query(text)
    return query_result


def loadDocumentUrl(weburl):
    loader = WebBaseLoader(weburl)

    data = loader.load()

    print('here is an example-> ', data[0].page_content[:100], '\n\n\n\n\n')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    # 'data' holds the text you want to split, split the text into documents using the text splitter.
    docs = text_splitter.split_documents(data)

    print('After parsing and splitting here is an example -> ', docs[:2])
    print('\n\n\n')

    return docs

if __name__ == '__main__':
    result = embedThis(embeddings, 'My name is buga buga')

    print(result[:5], len(result))