from langchain.vectorstores import FAISS

def create_vectorstore(docs, embeddings):
    db = FAISS.from_documents(docs, embeddings)
    return db

def merge2_vectorstore(db1, db2):
    db1.merge_from(db2)
    return db1

def save_vectorstore(db, filename):
    db.save_local(filename)

def load_vectorstore(filename, db=None):
    db = FAISS.load_local(filename)

    if db is None:
        return db

