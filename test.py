# import streamlit as st
# import pandas as pd
# import pdfplumber
# from haystack import Document
# from haystack import Pipeline
# from haystack.document_stores.in_memory import InMemoryDocumentStore
# from haystack.document_stores.types import DuplicatePolicy
# from haystack_integrations.components.embedders.cohere.text_embedder import CohereTextEmbedder
# from haystack_integrations.components.embedders.cohere.document_embedder import CohereDocumentEmbedder
# from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
# from haystack.components.builders.prompt_builder import PromptBuilder
# from haystack_integrations.components.generators.cohere import CohereGenerator
# import os
# from io import StringIO

# def change_bgm():
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("https://t4.ftcdn.net/jpg/01/17/22/31/360_F_117223193_GVPlYQZIF3BSwsUOHakbIsOM9Z5FJfFQ.jpg");
#             background-size: cover;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# change_bgm()

# # Initialize the Cohere API key
# os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]

# st.title("Doclingual - Multilingual Document Understanding")

# # Upload file
# uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=["pdf", "csv"])

# if uploaded_file:
#     documents = []

#     # Process CSV file
#     if uploaded_file.name.endswith('.csv'):
#         data = pd.read_csv(uploaded_file)
#         data.rename(columns={"anwer-command-r": "answer"}, inplace=True)

#         for index, doc in data.iterrows():
#             if isinstance(doc["answer"], str) and len(doc["answer"]):
#                 ques = doc["question"].encode().decode()
#                 ans = doc["answer"].encode().decode()

#                 documents.append(
#                     Document(
#                         content="Question: " + ques + "\nAnswer: " + ans
#                     )
#                 )

#     # Process PDF file
#     elif uploaded_file.name.endswith('.pdf'):
#         with pdfplumber.open(uploaded_file) as pdf:
#             for page in pdf.pages:
#                 text = page.extract_text()
#                 # Simple heuristic to split questions and answers
#                 lines = text.split("\n")
#                 for i in range(0, len(lines), 2):
#                     if i + 1 < len(lines):
#                         ques = lines[i]
#                         ans = lines[i + 1]
#                         documents.append(
#                             Document(
#                                 content="Question: " + ques + "\nAnswer: " + ans
#                             )
#                         )

#     # Initialize the document store
#     document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

#     # Embed the documents
#     document_embedder = CohereDocumentEmbedder(model="embed-multilingual-v2.0")
#     documents_with_embeddings = document_embedder.run(documents)["documents"]

#     # for doc in documents_with_embeddings:
#     #     st.write(f"Embedded Document: {doc.content[:500]} with Embedding Shape: {doc.embedding.shape}")

    
#     document_store.write_documents(documents_with_embeddings, policy=DuplicatePolicy.SKIP)
#     # Define the pipeline components
#     template = """
#         You are a highly accurate and reliable information retriever. Your task is to carefully analyze the provided context and answer the question with the highest level of accuracy.

# Context:
# {% for document in documents %}
#    {{ document.content }}
# {% endfor %}

# Question: {{ query }}?

# Instructions:
# Your answer must be based strictly on the information provided in the context.
# Focus on clarity, precision and maximum accurate answer
# If the context provides conflicting information, prioritize the most reliable or recent data.
# If the context does not contain enough information to answer the question, state that explicitly.

# Provide a concise and accurate response to the question.
         
#         """

#     query_pipeline = Pipeline()
#     query_pipeline.add_component("text_embedder", CohereTextEmbedder(model="embed-multilingual-v2.0"))
#     query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
#     query_pipeline.add_component("prompt_builder", PromptBuilder(template=template))
#     query_pipeline.add_component("llm", CohereGenerator())
#     query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
#     query_pipeline.connect("retriever", "prompt_builder.documents")
#     query_pipeline.connect("prompt_builder", "llm")

#     # Allow the user to ask questions
#     query = st.text_input("Ask a question based on the uploaded document:")
#     if query:
#         result = query_pipeline.run({"text_embedder": {"text": query}})
#         st.write("Answer:", result["llm"]["replies"][0])

# else:
#     st.info("Please upload a PDF or CSV file to proceed.")






import os
import streamlit as st
import pdfplumber  # For PDF processing
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.cohere.text_embedder import CohereTextEmbedder
from haystack_integrations.components.embedders.cohere.document_embedder import CohereDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.cohere import CohereGenerator
from haystack import Pipeline
from haystack.document_stores.types import DuplicatePolicy
os.environ["COHERE_API_KEY"] = "XyBudfarv1bxPBBJRDK6BpX00NujnPz3RTRFXioy"
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text, chunk_size=500):
    """Split extracted text into smaller chunks."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])

def prepare_documents_from_pdf(text, doc_id, chunk_size=500):
    """Prepare documents from the extracted PDF text by splitting it into chunks."""
    chunks = list(split_text_into_chunks(text, chunk_size))
    documents = [
        Document(
            content=f"Context: {chunk}\ndoc_id: {doc_id}_part_{i+1}",
            meta={"doc_id": f"{doc_id}_part_{i+1}"}
        )
        for i, chunk in enumerate(chunks)
    ]
    return documents

def build_pipeline_and_query(documents, query, chunk_size=500):
    """Build the pipeline, process the PDF, and answer the query."""
    # Initialize the document store
    document_store = InMemoryDocumentStore()

    # Embed and store documents with embeddings
    document_embedder = CohereDocumentEmbedder(model="embed-multilingual-v2.0")
    documents_with_embeddings = document_embedder.run(documents)['documents']
    document_store.write_documents(documents_with_embeddings, policy=DuplicatePolicy.SKIP)

    # Define the template for the prompt
    # template = """
    # You are a highly accurate and reliable information retriever. Your task is to carefully analyze the provided context and answer the question with the highest level of accuracy. When giving answer try to take a step back and check from the ocntext whther the answer is correct or not.
    # ALSO PROVIDE COMPLETE ANSWER IN THE LANGUAGE THE QUESTION IS ASKED. TRY TO LOOK FOR KEYWORDS MATCHING THE QUESTION IN THE CONTEXT AND ANSWER BY CHECKING IT THROUGHLY.

    # Context:
    # {% for document in documents %}
    #     {{ document.content }}
    # {% endfor %}

    # Question: {{ query }}?

    # Answer based on the above context from doc_id(s): {% for document in documents %}{{ document.meta['doc_id'] }} {% endfor %}
    # """
    template = """
    Act as if you are a college professor helping people to better understand the contents of a document. Your task is to analyze the provided context and answer each question. Before answering the question, take a step and ensure that to carefully analyze the context and answer the question fully. The answer should always be in the language that the question was provided. Also paraphrase the answer.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{ query }}?

    Answer: 

    """

    # Initialize the query pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", CohereTextEmbedder(model="embed-multilingual-v2.0"))
    query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
    query_pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    query_pipeline.add_component("llm", CohereGenerator())

    # Connect the components in the pipeline
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("retriever", "prompt_builder.documents")
    query_pipeline.connect("prompt_builder", "llm")

    # Run the query through the pipeline
    result = query_pipeline.run({"text_embedder": {"text": query}})

    return result

# Streamlit app setup
st.title("Multilingual PDF Question Answering App")

# File upload
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    pdf_path = f"./{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text from PDF using pdfplumber
    text = extract_text_from_pdf(pdf_path)

    # Prepare documents from PDF text
    doc_id = uploaded_file.name.split(".")[0]
    documents = prepare_documents_from_pdf(text, doc_id)

    # Get the user's query
    query = st.text_input("Enter your question:")

    if query:
        # Build the pipeline and get the answer
        result = build_pipeline_and_query(documents, query)
        
        # Display the result
        st.subheader("Answer:")
        st.write(result['llm']["replies"][0])
