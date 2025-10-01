import asyncio
import logging
from langchain_mistralai import ChatMistralAI
from langchain_core.language_models import BaseChatModel

from typing import Dict, Any
import httpx
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


## 1. Set up LLM provider

# Load an LLM
def load_chat_model(model: str) -> BaseChatModel:
    """
    Load an LLM.

    Pick the right LangChain chat wrapper and sets sensible defaults 
    (temperature=0 → deterministic; max_tokens=1024).
    A LangChain chat wrapper = a standardized Python class (and JS) that hides the messy 
    provider-specific details and givesthe same simple API across different LLM backends
    and providers (OpenAI, Mistral, Groq, Ollama, etc.).
    """
    provider, model_name = model.split("/", maxsplit=1)
    if provider == "mistralai":
        # https://python.langchain.com/docs/integrations/chat/mistralai/
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            model=model_name,
            temperature=0,
            max_tokens=1024,
        )
    if provider == "groq":
        # https://python.langchain.com/docs/integrations/chat/groq/
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model_name,
            temperature=0,
            max_tokens=1024,
        )
    if provider == "ollama":
        # https://python.langchain.com/docs/integrations/chat/ollama/
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, temperature=0)

    raise ValueError(f"Unknown provider: {provider}")

## Pick a model
# Ollama = runs models locally. For Mistral/Groq we'd need API keys.
##
# Model Mistral
# llm = load_chat_model("mistralai/mistral-small-latest")
# Model Llama
# llm = load_chat_model("groq/meta-llama/llama-4-scout-17b-16e-instruct")
# Model Ollama
llm = load_chat_model("ollama/mistral")


## 2. Initialize vector database for similarity search, and index relevant documents

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

### Set up vector database for document retrieval
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5") # turns text → 384-dim vectors
embedding_dimensions = 384
collection_name = "sparql-docs"
vectordb = QdrantClient(path="data/vectordb") # create local Qdrant store on disk

# Qdrant is a vector database. Instead of storing and searching normal data 
# (like names, numbers, rows in SQL), it stores vectors (lists of numbers, e.g. [0.23, -0.77, 1.02, …])
# Those vectors usually come from an embedding model — a neural network that turns text into numbers 
# that capture its meaning. Example: “cat” and “kitten” → two vectors close together in vector space.
# “cat” and “car” → vectors further apart.
# So, we ask Qdrant: “which vectors are closest to this new one?” It does fast nearest-neighbor 
# search using algorithms optimized for millions of vectors.

from langchain_core.documents import Document
from qdrant_client.http.models import Distance, VectorParams
from sparql_llm import SparqlExamplesLoader, SparqlVoidShapesLoader, SparqlInfoLoader

## 2. Set up vector database for document retrieval
endpoints: list[dict[str, str]] = [
    { "endpoint_url": "https://sparql.uniprot.org/sparql/" },
    { "endpoint_url": "https://www.bgee.org/sparql/" },
    { "endpoint_url": "https://sparql.omabrowser.org/sparql/" },
]

def index_endpoints():
    """Index SPARQL endpoints metadata in the vector database."""
    docs: list[Document] = []
    # Fetch documents from endpoints
    for endpoint in endpoints:
        logging.info(f"🔎 Retrieving metadata for {endpoint['endpoint_url']}")
        docs += SparqlExamplesLoader(
            endpoint["endpoint_url"],
            examples_file=endpoint.get("examples_file"),
        ).load()
        docs += SparqlVoidShapesLoader(
            endpoint["endpoint_url"],
            void_file=endpoint.get("void_file"),
            examples_file=endpoint.get("examples_file"),
        ).load()
    docs += SparqlInfoLoader(endpoints, source_iri="https://www.expasy.org/").load()

    # Load documents in vectordb
    # to build a retrieval memory of examples and schemas, 
    # so we can automatically fetch the most relevant context 
    # for each user question before asking the LLM to generate a query
    #
    # LLM is bad at remembering or searching large amounts of text. Instead of giving the LLM 
    # all possible examples every time, we:
    # - Upload documents (examples, schemas) into Qdrant once.
    # - At runtime, when a user asks a question:
    # - Embed the question into a vector.
    # - Ask Qdrant: “Which stored docs are most similar to this question?”
    # - Qdrant quickly finds the 3–5 most relevant examples.
    # - Give only those relevant examples to the LLM inside the system prompt.
    # This makes the LLM’s job easier → it can base its query generation on actual real examples, not just its “imagination.”


    if vectordb.collection_exists(collection_name):
        vectordb.delete_collection(collection_name)
    vectordb.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimensions, 
                                    distance=Distance.COSINE),
    )
    embeddings = embedding_model.embed([q.page_content for q in docs])
    vectordb.upload_collection(
        collection_name=collection_name,
        vectors=[embed.tolist() for embed in embeddings],
        payload=[doc.metadata for doc in docs],
    )
    logging.info(f"✅ Indexed {len(docs)} documents in collection {collection_name}")

    ex_question = "How to retrieve proteins?"
    docs.append(Document(
    page_content=ex_question,
    metadata={
        "question": ex_question,
        "answer": """SELECT ?protein WHERE {
    ?protein a up:Protein .
}""",
        "endpoint_url": "https://sparql.uniprot.org/",
        "query_type": "SelectQuery",
        "doc_type": "SPARQL endpoints query examples",
    },
    ))

if not vectordb.collection_exists(collection_name) or vectordb.get_collection(collection_name).points_count == 0:
    index_endpoints()
else:
    logging.info(
        f"ℹ️  Using existing collection '{collection_name}' with {vectordb.get_collection(collection_name).points_count} vectors"
    )


## 3. Set up document retrieval, and pass relevant context to the system prompt
# Retrieve the most relevant docs for a user question

from qdrant_client.models import FieldCondition, Filter, MatchValue, ScoredPoint
retrieved_docs_count = 3
def retrieve_docs(question: str) -> list[ScoredPoint]:
    """
    Retrieve documents relevant to the user's question.
    It embeds the user’s question and asks Qdrant for the nearest examples and class shapes.
    """
    question_embeddings = next(iter(embedding_model.embed([question])))
    retrieved_docs = vectordb.query_points(
        collection_name=collection_name,
        query=question_embeddings,
        limit=retrieved_docs_count,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="doc_type",
                    match=MatchValue(value="SPARQL endpoints query examples"),
                )
            ]
        ),
    ).points
    retrieved_docs += vectordb.query_points(
        collection_name=collection_name,
        query=question_embeddings,
        limit=retrieved_docs_count,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="doc_type",
                    match=MatchValue(value="SPARQL endpoints classes schema"),
                )
            ]
        ),
    ).points
    return retrieved_docs

def format_doc(doc: ScoredPoint) -> str:
    """Format a question/answer document to be provided as context to the model."""
    doc_lang = (
        f"sparql\n#+ endpoint: {doc.payload.get('endpoint_url', 'not provided')}"
        if "query" in doc.payload.get("doc_type", "")
        else ""
    )
    return f"\n{doc.payload['question']} ({doc.payload.get('endpoint_url', '')}):\n\n```{doc_lang}\n{doc.payload.get('answer')}\n```\n\n"
SYSTEM_PROMPT = """You are an assistant that helps users to write SPARQL queries.
Put the SPARQL query inside a markdown codeblock with the "sparql" language tag, and always add the URL of the endpoint on which the query should be executed in a comment at the start of the query inside the codeblocks starting with "#+ endpoint: " (always only 1 endpoint).
Use the queries examples and classes shapes provided in the prompt to derive your answer, don't try to create a query from nothing and do not provide a generic query.
Try to always answer with one query, if the answer lies in different endpoints, provide a federated query.
And briefly explain the query.
Here is a list of documents (reference questions and query answers, classes schema) relevant to the user question that will help you answer the user question accurately:
{relevant_docs}
"""


## 4. Automatically execute generated query and interpret results

# Execute generated SPARQL query -> SPARQL = bridge between LLM and the real databases
# SPARQL (pronounced “sparkle”) is a query language, like SQL but for RDF data (Resource Description Framework)
# RDF is a way to store knowledge as triples, e.g.:
# Subject     Predicate     Object
# "TP53"      "ortholog"    "Trp53" (rat)
# SPARQL lets us ask questions over these triples, often hosted in SPARQL endpoints (special databases accessible via HTTP).
# we need SPARQL queries because many biological databases expose their data through SPARQL endpoints:
# UniProt (proteins, sequences, annotations)
# OMA Browser (orthologs / evolutionary relationships)
# Bgee (gene expression in species)
# 
# Why not just ask the LLM directly? An LLM like Mistral doesn’t have live access to UniProt or OMA. It only “knows” 
# what was in its training data (static, possibly outdated).
# If we ask: “What are the rat orthologs of human TP53?”
# The LLM might hallucinate an answer. But if it generates a correct SPARQL query, your system can run it against OMA’s 
# endpoint and fetch the real, up-to-date data, and many others in the life sciences.
# SPARQL example:
# #+ endpoint: https://sparql.omabrowser.org/sparql/
# SELECT ?ratGene WHERE {
#   ?humanGene a orth:Gene ;
#              rdfs:label "TP53" ;
#             orth:orthologous ?ratGene .
#   ?ratGene orth:inSpecies "Rattus norvegicus" .
#}

from sparql_llm.validate_sparql import extract_sparql_queries
from sparql_llm.utils import query_sparql
from httpx import HTTPStatusError

def execute_query(last_msg: str) -> list[dict[str, str]]:
    """Extract SPARQL query from markdown and execute it."""
    for extracted_query in extract_sparql_queries(last_msg):
        if extracted_query.get("query") and extracted_query.get("endpoint_url"):
            res = query_sparql(extracted_query.get("query"), extracted_query.get("endpoint_url"))
            return res.get("results", {}).get("bindings", [])
        
def execute_query(last_msg: str):
    for q in extract_sparql_queries(last_msg):
        query = q.get("query","")
        endpoint = q.get("endpoint_url","")
        if "SERVICE <" in query:
            raise ValueError("Federated queries are not allowed. Please use a single endpoint.")
        # then call query_sparql(query, endpoint) as you do now


from sparql_llm.validate_sparql import extract_sparql_queries
from sparql_llm.utils import query_sparql  # this is what you already call under the hood

# --- simple validator: forbid federated queries and require an endpoint ---
_SERVICE_RE = re.compile(r"\bSERVICE\s*<", re.IGNORECASE)

def execute_query(last_msg: str) -> Dict[str, Any]:
    """
    Extract a SPARQL query + endpoint from the model's last message,
    validate it's single-endpoint (no SERVICE), and run it.
    Returns a dict with 'status' and optional 'result' / 'error'.
    """
    queries = list(extract_sparql_queries(last_msg))

    if not queries:
        return {
            "status": "no_query",
            "error": "No SPARQL query found in the model output."
        }

    # take the first extracted query (or pick the longest if you prefer)
    q = queries[0]
    query = (q.get("query") or "").strip()
    endpoint = (q.get("endpoint_url") or "").strip()

    if not query:
        return {
            "status": "invalid",
            "error": "Empty SPARQL query extracted."
        }

    if not endpoint:
        return {
            "status": "invalid",
            "error": "Missing endpoint_url annotation (e.g., '#+ endpoint: https://…')."
        }

    if _SERVICE_RE.search(query):
        return {
            "status": "invalid",
            "error": "Federated queries are not allowed. Remove any SERVICE clauses and use a single endpoint."
        }

    try:
        # You can pass extra params like timeout=… if your helper supports it
        res = query_sparql(query, endpoint)
        return {
            "status": "ok",
            "endpoint": endpoint,
            "result": res,
        }

    except httpx.HTTPStatusError as e:
        # Endpoint responded but with an error code (e.g., 400/500)
        return {
            "status": "endpoint_error",
            "endpoint": endpoint,
            "error": f"{e}",
            "status_code": e.response.status_code,
            "body": e.response.text[:1000],  # trim for logs
        }
    except httpx.HTTPError as e:
        # Network-level error (timeouts, DNS, connect errors)
        return {
            "status": "network_error",
            "endpoint": endpoint,
            "error": f"{e}",
        }
    except Exception as e:
        # Anything else unexpected – don’t crash the app
        return {
            "status": "unexpected_error",
            "endpoint": endpoint,
            "error": f"{type(e).__name__}: {e}",
        }
    
# Default main
#async def main() -> None:
#    question = "What are the rat orthologs of human TP53?"
#    logging.info("Hello world")
#    # 🔨 Call the different steps of the pipeline here

# Main with invoke()
#async def main():
#    question = "What are the rat orthologs of human TP53?"
#    resp = llm.invoke(question)
#    print(resp)

# Main with stream()
#async def main():
#    question = "What are the rat orthologs of human TP53?"
#    for msg in llm.stream(question):
#        print(msg.content, end="", flush=True)


#import httpx
#question = "What are the rat orthologs of human TP53?"
#SYSTEM_PROMPT = """You are an assistant that helps users to navigate the resources and databases from the SIB Swiss Institute of Bioinformatics.
#Here is the description of resources available at the SIB:
#{context}
#Use it to answer the question"""
#async def main() -> None:
#    # ...
#    response = httpx.get("https://github.com/sib-swiss/sparql-llm/raw/refs/heads/main/src/expasy-agent/expasy_resources_metadata.csv", follow_redirects=True)
#    messages = [
#        ("system", SYSTEM_PROMPT.format(context=response.text)),
#        ("human", question),
#    ]
#    for resp in llm.stream(messages):
#        print(resp.content, end="", flush=True)
#        if resp.usage_metadata:
#            print(f"\n\n{resp.usage_metadata}")


#async def main():
#    question = "What are the rat orthologs of human TP53?"
#    retrieved_docs = retrieve_docs(question)
#    formatted_docs = "\n".join(format_doc(doc) for doc in retrieved_docs)
#    messages = [
#        ("system", SYSTEM_PROMPT.format(relevant_docs=formatted_docs)),
#        ("user", question),
#    ]
#    for resp in llm.stream(messages):
#        print(resp.content, end="", flush=True)
#        if resp.usage_metadata:
#            print("\n")
#            logging.info(f"🎰 {resp.usage_metadata}")

max_try_count = 3

import json
import logging

async def main():
    question = "What are the rat orthologs of human TP53?"
    print(question)

    # Retrieve relevant documents and add them to conversation
    retrieved_docs = retrieve_docs(question)
    formatted_docs = "\n".join(format_doc(doc) for doc in retrieved_docs)

    messages = [
        ("system", SYSTEM_PROMPT.format(relevant_docs=formatted_docs)),
        ("user", question),
    ]

    for _i in range(max_try_count):
        # --- stream the model’s answer and collect it
        complete_answer = ""
        for resp in llm.stream(messages):
            print(resp.content, end="", flush=True)
            complete_answer += resp.content
            if resp.usage_metadata:
                print("\n")
                logging.info(f"🎰 {resp.usage_metadata}")

        messages.append(("assistant", complete_answer))

        # --- try to execute what the model produced
        res = execute_query(complete_answer)

        if res["status"] == "ok":
            rows = res.get("result") or []
            if not rows:
                logging.warning("⚠️ Query returned 0 rows, asking for a fix")
                messages.append((
                    "user",
                    "The query executed but returned **no results**. "
                    "Please correct it. Keep it **single-endpoint** and **do not use SERVICE**."
                ))
                continue

            logging.info(f"✅ Got {len(rows)} rows, asking for a summary")
            messages.append((
                "user",
                "The query succeeded. Please summarize these results for a biologist:\n\n"
                + json.dumps(rows, ensure_ascii=False, indent=2)
            ))
            break  # success path ends the loop

        elif res["status"] in ("invalid", "no_query"):
            # Typical causes: missing endpoint annotation, empty query, or contains SERVICE
            reason = res.get("error", res["status"])
            logging.warning(f"⚠️ Invalid query: {reason}")
            messages.append((
                "user",
                f"Invalid query: {reason}\n\n"
                "Please regenerate a **single-endpoint** SPARQL query (no `SERVICE` clauses). "
                "Annotate the endpoint with a line like:\n"
                "`#+ endpoint: https://sparql.omabrowser.org/sparql/`"
            ))
            continue

        elif res["status"] == "endpoint_error":
            code = res.get("status_code")
            ep = res.get("endpoint", "")
            logging.warning(f"⚠️ Endpoint error {code} from {ep}")
            messages.append((
                "user",
                f"The endpoint `{ep}` returned HTTP {code}. This often happens with federated queries or "
                "unsupported predicates.\n\n"
                "Please **rewrite the query for a single endpoint** (no `SERVICE`), and choose an endpoint that "
                "actually has orthology data (e.g., `https://sparql.omabrowser.org/sparql/`)."
            ))
            continue

        elif res["status"] in ("network_error", "unexpected_error"):
            reason = res.get("error", res["status"])
            logging.warning(f"⚠️ Execution failed: {reason}")
            messages.append((
                "user",
                f"Execution failed due to `{res['status']}`: {reason}\n"
                "Please try a simpler **single-endpoint** query without federation."
            ))
            continue

    else:
        logging.error("❌ Gave up after max tries without a successful execution.")


 ## 5. Setup chat web UI (with Chainlit)       
# Deploy with a nice web UI
import chainlit as cl

@cl.on_message
async def on_message(msg: cl.Message):
    """Main function to handle when user send a message to the assistant."""
    retrieved_docs = retrieve_docs(msg.content)
    formatted_docs = "\n".join(format_doc(doc) for doc in retrieved_docs)
    async with cl.Step(name=f"{len(retrieved_docs)} relevant documents 📚️") as step:
        step.output = formatted_docs
    messages = [
        ("system", SYSTEM_PROMPT.format(relevant_docs=formatted_docs)),
        *cl.chat_context.to_openai(),
    ]

    query_success = False
    for _i in range(max_try_count):
        answer = cl.Message(content="")
        for resp in llm.stream(messages):
            await answer.stream_token(resp.content)
            if resp.usage_metadata:
                logging.info(f"🎰 {resp.usage_metadata}")
        await answer.send()

        if query_success:
            break

        query_res = execute_query(answer.content)
        if len(query_res) < 1:
            logging.warning("⚠️ No results, trying to fix")
            messages.append(("user", f"""The query you provided returned no results, please fix the query:\n\n{answer.content}"""))
        else:
            logging.info(f"✅ Got {len(query_res)} results! Summarizing them, then stopping the chat")
            async with cl.Step(name=f"{len(query_res)} query results ✨") as step:
                step.output = f"```json\n{json.dumps(query_res, indent=2)}\n```"
            messages.append(("user", f"""The query you provided returned these results, summarize them:\n\n{json.dumps(query_res, indent=2)}"""))
            query_success = True       



if __name__ == "__main__":
    asyncio.run(main())
