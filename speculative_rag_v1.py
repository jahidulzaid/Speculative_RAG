
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
from datasets import Dataset, load_dataset
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from tiktoken import Encoding, encoding_for_model, get_encoding

from qdrant_client import AsyncQdrantClient, models


async def main():
    # Qdrant Client
    path: Path = Path("qdrant_client")
    qdrant_client: AsyncQdrantClient = AsyncQdrantClient(path=path)


    # OpenAI Client
    openai_client: AsyncOpenAI = AsyncOpenAI()


    # Embeddings specs
    embedding_model: str = "text-embedding-3-small"
    dimension: int = 1536
    collection_name: str = "speculative_rag"




    # Get existing collections
    current_collections: models.CollectionsResponse = await qdrant_client.get_collections()

    # Create collection
    if collection_name not in [col.name for col in current_collections.collections]:
        logger.info("Collection {col} doesn't exist. Creating...", col=collection_name)
        await qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.DOT
            ),
        )
        logger.info("Collection {col} created!", col=collection_name)
    else:
        logger.info(
            "Collection {col} already exists, skipping creation.", col=collection_name
        )

    # Load dataset
    dataset: Dataset = load_dataset(
        path="jamescalam/ai-arxiv2-semantic-chunks", split="train"
        # path="rajpurkar/squad_v2", split="train"
    )
    print(json.dumps(dataset[0], indent=4))

    # Using only 50k rows
    rows_to_keep: int = 50_000

    # Easier to handle as pandas df
    records: list[dict[str, Any]] = (
        dataset.to_pandas().iloc[:rows_to_keep].to_dict(orient="records")
    )

    print(records[0])


# Auxiliar functions to prepare the Points
async def create_point(
    client: AsyncOpenAI,
    example: dict[str, Any],
    model: str,
    encoding_name: str,
    max_context_len: int,
) -> models.PointStruct:
    """Creates a Point that contains the payload and the vector."""

    encoding: Encoding = get_encoding(encoding_name=encoding_name)

    embedding_result: Any = await client.embeddings.create(
        input=encoding.encode(text=example.get("content"), disallowed_special=())[
            :max_context_len
        ],
        model=model,
    )
    vector: list[float] = embedding_result.data[0].embedding

    return models.PointStruct(
        id=str(uuid4()),
        vector=vector,
        payload=dict(
            chunk_id=example.get("id"),
            arxiv_id=example.get("arxiv_id"),
            title=example.get("title"),
            content=example.get("content"),
            prechunk_id=example.get("prechunk_id"),
            postchunk_id=example.get("postchunk_id"),
            references=example.get("references").tolist(),
        ),
    )


async def process_batch(
    client: AsyncOpenAI,
    batch: list[dict[str, Any]],
    model: str,
    encoding_name: str,
    max_context_len: int,
) -> list[models.PointStruct]:
    """Processes a batch of examples to create PointStructs."""
    return await asyncio.gather(
        *[
            create_point(
                client=client,
                example=example,
                model=model,
                encoding_name=encoding_name,
                max_context_len=max_context_len,
            )
            for example in batch
        ]
    )


# batch_size: int = 512
# max_context_len: int = 8192
# encoding_name: str = "cl100k_base"
# total_batches: int = len(records) // batch_size
# all_points: list[models.PointStruct | None] = []
    # batch_size: int = 512
    # max_context_len: int = 8192
    # encoding_name: str = "cl100k_base"
    # total_batches: int = len(records) // batch_size
    # all_points: list[models.PointStruct | None] = []
    # _now: float = perf_counter()
    # for i in tqdm(range(0, len(records), batch_size), total=total_batches, desc="Points"):
    #     batch: list[dict[str, Any]] = records[i : i + batch_size]
    #     points: list[models.PointStruct] = await process_batch(
    #         client=openai_client,
    #         batch=batch,
    #         model=embedding_model,
    #         encoding_name=encoding_name,
    #         max_context_len=max_context_len,
    #     )
    #     all_points.extend(points)
    # logger.info("Generated all Points in {secs:.4f} seconds.", secs=perf_counter() - _now)


# Upsert Points
# await qdrant_client.upsert(collection_name=collection_name, points=points)

# #### testing vector search


    query: str = "Mixture of Experts"
    query_vector: Any = await openai_client.embeddings.create(
        input=query, model=embedding_model
    )
    query_vector: list[float] = query_vector.data[0].embedding
    out: list[models.ScoredPoint] = await qdrant_client.search(
        collection_name=collection_name, query_vector=query_vector, with_vectors=True
    )


    print(f"Id: {out[0].id}")
    print(f"Score: {out[0].score:.3}")
    print(f"Title: {out[0].payload.get('title')} [{out[0].payload.get('arxiv_id')}]")
    print(f"Chunk: {out[0].payload.get('content')[:1000]} ...")
    print(f"Vector: {out[0].vector[:5]} ... ")


# ## 2. Speculative RAG


# #### Multi-Perspective Sampling

def multi_perspective_sampling(
    k: int, retrieved_points: list[models.ScoredPoint], seed: int = 1399
) -> list[list[str]]:
    # Generate clusters
    logger.info("Finding {k} clusters.", k=k)
    algo: Any = KMeans(n_clusters=k, random_state=seed)
    _vectors = [point.vector for point in retrieved_points]
    clusters: list[int] = algo.fit_predict(X=_vectors)

    # Unique clusters
    unique_clusters: set[int] = set(clusters)

    # Create a dictionary with the members of each cluster
    cluster_dict: defaultdict[int, list[int | None]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        cluster_dict[cluster].append(index)
    logger.info("Clusters distribution: {dist}", dist=dict(cluster_dict))

    # M subsets
    m: int = min(len(indices) for indices in cluster_dict.values())
    logger.info("{m} document subsets will be created.", m=m)

    # Generate m unique subsets without replacement
    np.random.seed(seed=seed)
    subsets: list[list[str]] = []

    for _ in range(m):
        subset: list[int] = []
        for cluster in unique_clusters:
            chosen_element: int = np.random.choice(cluster_dict[cluster])
            subset.append(chosen_element)
            cluster_dict[cluster].remove(chosen_element)
        subset_documents = [
            retrieved_points[idx].payload.get("content") for idx in subset
        ]
        subsets.append(subset_documents)

    return subsets

# Testing
    k: int = 2
    seed: int = 1399
    now: float = perf_counter()
    sampled_docs: list[list[str]] = multi_perspective_sampling(
        k=k, retrieved_points=out, seed=seed
    )
    logger.info(
        "Multi perspective sampling done in {s:.4f} seconds.", s=perf_counter() - now
    )


    print(sampled_docs)
# #### Rag Drafting

rag_drafting_prompt: str = """Response to the instruction. Also provide rationale for your response.
## Instruction: {instruction}

## Evidence: {evidence}"""


class RagDraftingResponse(BaseModel):
    rationale: str = Field(description="Response rationale.")
    response: str = Field(description="Response to the instruction.")


async def rag_drafting_generator(
    client: AsyncOpenAI,
    model_name: str,
    instruction: str,
    evidence: str,
    **kwargs,
) -> tuple[RagDraftingResponse, float]:
    completion: Any = await client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": rag_drafting_prompt.format(
                    instruction=instruction, evidence=evidence
                ),
            }
        ],
        response_format=RagDraftingResponse,
        temperature=0.0,
        logprobs=True,
        max_tokens=512,
        **kwargs,
    )
    return (
        completion.choices[0].message.parsed,
        np.exp(mean(token.logprob for token in completion.choices[0].logprobs.content)),
    )

# Testing
    m_drafter: str = "gpt-4o-mini-2024-07-18"
    instruction: str = "What is MoE?"

    now: float = perf_counter()
    rag_drafts: list[tuple[RagDraftingResponse, float]] = await asyncio.gather(
        *[
            rag_drafting_generator(
                client=openai_client,
                model_name=m_drafter,
                instruction=instruction,
                evidence="\n".join(
                    [f"[{idx}] {doc}" for idx, doc in enumerate(subset, start=1)]
                ),
            )
            for subset in sampled_docs
        ]
    )
    logger.info("RAG Drafting done in {s:.4f} seconds.", s=perf_counter() - now)
    print(rag_drafts)

# #### Generalist RAG Verifier

# %%
rag_verifier_prompt: str = """## Instruction: {instruction}

## Response: {response} 

## Rationale: {rationale}

Is the rationale good enough to support the answer? (Yes or No)"""


async def rag_verifier_generator(
    client: AsyncOpenAI,
    model_name: str,
    instruction: str,
    evidence: str,
    response: str,
    rationale: str,
    **kwargs,
) -> tuple[Any, float]:
    encoder: Encoding = encoding_for_model(model_name=model_name)
    completion: Any = await client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": rag_verifier_prompt.format(
                    instruction=instruction,
                    evidence=evidence,
                    response=response,
                    rationale=rationale,
                ),
            }
        ],
        temperature=0.0,
        logprobs=True,
        max_tokens=2,
        **kwargs,
    )
    response: str = completion.choices[0].message.content
    cond: bool = encoder.encode(text=response.lower()) == encoder.encode(text="yes")
    p_yes: float = (
        np.exp(mean(token.logprob for token in completion.choices[0].logprobs.content))
        if cond
        else 0.0
    )  # Naive

    return (response, p_yes)

# %%
# Testing
    m_verifier: str = "gpt-4o-2024-08-06"
    instruction: str = "What is MoE?"

    now: float = perf_counter()
    rag_verifications: list[tuple[str, float]] = await asyncio.gather(
        *[
            rag_verifier_generator(
                client=openai_client,
                model_name=m_verifier,
                instruction=instruction,
                evidence="\n".join(
                    [f"[{idx}] {doc}" for idx, doc in enumerate(subset, start=1)]
                ),
                response=rag_drafting_response.response,
                rationale=rag_drafting_response.rationale,
            )
            for subset, (rag_drafting_response, _) in zip(sampled_docs, rag_drafts)
        ]
    )
    logger.info("RAG Drafting done in {s:.4f} seconds.", s=perf_counter() - now)
    print(rag_verifications)

# %% [markdown]
# #### Final Response

# %%
    best_answer: int = np.argmax(
        p_draft * p_self for (_, p_draft), (_, p_self) in zip(rag_drafts, rag_verifications)
    )
    print(f"Response:\n ------ \n{rag_drafts[best_answer][0].response}")

# %% [markdown]
# ## 3. "end-to-end" Code

# %% [markdown]
# #### Speculative Rag

# %%
async def speculative_rag(
    query: str,
    embedding_model: str,
    collection_name: str,
    k: int,
    seed: int,
    client: AsyncOpenAI,
    qdrant_client: AsyncQdrantClient,
    m_drafter: str,
    m_verifier: str,
) -> str:
    _start = perf_counter()

    # Generate query vector embedding
    logger.info("Generating query vector...")
    _now: float = perf_counter()
    query_vector: Any = await client.embeddings.create(
        input=query, model=embedding_model
    )
    query_vector: list[float] = query_vector.data[0].embedding
    logger.info("Query vector generated in {s:.4f} seconds.", s=perf_counter() - _now)

    # Fetching relevant documents
    logger.info("Fetching relevant documents...")
    _now: float = perf_counter()
    out: list[models.ScoredPoint] = await qdrant_client.search(
        collection_name=collection_name, query_vector=query_vector, with_vectors=True
    )
    logger.info("Documents retrieved in {s:.4f} seconds.", s=perf_counter() - _now)

    # Multi Perspective Sampling
    logger.info("Doing Multi Perspective Sampling...")
    _now: float = perf_counter()
    sampled_docs: list[list[str]] = multi_perspective_sampling(
        k=k, retrieved_points=out, seed=seed
    )
    logger.info(
        "Multi Perspective Sampling done in {s:.4f} seconds.", s=perf_counter() - _now
    )

    # RAG Drafting
    logger.info("Doing RAG Drafting...")
    _now: float = perf_counter()
    rag_drafts: list[tuple[RagDraftingResponse, float]] = await asyncio.gather(
        *[
            rag_drafting_generator(
                client=client,
                model_name=m_drafter,
                instruction=query,
                evidence="\n".join(
                    [f"[{idx}] {doc}" for idx, doc in enumerate(subset, start=1)]
                ),
            )
            for subset in sampled_docs
        ]
    )
    logger.info("RAG Drafting done in {s:.4f} seconds.", s=perf_counter() - _now)

    # RAG Verifier
    logger.info("Doing RAG Verification...")
    _now: float = perf_counter()
    rag_verifications: list[tuple[str, float]] = await asyncio.gather(
        *[
            rag_verifier_generator(
                client=client,
                model_name=m_verifier,
                instruction=query,
                evidence="\n".join(
                    [f"[{idx}] {doc}" for idx, doc in enumerate(subset, start=1)]
                ),
                response=rag_drafting_response.response,
                rationale=rag_drafting_response.rationale,
            )
            for subset, (rag_drafting_response, _) in zip(sampled_docs, rag_drafts)
        ]
    )
    logger.info("RAG Verification done in {s:.4f} seconds.", s=perf_counter() - _now)

    best_answer: int = np.argmax(
        p_draft * p_self
        for (_, p_draft), (_, p_self) in zip(rag_drafts, rag_verifications)
    )
    logger.info("Entire process done in {s:.4f} seconds.", s=perf_counter() - _start)
    print(f"\nQuestion:\n ------ \n{query}\n\n")
    print(f"Response:\n ------ \n{rag_drafts[best_answer][0].response}")
    return rag_drafts[best_answer][0].response



async def base_rag(
    query: str,
    embedding_model: str,
    collection_name: str,
    client: AsyncOpenAI,
    qdrant_client: AsyncQdrantClient,
    generation_model: str,
) -> str:
    _start = perf_counter()

    # Generate query vector embedding
    logger.info("Generating query vector...")
    _now: float = perf_counter()
    query_vector: Any = await client.embeddings.create(
        input=query, model=embedding_model
    )
    query_vector: list[float] = query_vector.data[0].embedding
    logger.info("Query vector generated in {s:.4f} seconds.", s=perf_counter() - _now)

    # Fetching relevant documents
    logger.info("Fetching relevant documents...")
    _now: float = perf_counter()
    out: list[models.ScoredPoint] = await qdrant_client.search(
        collection_name=collection_name, query_vector=query_vector, with_vectors=True
    )
    logger.info("Documents retrieved in {s:.4f} seconds.", s=perf_counter() - _now)

    # Base RAG
    logger.info("Generating response...")
    prompt: str = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Evidence: {evidence} 

    ### Instruction: {instruction}

    ### Response:"""

    completion: Any = await client.chat.completions.create(
        model=generation_model,
        messages=[
            {
                "role": "system",
                "content": prompt.format(
                    instruction=query,
                    evidence="\n".join(
                        [
                            f"[{idx}] {point.payload.get('content')}"
                            for idx, point in enumerate(out, start=1)
                        ]
                    ),
                ),
            }
        ],
        temperature=0.0,
        logprobs=True,
    )
    response: str = completion.choices[0].message.content
    logger.info("Response generated in {s:.4f} seconds.", s=perf_counter() - _now)

    logger.info("Entire process done in {s:.4f} seconds.", s=perf_counter() - _start)
    print(f"\nQuestion:\n ------ \n{query}\n\n")
    print(f"Response:\n ------ \n{response}")
    return response

# %%
    # Example call to base_rag (uncomment and adjust as needed)
    # final_answer: str = await base_rag(
    #     query="What is Query2doc?",
    #     embedding_model=embedding_model,
    #     collection_name=collection_name,
    #     client=openai_client,
    #     qdrant_client=qdrant_client,
    #     generation_model=m_verifier,
    # )

if __name__ == "__main__":
    asyncio.run(main())


