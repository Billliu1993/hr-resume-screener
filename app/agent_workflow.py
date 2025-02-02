from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Annotated
import operator
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .vector_db import VectorDB

load_dotenv()


# Pydantic models
class Subquery(BaseModel):
    query_topic: str = Field(description="Topic of the query to be searched")
    query_text: str = Field(description="Full text of the query to be searched")


class Subqueries(BaseModel):
    subqueries: List[Subquery] = Field(description="List of subqueries to be searched")


class JobDescription(BaseModel):
    title: str = Field(description="Job title")
    location: str = Field(description="Job location")
    departments: List[str] = Field(description="List of departments")
    metadata: List[str] = Field(description="Additional metadata tags for job function")
    description: str = Field(description="Job description text")


class RAGResult(BaseModel):
    chunk_heading: str = Field(description="Heading of the chunk")
    text: str = Field(description="Text of the chunk")


class Candidate(BaseModel):
    profile_id: str = Field(description="Unique identifier for the candidate profile")
    reason: str = Field(description="Reason for selecting the candidate")
    relavent_text_summary: str = Field(description="Summary of the relevant text from the candidate's profile")


# LLM and structured output
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, verbose=True)
subquery_generator = llm.with_structured_output(Subqueries)
candidate_generator = llm.with_structured_output(Candidate)

# VectorDB
db = VectorDB("linkedin_profiles")
db.load_db()

# States
class State(TypedDict):
    job_description: JobDescription
    subqueries: list[Subquery]
    completed_subqueries: Annotated[list, operator.add]
    candidates: list[RAGResult]
    final_candidates: list[Candidate]


class WorkerState(TypedDict):
    subquery: Subquery
    completed_subqueries: Annotated[list, operator.add]


# Nodes
def orchestrator(state: State):
    """Orchestrator that generates subqueries"""
    system_prompt = """
    You are a helpful talent acquisition AI. You separate the provided job description into 3-4 more focused subqueries for efficient resume retrieval. 
    Make sure every single relevant aspect of the job description is covered in at least one query. You may choose to remove irrelevant information that doesn't contribute to finding resumes such as the expected salary of the job, the ID of the job, the duration of the contract, etc.
    Only use the information provided in the initial query. Do not make up any requirements of your own.
    Each subquery should contains the following information:
    - query_topic: The asepct of the job description that the subquery is focusing on
    - query_text: The full text of the subquery    
    """
    subqueries = subquery_generator.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Separate this provided job description into 3-4 more focused subqueries: {state['job_description'].model_dump_json()}"
            ),
        ]
    )
    return {"subqueries": subqueries.subqueries}

def worker(state: WorkerState):
    """Worker that retrieves resumes based on subqueries"""
    subquery = state["subquery"]
    formatted_subquery = f"{subquery.query_topic}: {subquery.query_text}"
    rag_results = db.search(formatted_subquery, k=2, similarity_threshold=0.2)
    processed_results = [result['metadata'] for result in rag_results] if len(rag_results) > 0 else []

    return {"completed_subqueries": processed_results}

def reranker(state: State):
    """Reranks the candidates based on relevance"""
    # TODO: this is just a placeholder for the reranker logic
    rerank_list = []
    for result in state["completed_subqueries"]:
        formatted_result = RAGResult(
            chunk_heading=result["chunk_heading"],
            text=result["text"]
        )
        rerank_list.append(formatted_result)
        if len(rerank_list) > 2:
            break
    return {"candidates": rerank_list}

def summarizer(state: State):
    """summarize the candidate results"""
    system_prompt = """
    You are a helpful talent acquisition AI summarizer. You complete the following tasks in order:
    1. Summarize the retrieved texts from resume. The texts can be found in the <text> from the retrieved resume. 
    2. Compare the provided job description with the summary of the retrieved texts from resume
    3. Provide a brief summary on why you think they are matching. 
    Only focus on the matched part between the job description and resume. Do not make up any requirements of your own.
    Each candidate profile summary should contain the following information:
    - profile_id: the unique identifier of the candidate profile. It should be the same as the <chunk_heading> from the retrieved resume.
    - reason: the summary on why you think this candidate matches the job description
    - relavent_text_summary: the summary of the relevant text from the candidate's profile. You should use the summary you prepared from step 1.
    """

    candidates = state["candidates"]
    final_candidates = []

    for candidate in candidates:
        candidate_summary = candidate_generator.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"Summarize the candidate profile: {candidate.model_dump_json()} based on the job description: {state['job_description'].model_dump_json()}"
                ),
            ]
        )
        final_candidates.append(candidate_summary)

    return {"final_candidates": final_candidates}


# conditional edges
def assign_worker(state: State):
    """Assigns a worker to each subquery"""

    return [Send("worker", {"subquery": subquery}) for subquery in state["subqueries"]]

# workflow
def build_workflow():
    builder = StateGraph(State)

    # add nodes
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("worker", worker)
    builder.add_node("reranker", reranker)
    builder.add_node("summarizer", summarizer)
    
    # add edges
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", assign_worker, ["worker"])
    builder.add_edge("worker", "reranker")
    builder.add_edge("reranker", "summarizer")
    builder.add_edge("summarizer", END)

    # compile workflow
    workflow = builder.compile()

    return workflow
