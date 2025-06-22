# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import logging
import re
from collections.abc import AsyncGenerator
from typing import Literal

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types
from pydantic import BaseModel, Field
from google.genai import types as genai_types

from .config import config

from typing import List
import datetime

from app.prompt import PLANNER_AGENT_INSTR, LEGAL_QA_INSTRUCTION, CASE_STUDY_AGENT_INSTR, SUMMARIZATION_AGENT_INSTR, INTERACTIVE_AGENT_INSTR

# --- Structured Output Models ---
class SearchQuery(BaseModel):
    """Model representing a specific search query for web search."""

    search_query: str = Field(
        description="A highly specific and targeted query for web search."
    )


class Feedback(BaseModel):
    """Model for providing evaluation feedback on research quality."""

    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result. 'pass' if the research is sufficient, 'fail' if it needs revision."
    )
    comment: str = Field(
        description="Detailed explanation of the evaluation, highlighting strengths and/or weaknesses of the research."
    )
    follow_up_queries: list[SearchQuery] | None = Field(
        default=None,
        description="A list of specific, targeted follow-up search queries needed to fix research gaps. This should be null or empty if the grade is 'pass'.",
    )



# --- Callbacks ---
def collect_research_sources_callback(callback_context: CallbackContext) -> None:
    """Collects and organizes web-based research sources and their supported claims from agent events.

    This function processes the agent's `session.events` to extract web source details (URLs,
    titles, domains from `grounding_chunks`) and associated text segments with confidence scores
    (from `grounding_supports`). The aggregated source information and a mapping of URLs to short
    IDs are cumulatively stored in `callback_context.state`.

    Args:
        callback_context (CallbackContext): The context object providing access to the agent's
            session events and persistent state.
    """
    session = callback_context._invocation_context.session
    url_to_short_id = callback_context.state.get("url_to_short_id", {})
    sources = callback_context.state.get("sources", {})
    id_counter = len(url_to_short_id) + 1
    for event in session.events:
        if not (event.grounding_metadata and event.grounding_metadata.grounding_chunks):
            continue
        chunks_info = {}
        for idx, chunk in enumerate(event.grounding_metadata.grounding_chunks):
            if not chunk.web:
                continue
            url = chunk.web.uri
            title = (
                chunk.web.title
                if chunk.web.title != chunk.web.domain
                else chunk.web.domain
            )
            if url not in url_to_short_id:
                short_id = f"src-{id_counter}"
                url_to_short_id[url] = short_id
                sources[short_id] = {
                    "short_id": short_id,
                    "title": title,
                    "url": url,
                    "domain": chunk.web.domain,
                    "supported_claims": [],
                }
                id_counter += 1
            chunks_info[idx] = url_to_short_id[url]
        if event.grounding_metadata.grounding_supports:
            for support in event.grounding_metadata.grounding_supports:
                confidence_scores = support.confidence_scores or []
                chunk_indices = support.grounding_chunk_indices or []
                for i, chunk_idx in enumerate(chunk_indices):
                    if chunk_idx in chunks_info:
                        short_id = chunks_info[chunk_idx]
                        confidence = (
                            confidence_scores[i] if i < len(confidence_scores) else 0.5
                        )
                        text_segment = support.segment.text if support.segment else ""
                        sources[short_id]["supported_claims"].append(
                            {
                                "text_segment": text_segment,
                                "confidence": confidence,
                            }
                        )
    callback_context.state["url_to_short_id"] = url_to_short_id
    callback_context.state["sources"] = sources


def citation_replacement_callback(
    callback_context: CallbackContext,
) -> genai_types.Content:
    """Replaces citation tags in a report with Markdown-formatted links.

    Processes 'final_cited_report' from context state, converting tags like
    `<cite source="src-N"/>` into hyperlinks using source information from
    `callback_context.state["sources"]`. Also fixes spacing around punctuation.

    Args:
        callback_context (CallbackContext): Contains the report and source information.

    Returns:
        genai_types.Content: The processed report with Markdown citation links.
    """
    final_report = callback_context.state.get("final_cited_report", "")
    sources = callback_context.state.get("sources", {})

    def tag_replacer(match: re.Match) -> str:
        short_id = match.group(1)
        if not (source_info := sources.get(short_id)):
            logging.warning(f"Invalid citation tag found and removed: {match.group(0)}")
            return ""
        display_text = source_info.get("title", source_info.get("domain", short_id))
        return f" [{display_text}]({source_info['url']})"

    processed_report = re.sub(
        r'<cite\s+source\s*=\s*["\']?\s*(src-\d+)\s*["\']?\s*/>',
        tag_replacer,
        final_report,
    )
    processed_report = re.sub(r"\s+([.,;:])", r"\1", processed_report)
    callback_context.state["final_report_with_citations"] = processed_report
    return genai_types.Content(parts=[genai_types.Part(text=processed_report)])


# --- Custom Agent for Loop Control ---
class EscalationChecker(BaseAgent):
    """Checks research evaluation and escalates to stop the loop if grade is 'pass'."""

    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        evaluation_result = ctx.session.state.get("research_evaluation")
        if evaluation_result and evaluation_result.get("grade") == "pass":
            logging.info(
                f"[{self.name}] Research evaluation passed. Escalating to stop loop."
            )
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            logging.info(
                f"[{self.name}] Research evaluation failed or not found. Loop will continue."
            )
            # Yielding an event without content or actions just lets the flow continue.
            yield Event(author=self.name)



# --- TOOLS ---
tools = [google_search]

# --- AGENT DEFINITIONS ---

plan_generator = LlmAgent(
    name="plan_generator",
    model="gemini-2.0-flash",
    description="Generates a 4-5 line research plan.",
    instruction=f"""
    You are a research strategist creating concise, action-oriented research plans.
    * Output must be 4-5 bullet points.
    * Use verbs like "Analyze", "Investigate", "Compare".
    * Avoid summaries or factual answers.
    * Search only to disambiguate the topic.
    Date: {datetime.datetime.now().strftime('%Y-%m-%d')}
    """,
    tools=tools
)

section_planner = LlmAgent(
    name="section_planner",
    model="gemini-2.0-flash",
    description="Breaks the research plan into report sections.",
    instruction="""
    Create a markdown report outline from a given research plan. Include 4–6 sections.
    Do not include references or citations.
    """,
    output_key="report_sections"
)

section_researcher = LlmAgent(
    name="section_researcher",
    model=config.worker_model,
    description="Performs initial web research per section.",
    planner = BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(
            include_thoughts=True,
        )
    ),
    instruction="""
    For each section marked for research, generate 4–5 search queries.
    Run searches and synthesize a detailed summary per section.
    """,
    tools=tools,
    output_key="section_research_findings",
    after_agent_callback=collect_research_sources_callback
)



research_evaluator = LlmAgent(
    name="research_evaluator",
    model="gemini-2.5-pro",
    description="Evaluates research and suggests follow-up queries.",
    instruction="""
    Critically assess completeness, clarity, and quality of research.
    * Grade: "pass" or "fail"
    * If "fail", provide reasoning and 5–7 follow-up queries.
    Date: {datetime.datetime.now().strftime('%Y-%m-%d')}
    """,
    output_key="research_evaluation",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True
)

enhanced_search_executor = LlmAgent(
    name="enhanced_search_executor",
    model=config.worker_model,
    description="Executes follow-up queries and updates findings.",
    planner = BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(
            include_thoughts=True,
        )
    ),
    instruction="""
    Use feedback to run follow-up searches. Combine with previous results into updated research.
    """,
    tools=tools,
    output_key="section_research_findings",
    after_agent_callback=collect_research_sources_callback
)


report_composer = LlmAgent(
    name="report_composer_with_citations",
    model="gemini-2.5-pro",
    description="Generates a fully cited report from research.",
    instruction="""
    Convert markdown outline and research findings into a professional report.
    Use <cite source=\"src-ID\" /> tags for inline citations. Do not list references separately.
    """,
    output_key="final_cited_report",
    after_agent_callback=citation_replacement_callback
)

# --- PIPELINE AGENT ---

research_pipeline = SequentialAgent(
    name="research_pipeline",
    description="Executes a full research loop and report generation.",
    sub_agents=[
        section_planner,
        section_researcher,
        LoopAgent(
            name="iterative_refinement_loop",
            max_iterations=2,
            sub_agents=[
                research_evaluator,
                enhanced_search_executor
            ]
        ),
        report_composer
    ]
)

# --- INTERACTIVE ENTRYPOINT ---

interactive_planner_agent = LlmAgent(
    name="interactive_planner_agent",
    model="gemini-2.0-flash",
    description="Works with the user to define and execute research goals.",
    instruction=INTERACTIVE_AGENT_INSTR,
    sub_agents=[research_pipeline],
    tools=[
        AgentTool(plan_generator),
        AgentTool(research_pipeline),
    ],
    output_key="research_plan"
)

# --- LEGAL WORKFLOW AGENTS ---

summary_simulation_agent = LlmAgent(
    name="SummarySimulationAgent",
    model="gemini-2.0-flash",
    instruction= SUMMARIZATION_AGENT_INSTR,
    description="Clause-by-clause analyzer with risk detection.",
    output_key="summary_simulation"
)

case_study_agent = LlmAgent(
    name="CaseStudyAgent",
    model="gemini-2.0-flash",
    instruction= CASE_STUDY_AGENT_INSTR,
    description="Interactive legal case study generator.",
    output_key="case_study"
)



class LegalQAAgent(LlmAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def run(self, input_data: dict):
        question = input_data.get("question")
        summary = input_data.get("summary")

        # Use Google Search manually
        search_results = await google_search.invoke(question)

        # Build the prompt
        prompt = LEGAL_QA_INSTRUCTION + "\n\n"
        if summary:
            prompt += f"Summary:\n{summary}\n\n"
        prompt += f"Search Results:\n{search_results}\n\n"
        prompt += f"Question:\n{question}"

        # Invoke the LLM
        return await self.invoke(prompt)

# THIS is your planner-compatible agent
legal_qa_search_agent = LegalQAAgent(
    name="LegalQAAgent",
    description="Answers legal questions using Google Search manually and responds in JSON format.",
    instruction=LEGAL_QA_INSTRUCTION,
)


# --- MASTER ROUTING PLANNER ---


planner_agent = LlmAgent(
    name="LexAIPlannerAgent",
    model="gemini-2.0-flash",
    instruction=PLANNER_AGENT_INSTR,
    sub_agents=[summary_simulation_agent, case_study_agent, legal_qa_search_agent, interactive_planner_agent],
    description="Top-level planner agent routing to task-specific sub-agents."
)

# Root agent for system startup
root_agent = planner_agent
