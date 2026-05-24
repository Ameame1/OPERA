from __future__ import annotations

import re
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .llm import ChatLLM
from .prompts import (
    ANALYSIS_SYSTEM,
    FINAL_SYSTEM,
    PLAN_SYSTEM,
    REWRITE_SYSTEM,
    build_analysis_prompt,
    build_final_synthesis_prompt,
    build_plan_prompt,
    build_rewrite_prompt,
)
from .retriever import BGEM3FaissRetriever
from .schema import AnalysisResult, Document, PlanStep, RewriteResult
from .utils import extract_json


@dataclass
class PipelineConfig:
    top_k: int = 5
    top_k_schedule: Optional[List[int]] = None
    max_steps: int = 6
    max_rewrites: int = 2
    max_docs_in_prompt: int = 5
    max_doc_chars: int = 900
    plan_max_tokens: int = 512
    analysis_max_tokens: int = 512
    rewrite_max_tokens: int = 256
    final_max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    final_synthesis: bool = False
    stop_on_step_failure: bool = True
    multi_query_dependencies: bool = False
    max_dependency_queries: int = 4
    include_original_query: bool = False
    direct_answer_fallback: bool = False


class OperaPipeline:
    """Inference-only OPERA pipeline: plan -> retrieve -> rewrite -> reason."""

    def __init__(
        self,
        *,
        planner_llm: ChatLLM,
        analysis_llm: ChatLLM,
        rewrite_llm: ChatLLM,
        retriever: BGEM3FaissRetriever,
        config: Optional[PipelineConfig] = None,
    ):
        self.planner_llm = planner_llm
        self.analysis_llm = analysis_llm
        self.rewrite_llm = rewrite_llm
        self.retriever = retriever
        self.config = config or PipelineConfig()

    def run(self, question: str, *, question_id: Optional[str] = None) -> Dict[str, Any]:
        trace: Dict[str, Any] = {
            "id": question_id,
            "question": question,
            "config": asdict(self.config),
            "retriever": self.retriever.info(),
            "plan": None,
            "steps": [],
            "stats": {
                "retrieval_calls": 0,
                "rewrite_calls": 0,
                "analysis_calls": 0,
                "completed_steps": 0,
            },
        }

        plan_raw = self.planner_llm.generate(
            [{"role": "system", "content": PLAN_SYSTEM}, {"role": "user", "content": build_plan_prompt(question)}],
            max_tokens=self.config.plan_max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        plan = self._parse_plan(plan_raw, question)
        plan = plan[: self.config.max_steps]
        trace["plan"] = {
            "raw_response": plan_raw,
            "steps": [step.to_dict() for step in plan],
        }

        answers: Dict[int, str] = {}
        executed_steps: List[Dict[str, Any]] = []
        failed = False

        for step in plan:
            filled_subgoal = self._fill_placeholders(step.subgoal, answers)
            if step.dependencies:
                filled_subgoal = self._ensure_dependency_context(
                    filled_subgoal,
                    step.subgoal,
                    step.dependencies,
                    answers,
                )
            step_trace: Dict[str, Any] = {
                "step_id": step.step_id,
                "subgoal": step.subgoal,
                "filled_subgoal": filled_subgoal,
                "dependencies": step.dependencies,
                "attempts": [],
                "final_answer": "",
                "status": "failed",
            }

            missing_dependencies = [dep for dep in step.dependencies if dep not in answers]
            if missing_dependencies:
                step_trace["failure_reason"] = f"missing dependencies: {missing_dependencies}"
                trace["steps"].append(step_trace)
                failed = True
                if self.config.stop_on_step_failure:
                    break
                continue

            identity_answer = self._identity_echo_answer(question, filled_subgoal)
            if identity_answer:
                answers[step.step_id] = identity_answer
                executed_steps.append(
                    {
                        "step_id": step.step_id,
                        "subgoal": step.subgoal,
                        "filled_subgoal": filled_subgoal,
                        "answer": identity_answer,
                    }
                )
                step_trace["final_answer"] = identity_answer
                step_trace["status"] = "completed"
                step_trace["attempts"].append(
                    {
                        "attempt": 1,
                        "query": filled_subgoal,
                        "retrieval_queries": [],
                        "top_k": 0,
                        "documents": [],
                        "analysis": {
                            "status": "yes",
                            "answer": identity_answer,
                            "analysis": "Identity anchor already appears in the original question; this step only binds the placeholder.",
                            "confidence": None,
                            "raw_response": "",
                        },
                        "rewrite": None,
                    }
                )
                trace["stats"]["completed_steps"] += 1
                trace["steps"].append(step_trace)
                continue

            current_query = filled_subgoal
            final_analysis: Optional[AnalysisResult] = None
            for attempt_idx in range(self.config.max_rewrites + 1):
                top_k = self._top_k_for_attempt(attempt_idx)
                retrieval_queries = self._build_retrieval_queries(
                    original_question=question,
                    current_query=current_query,
                    filled_subgoal=filled_subgoal,
                    dependencies=step.dependencies,
                    answers=answers,
                )
                docs, retrieval_queries = self._retrieve_with_expansions(
                    retrieval_queries,
                    top_k=top_k,
                    max_docs=max(self.config.max_docs_in_prompt, top_k),
                )
                docs = self._rerank_documents_for_subgoal(docs, filled_subgoal)
                trace["stats"]["retrieval_calls"] += len(retrieval_queries)

                analysis = self._analyze(question, filled_subgoal, docs, answers)
                trace["stats"]["analysis_calls"] += 1
                final_analysis = analysis

                attempt_trace: Dict[str, Any] = {
                    "attempt": attempt_idx + 1,
                    "query": current_query,
                    "retrieval_queries": retrieval_queries,
                    "top_k": top_k,
                    "documents": [
                        doc.to_dict(max_chars=self.config.max_doc_chars)
                        for doc in docs
                    ],
                    "analysis": analysis.to_dict(),
                    "rewrite": None,
                }

                if analysis.is_sufficient():
                    answer = analysis.answer.strip()
                    answers[step.step_id] = answer
                    executed_steps.append(
                        {
                            "step_id": step.step_id,
                            "subgoal": step.subgoal,
                            "filled_subgoal": filled_subgoal,
                            "answer": answer,
                        }
                    )
                    step_trace["final_answer"] = answer
                    step_trace["status"] = "completed"
                    step_trace["attempts"].append(attempt_trace)
                    trace["stats"]["completed_steps"] += 1
                    break

                if attempt_idx < self.config.max_rewrites:
                    rewrite = self._rewrite(
                        original_question=question,
                        subgoal=filled_subgoal,
                        failure_info=analysis.analysis or "Documents are insufficient.",
                        documents=docs,
                        previous_answers=answers,
                    )
                    trace["stats"]["rewrite_calls"] += 1
                    attempt_trace["rewrite"] = rewrite.to_dict()
                    current_query = rewrite.rewritten_query or current_query

                step_trace["attempts"].append(attempt_trace)

            if step_trace["status"] != "completed":
                failed = True
                if final_analysis:
                    step_trace["failure_reason"] = final_analysis.analysis
                trace["steps"].append(step_trace)
                if self.config.stop_on_step_failure:
                    break
            else:
                trace["steps"].append(step_trace)

        final_answer = ""
        final_analysis = ""
        if answers and not (failed and not self.config.final_synthesis):
            if self.config.final_synthesis:
                final = self._final_synthesis(question, answers, executed_steps)
                final_answer = final.get("answer", "").strip()
                final_analysis = final.get("analysis", "")
                trace["final_synthesis"] = final
            else:
                last_step_id = max(answers)
                final_answer = answers[last_step_id]
        if not final_answer:
            final_answer = "Not found in retrieved documents"
        pipeline_success = bool(answers) and not failed
        final_answer, postprocess = self._postprocess_final_answer(
            question,
            final_answer,
            executed_steps,
            pipeline_success=pipeline_success,
        )
        if postprocess:
            trace["final_postprocess"] = postprocess

        if self.config.direct_answer_fallback and self._is_not_found_answer(final_answer):
            direct = self._direct_answer_fallback(question)
            trace["direct_answer_fallback"] = direct
            trace["stats"]["retrieval_calls"] += len(direct.get("retrieval_queries") or [])
            trace["stats"]["analysis_calls"] += int(bool(direct.get("analysis")))
            direct_analysis = direct.get("analysis") or {}
            direct_answer = str(direct_analysis.get("answer") or "").strip()
            direct_status = str(direct_analysis.get("status") or "").lower()
            if direct_status == "yes" and not self._is_not_found_answer(direct_answer):
                final_answer = direct_answer
                pipeline_success = True

        trace["final_answer"] = final_answer
        trace["final_analysis"] = final_analysis
        trace["success"] = pipeline_success
        return trace

    def _top_k_for_attempt(self, attempt_idx: int) -> int:
        if not self.config.top_k_schedule:
            return self.config.top_k
        idx = min(attempt_idx, len(self.config.top_k_schedule) - 1)
        return max(1, int(self.config.top_k_schedule[idx]))

    def _retrieve_documents(
        self,
        queries: List[str],
        *,
        top_k: int,
        max_docs: int,
    ) -> List[Document]:
        per_query_docs: List[List[Document]] = []
        for query in queries:
            docs = self.retriever.search(query, top_k=top_k)
            for doc in docs:
                doc.metadata = dict(doc.metadata or {})
                doc.metadata.setdefault("retrieval_query", query)
            per_query_docs.append(docs)

        merged: List[Document] = []
        seen = set()
        for rank_idx in range(top_k):
            for docs in per_query_docs:
                if rank_idx >= len(docs):
                    continue
                doc = docs[rank_idx]
                key = (doc.doc_id, doc.title, doc.content[:120])
                if key in seen:
                    continue
                seen.add(key)
                doc.rank = len(merged) + 1
                merged.append(doc)
                if len(merged) >= max_docs:
                    return merged
        return merged

    def _retrieve_with_expansions(
        self,
        queries: List[str],
        *,
        top_k: int,
        max_docs: int,
    ) -> tuple[List[Document], List[str]]:
        base_docs = self._retrieve_documents(queries, top_k=top_k, max_docs=max_docs)
        expansion_queries = self._build_canonical_title_queries(queries, base_docs)
        if not expansion_queries:
            return base_docs, queries

        extra_docs = self._retrieve_documents(expansion_queries, top_k=top_k, max_docs=max_docs)
        merged = self._merge_document_lists([base_docs, extra_docs], max_docs=max_docs)
        return merged, [*queries, *expansion_queries]

    @classmethod
    def _rerank_documents_for_subgoal(cls, docs: List[Document], subgoal: str) -> List[Document]:
        requested_titles = cls._extract_requested_titles(subgoal)
        if not requested_titles:
            return docs

        requested_norms = [cls._normalize_title_for_match(title) for title in requested_titles]

        def score(doc: Document) -> tuple[int, int]:
            title_norm = cls._normalize_title_for_match(doc.title)
            if title_norm in requested_norms:
                return (0, doc.rank)
            if any(title_norm.startswith(f"{requested} ") or requested.startswith(f"{title_norm} ") for requested in requested_norms):
                return (1, doc.rank)
            if any(requested in cls._normalize_title_for_match(f"{doc.title} {doc.content[:200]}") for requested in requested_norms):
                return (2, doc.rank)
            return (3, doc.rank)

        ranked = sorted(docs, key=score)
        for idx, doc in enumerate(ranked, start=1):
            doc.rank = idx
        return ranked

    @classmethod
    def _extract_requested_titles(cls, text: str) -> List[str]:
        titles: List[str] = []
        for match in re.findall(r'"([^"]+)"|\'([^\']+)\'', text or ""):
            title = next((item for item in match if item), "").strip()
            if title:
                titles.append(title)
        return titles

    @classmethod
    def _normalize_title_for_match(cls, text: str) -> str:
        text = re.sub(r"\s*\([^)]*\)\s*$", "", text or "")
        return cls._normalize_for_match(text)

    @staticmethod
    def _merge_document_lists(document_lists: List[List[Document]], *, max_docs: int) -> List[Document]:
        merged: List[Document] = []
        seen = set()
        max_len = max((len(docs) for docs in document_lists), default=0)
        for rank_idx in range(max_len):
            for docs in document_lists:
                if rank_idx >= len(docs):
                    continue
                doc = docs[rank_idx]
                key = (doc.doc_id, doc.title, doc.content[:120])
                if key in seen:
                    continue
                seen.add(key)
                doc.rank = len(merged) + 1
                merged.append(doc)
                if len(merged) >= max_docs:
                    return merged
        return merged

    @classmethod
    def _build_canonical_title_queries(cls, queries: List[str], docs: List[Document]) -> List[str]:
        expansion_queries: List[str] = []
        for query in queries:
            query_norm = cls._normalize_for_match(query)
            for doc in docs[:8]:
                title = (doc.title or "").strip()
                if not title:
                    continue
                title_norm = cls._normalize_for_match(title)
                if not title_norm or title_norm not in query_norm:
                    continue
                alias = cls._extract_canonical_alias(title, doc.content)
                if not alias:
                    continue
                alias_norm = cls._normalize_for_match(alias)
                if alias_norm == title_norm or title_norm not in alias_norm:
                    continue
                variant = re.sub(re.escape(title), alias, query, count=1, flags=re.IGNORECASE)
                variant = re.sub(r"\s+", " ", variant).strip()
                if variant and variant not in queries and variant not in expansion_queries:
                    expansion_queries.append(variant)
                    if len(expansion_queries) >= 2:
                        return expansion_queries
        return expansion_queries

    @staticmethod
    def _extract_canonical_alias(title: str, content: str) -> str:
        text = re.sub(r"\s+", " ", (content or "").strip())
        if not text:
            return ""
        delimiter_positions = [
            idx
            for delimiter in ("(", " is ", " was ", " are ", " were ", ",")
            for idx in [text.find(delimiter)]
            if 5 <= idx <= 90
        ]
        candidate = text[: min(delimiter_positions)] if delimiter_positions else text[:90]
        title_norm = OperaPipeline._normalize_for_match(title)
        alias = candidate.strip(" ;:-,")
        alias_norm = OperaPipeline._normalize_for_match(alias)
        if not (5 <= len(alias) <= 90 and title_norm):
            return ""
        title_tokens = title_norm.split()
        alias_tokens = alias_norm.split()
        if len(title_tokens) == 1 and (not alias_tokens or alias_tokens[0] != title_tokens[0]):
            return ""
        if OperaPipeline._contains_token_sequence(alias_tokens, title_tokens):
            return alias
        return ""

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

    @staticmethod
    def _contains_token_sequence(tokens: List[str], needle: List[str]) -> bool:
        if not needle or len(needle) > len(tokens):
            return False
        width = len(needle)
        return any(tokens[idx : idx + width] == needle for idx in range(len(tokens) - width + 1))

    def _build_retrieval_queries(
        self,
        *,
        original_question: str,
        current_query: str,
        filled_subgoal: str,
        dependencies: List[int],
        answers: Dict[int, str],
    ) -> List[str]:
        queries = [current_query]
        if self.config.include_original_query and original_question not in queries:
            queries.append(original_question)
        for candidate_query in self._build_candidate_verification_queries(current_query):
            if candidate_query not in queries:
                queries.append(candidate_query)
        for relation_query in self._build_relation_keyword_queries(filled_subgoal):
            if relation_query not in queries:
                queries.append(relation_query)
        if not self.config.multi_query_dependencies or not dependencies:
            return queries

        for dep in dependencies:
            answer = answers.get(dep, "").strip()
            if not answer:
                continue
            for concise_query in self._build_concise_dependency_queries(filled_subgoal, answer):
                if concise_query not in queries:
                    queries.append(concise_query)
            for candidate in self._split_candidate_answer(answer)[: self.config.max_dependency_queries]:
                if not candidate or candidate == answer:
                    continue
                variant = filled_subgoal
                if answer in variant:
                    variant = variant.replace(answer, candidate)
                else:
                    variant = f"{filled_subgoal}\nDependency candidate: {candidate}"
                if variant not in queries:
                    queries.append(variant)
        return queries

    @classmethod
    def _build_relation_keyword_queries(cls, filled_subgoal: str) -> List[str]:
        text = (filled_subgoal or "").strip()
        if not text:
            return []

        queries: List[str] = []

        def add(query: str) -> None:
            query = re.sub(r"\s+", " ", query).strip(" ?")
            if query and query not in queries:
                queries.append(query)

        possessive = cls._extract_possessive_relation(text)
        if possessive:
            target, relation = possessive
            keyword_map = {
                "spouse": "spouse wife husband married",
                "wife": "wife spouse married",
                "husband": "husband spouse married",
                "father": "father parents family",
                "mother": "mother parents family",
                "parent": "parents father mother family",
            }
            add(f"{target} {keyword_map.get(relation, relation)}")
            add(target)
            return queries[:3]

        for relation, keywords in [
            (r"spouse of", "spouse wife husband married"),
            (r"wife of", "wife spouse married"),
            (r"husband of", "husband spouse married"),
            (r"father of", "father parents family"),
            (r"mother of", "mother parents family"),
            (r"parent of", "parents father mother family"),
        ]:
            target = cls._extract_relation_target(text, [relation])
            if target:
                add(f"{target} {keywords}")
                add(target)
                return queries[:3]

        birth_subject = cls._extract_birth_subject(text)
        if birth_subject:
            add(f"{birth_subject} born birthplace place of birth")
            add(birth_subject)
            return queries[:3]

        death_target = cls._extract_death_subject(text)
        if death_target:
            add(f"{death_target} died death date")
            add(death_target)
            return queries[:3]

        location_target = cls._extract_location_subject(text)
        if location_target:
            add(f"{location_target} located province county city district")
            add(location_target)
            return queries[:3]

        label_target = cls._extract_relation_target(text, [r"record label of", r"label of"])
        if label_target:
            add(f"{label_target} record label")
            add(label_target)
            return queries[:3]

        law_target = cls._extract_relation_target(text, [r"law was passed by", r"law was signed by", r"law passed by", r"law signed by"])
        if law_target:
            add(f"{law_target} signed law legislation")
            add(f"{law_target} passed law")
            add(law_target)
            return queries[:3]

        return queries[:3]

    @staticmethod
    def _extract_possessive_relation(subgoal: str) -> Optional[tuple[str, str]]:
        patterns = [
            r"\b(?:who|which\s+person|what\s+person)\s+(?:is|was)\s+(.+?)'s\s+(spouse|wife|husband|father|mother|parent)\b",
            r"\b(.+?)'s\s+(spouse|wife|husband|father|mother|parent)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, subgoal or "", flags=re.IGNORECASE)
            if not match:
                continue
            target = match.group(1).strip(" .?")
            relation = match.group(2).lower()
            if target:
                return target, relation
        return None

    @staticmethod
    def _extract_death_subject(subgoal: str) -> str:
        patterns = [
            r"\bwhen\s+did\s+(.+?)\s+die\b",
            r"\bdate\s+of\s+death\s+of\s+(.+?)(?:\?|$)",
            r"\bdeath\s+date\s+of\s+(.+?)(?:\?|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, subgoal or "", flags=re.IGNORECASE)
            if match:
                return match.group(1).strip(" .?")
        return ""

    @staticmethod
    def _extract_location_subject(subgoal: str) -> str:
        patterns = [
            r"\bwhat\s+(?:province|county|city|district|state|country)\s+(?:is|was)\s+(.+?)\s+located\s+in\b",
            r"\bwhere\s+(?:is|was)\s+(.+?)\s+located\b",
            r"\bwhere\s+(.+?)\s+(?:is|was)\s+located\b",
            r"\b(?:province|county|city|district|state|country)\s+where\s+(.+?)\s+(?:is|was)\s+located\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, subgoal or "", flags=re.IGNORECASE)
            if match:
                return match.group(1).strip(" .?")
        return ""

    @classmethod
    def _build_candidate_verification_queries(cls, current_query: str) -> List[str]:
        marker = "Candidate entities to verify:"
        if marker.lower() not in (current_query or "").lower():
            return []
        prefix, _, suffix = current_query.partition(marker)
        base_query = prefix.strip(" .")
        candidates = cls._extract_capitalized_entity_candidates(suffix)
        if len(candidates) < 2 or not base_query:
            return []
        return [f"{candidate} {base_query}" for candidate in candidates[:4]]

    @classmethod
    def _extract_capitalized_entity_candidates(cls, text: str) -> List[str]:
        candidates: List[str] = []
        seen = set()
        question_words = {"which", "what", "who", "when", "where", "how"}
        for match in re.finditer(r"\b[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){1,5}\b", text or ""):
            candidate = match.group(0).strip(" ,.;:()[]{}")
            tokens = candidate.split()
            if not tokens or tokens[0].lower() in question_words:
                continue
            norm = cls._normalize_for_match(candidate)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            candidates.append(candidate)
        return candidates

    @classmethod
    def _augment_rewrite_with_candidate_entities(
        cls,
        rewritten_query: str,
        failure_info: str,
        documents: List[Document],
    ) -> str:
        failure_norm = (failure_info or "").lower()
        if not re.search(
            r"\b(multiple|candidate|candidates|several|which of|do not specify which|does not specify which|list(?:s|ed)?)\b",
            failure_norm,
        ):
            return rewritten_query

        candidates = cls._extract_capitalized_entity_candidates(failure_info)
        if len(candidates) < 2:
            preview = " ".join((doc.content or "")[:500] for doc in documents[:3])
            candidates = cls._extract_capitalized_entity_candidates(preview)
        if len(candidates) < 2:
            return rewritten_query

        query_norm = cls._normalize_for_match(rewritten_query)
        missing = [candidate for candidate in candidates if cls._normalize_for_match(candidate) not in query_norm]
        if not missing:
            return rewritten_query
        return f"{rewritten_query} Candidate entities to verify: {', '.join(candidates[:4])}"

    @classmethod
    def _identity_echo_answer(cls, original_question: str, subgoal: str) -> str:
        match = re.match(
            r"^\s*(?:who|what)\s+(?:is|was|are|were)\s+(.+?)\s*\?\s*$",
            subgoal or "",
            flags=re.IGNORECASE,
        )
        if not match:
            return ""

        entity = match.group(1).strip(" .")
        if not entity or entity.lower().startswith(("the ", "a ", "an ")):
            return ""
        if cls._looks_like_attribute_slot(entity):
            return ""
        if cls._normalize_for_match(entity) == cls._normalize_for_match(original_question):
            return ""
        if not cls._looks_like_named_anchor(entity):
            return ""

        entity_norm = cls._normalize_for_match(entity)
        original_norm = cls._normalize_for_match(original_question)
        if not entity_norm or entity_norm not in original_norm:
            return ""

        original_identity = re.match(
            r"^\s*(?:who|what)\s+(?:is|was|are|were)\s+(.+?)\s*\?\s*$",
            original_question or "",
            flags=re.IGNORECASE,
        )
        if original_identity and cls._normalize_for_match(original_identity.group(1)) == entity_norm:
            return ""
        return entity

    @staticmethod
    def _looks_like_named_anchor(entity: str) -> bool:
        if re.search(r"[\"']", entity or ""):
            return True
        tokens = re.findall(r"\b[\w'.-]+\b", entity or "")
        if not tokens:
            return False
        lowercase_title_words = {"of", "the", "and", "de", "da", "di", "du", "la", "le", "van", "von", "st", "st."}
        proper = [
            token
            for token in tokens
            if token[:1].isupper() or re.search(r"\d", token) or token.lower() in lowercase_title_words
        ]
        return len(proper) >= max(1, min(2, len(tokens)))

    @staticmethod
    def _looks_like_attribute_slot(entity: str) -> bool:
        text = (entity or "").lower()
        if re.search(
            r"\b(birthplace|birth\s+place|place\s+of\s+birth|deathplace|place\s+of\s+death|father|mother|parent|spouse|wife|husband|county|province|population|date|year|age|section|topic|name)\b",
            text,
        ):
            return True
        return bool(re.search(r"'s\s+[a-z][a-z -]{2,}$", text))

    @classmethod
    def _build_concise_dependency_queries(cls, filled_subgoal: str, answer: str) -> List[str]:
        answer = answer.strip()
        if not answer:
            return []

        patterns = [
            (r"\bpresident of\s+{answer}\b", f"Who is the president of {answer}?"),
            (r"\bcapital of\s+{answer}\b", f"What is the capital of {answer}?"),
            (r"\bcapitol of\s+{answer}\b", f"What is the capitol of {answer}?"),
            (r"\bowner of\s+{answer}\b", f"Who is the owner of {answer}?"),
            (r"\bpublisher of\s+{answer}\b", f"Who published {answer}?"),
            (r"\bspouse of\s+{answer}\b", f"Who is the spouse of {answer}?"),
            (r"\bfather of\s+{answer}\b", f"Who is the father of {answer}?"),
            (r"\bmother of\s+{answer}\b", f"Who is the mother of {answer}?"),
            (r"\bmember of\s+{answer}\b", f"Which person is a member of {answer}?"),
            (r"\bpart of\s+{answer}\b", f"Which person is part of {answer}?"),
        ]
        queries: List[str] = []
        for pattern, query in patterns:
            regex = pattern.format(answer=re.escape(answer))
            if re.search(regex, filled_subgoal, flags=re.IGNORECASE):
                queries.append(query)

        if re.search(r"\baward\b", filled_subgoal, flags=re.IGNORECASE) and re.search(
            re.escape(answer),
            filled_subgoal,
            flags=re.IGNORECASE,
        ):
            queries.append(f"What award was received by {answer}?")

        return queries

    def _analyze(
        self,
        original_question: str,
        subgoal: str,
        documents: List[Document],
        previous_answers: Dict[int, str],
    ) -> AnalysisResult:
        raw = self.analysis_llm.generate(
            [
                {"role": "system", "content": ANALYSIS_SYSTEM},
                {
                    "role": "user",
                    "content": build_analysis_prompt(
                        original_question,
                        subgoal,
                        documents,
                        previous_answers,
                        max_docs=self.config.max_docs_in_prompt,
                        max_chars=self.config.max_doc_chars,
                    ),
                },
            ],
            max_tokens=self.config.analysis_max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        try:
            parsed = extract_json(raw)
            status = str(parsed.get("status", "no")).lower()
            answer = str(parsed.get("answer") or "")
            analysis = str(parsed.get("analysis") or "")
            confidence = parsed.get("confidence")
            if status not in {"yes", "no"}:
                status = "yes" if answer.strip() else "no"
            result = AnalysisResult(
                status=status,
                answer=answer,
                analysis=analysis,
                confidence=confidence if isinstance(confidence, (int, float)) else None,
                raw_response=raw,
            )
            return self._validate_analysis_result(subgoal, documents, result)
        except Exception:
            recovered = self._recover_analysis_result(raw)
            if recovered is not None:
                return self._validate_analysis_result(subgoal, documents, recovered)
            return AnalysisResult(status="no", analysis=raw, raw_response=raw)

    def _validate_analysis_result(
        self,
        subgoal: str,
        documents: List[Document],
        result: AnalysisResult,
    ) -> AnalysisResult:
        if not result.is_sufficient():
            return result

        reason = ""
        if self._is_member_question(subgoal):
            reason = self._member_validation_failure(subgoal, result.answer)
            if reason:
                correction = self._member_relation_correction(subgoal, documents)
                if correction:
                    analysis = (result.analysis or "").strip()
                    if analysis:
                        analysis += " "
                    analysis += f"Validation corrected the member answer from {result.answer!r} to {correction!r}."
                    return AnalysisResult(
                        status="yes",
                        answer=correction,
                        analysis=analysis,
                        confidence=result.confidence,
                        raw_response=result.raw_response,
                    )
        if not reason and self._is_writer_question(subgoal):
            if not self._answer_has_writer_support(subgoal, result.answer, documents):
                reason = "answer is not supported by explicit writer/author/composer evidence"
        if not reason and self._is_spouse_question(subgoal):
            if not self._answer_has_exact_relation_target_binding(subgoal, result.answer, documents):
                reason = "spouse evidence is not bound to the exact requested person"
        if not reason and self._is_parent_relation_question(subgoal):
            parent_correction = self._parent_relation_correction(subgoal, documents)
            if parent_correction and self._normalize_for_match(parent_correction) != self._normalize_for_match(result.answer):
                analysis = (result.analysis or "").strip()
                if analysis:
                    analysis += " "
                analysis += f"Validation corrected the parent answer from {result.answer!r} to {parent_correction!r}."
                return AnalysisResult(
                    status="yes",
                    answer=parent_correction,
                    analysis=analysis,
                    confidence=result.confidence,
                    raw_response=result.raw_response,
                )
            if not self._answer_has_parent_relation_support(subgoal, result.answer, documents):
                reason = "parent evidence is not directly bound to the requested person"
        if not reason and self._is_counterpart_relation_question(subgoal):
            reason = self._counterpart_relation_validation_failure(subgoal, result.answer)
            if reason:
                correction = self._counterpart_relation_correction(
                    subgoal,
                    result.answer,
                    documents,
                    result.analysis,
                )
                if correction:
                    analysis = (result.analysis or "").strip()
                    if analysis:
                        analysis += " "
                    analysis += f"Validation corrected the counterpart answer from {result.answer!r} to {correction!r}."
                    return AnalysisResult(
                        status="yes",
                        answer=correction,
                        analysis=analysis,
                        confidence=result.confidence,
                        raw_response=result.raw_response,
                    )
        if not reason and self._is_featured_entity_question(subgoal):
            reason = self._featured_entity_validation_failure(subgoal, result.answer, documents)
        if not reason and self._is_actor_role_question(subgoal):
            if not self._answer_has_actor_role_support(subgoal, result.answer, documents):
                reason = "actor evidence is not bound to the requested role or character"
        if not reason:
            comparison_correction = self._date_comparison_correction(subgoal, result.answer, result.analysis)
            if comparison_correction and self._normalize_for_match(comparison_correction) != self._normalize_for_match(result.answer):
                analysis = (result.analysis or "").strip()
                if analysis:
                    analysis += " "
                analysis += f"Validation corrected the comparison answer from {result.answer!r} to {comparison_correction!r}."
                return AnalysisResult(
                    status="yes",
                    answer=comparison_correction,
                    analysis=analysis,
                    confidence=result.confidence,
                    raw_response=result.raw_response,
                )
        if not reason:
            reason = self._candidate_choice_validation_failure(subgoal, result.answer)
        if not reason and self._is_section_question(subgoal):
            reason = self._section_answer_validation_failure(result.answer)
            if reason:
                correction = self._section_answer_correction(documents)
                if correction:
                    analysis = (result.analysis or "").strip()
                    if analysis:
                        analysis += " "
                    analysis += f"Validation corrected the section answer from {result.answer!r} to {correction!r}."
                    return AnalysisResult(
                        status="yes",
                        answer=correction,
                        analysis=analysis,
                        confidence=result.confidence,
                        raw_response=result.raw_response,
                    )
        if not reason:
            reason = self._numeric_answer_validation_failure(subgoal, result.answer)
        if not reason:
            reason = self._descriptor_constraint_validation_failure(subgoal, result.answer, documents)
        if not reason and self._is_birthplace_question(subgoal):
            reason = self._birthplace_validation_failure(subgoal, result.answer, documents)
            if reason:
                correction = self._birthplace_correction(subgoal, documents)
                if correction:
                    analysis = (result.analysis or "").strip()
                    if analysis:
                        analysis += " "
                    analysis += f"Validation corrected the birthplace answer from {result.answer!r} to {correction!r}."
                    return AnalysisResult(
                        status="yes",
                        answer=correction,
                        analysis=analysis,
                        confidence=result.confidence,
                        raw_response=result.raw_response,
                    )
        if not reason and self._is_characterizer_question(subgoal):
            reason = self._characterizer_validation_failure(subgoal, result.answer, documents)
            if reason:
                correction = self._characterizer_relation_correction(subgoal, documents)
                if correction:
                    analysis = (result.analysis or "").strip()
                    if analysis:
                        analysis += " "
                    analysis += f"Validation corrected the characterizer answer from {result.answer!r} to {correction!r}."
                    return AnalysisResult(
                        status="yes",
                        answer=correction,
                        analysis=analysis,
                        confidence=result.confidence,
                        raw_response=result.raw_response,
                    )
        if not reason and self._is_alias_question(subgoal):
            correction = self._alias_relation_correction(subgoal, result.answer, documents, result.analysis)
            if correction and self._normalize_for_match(correction) != self._normalize_for_match(result.answer):
                analysis = (result.analysis or "").strip()
                if analysis:
                    analysis += " "
                analysis += f"Validation corrected the alias answer from {result.answer!r} to {correction!r}."
                return AnalysisResult(
                    status="yes",
                    answer=correction,
                    analysis=analysis,
                    confidence=result.confidence,
                    raw_response=result.raw_response,
                )

        if not reason:
            return result
        analysis = (result.analysis or "").strip()
        if analysis:
            analysis += " "
        analysis += f"Validation rejected the proposed answer: {reason}."
        return AnalysisResult(
            status="no",
            answer="",
            analysis=analysis,
            confidence=result.confidence,
            raw_response=result.raw_response,
        )

    @classmethod
    def _is_writer_question(cls, subgoal: str) -> bool:
        text = subgoal.lower()
        return bool(
            re.search(
                r"\b(who\s+(?:wrote|authored|composed)|who\s+(?:is|was)\s+(?:the\s+)?(?:writer|author|composer|songwriter)\s+of|(?:which|what)\s+person\s+(?:wrote|authored|composed))\b",
                text,
            )
        )

    @classmethod
    def _is_member_question(cls, subgoal: str) -> bool:
        return bool(re.search(r"\b(part of|member of|belongs to)\b", subgoal, flags=re.IGNORECASE))

    @classmethod
    def _is_spouse_question(cls, subgoal: str) -> bool:
        return bool(re.search(r"\b(spouse|wife|husband)\b", subgoal, flags=re.IGNORECASE))

    @classmethod
    def _is_parent_relation_question(cls, subgoal: str) -> bool:
        return bool(
            re.search(
                r"^\s*(?:who|which\s+person|what\s+person)\b.+\b(father|mother|parent)\s+of\b",
                subgoal,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _is_counterpart_relation_question(cls, subgoal: str) -> bool:
        return bool(
            re.search(
                r"\b(which|what|who)\b.+\b(with|between|shares?\s+a\s+border\s+with|connected\s+to|commission\s+of|friendship\s+with)\b",
                subgoal,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _is_featured_entity_question(cls, subgoal: str) -> bool:
        return bool(
            re.search(r"\b(figure|character|topic|entity).+\bfeatured\s+in\b", subgoal, flags=re.IGNORECASE)
        )

    @classmethod
    def _is_actor_role_question(cls, subgoal: str) -> bool:
        return bool(re.search(r"\b(who|which actor).+\b(play|plays|played|portray|portrays|portrayed)\b", subgoal, flags=re.IGNORECASE))

    @classmethod
    def _is_characterizer_question(cls, subgoal: str) -> bool:
        return bool(
            re.search(
                r"\b(characteri[sz]ed?|described?|called?|named?|referred\s+to)\b.+\bas\b",
                subgoal,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _is_alias_question(cls, subgoal: str) -> bool:
        return bool(
            re.search(
                r"\b(also known as|known as|another name|alternative name|called|nicknamed|referred to as)\b",
                subgoal,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _is_birthplace_question(cls, subgoal: str) -> bool:
        return bool(
            re.search(r"\bwhere\s+was\s+.+?\s+born\b|\bplace\s+of\s+birth\s+of\b", subgoal, flags=re.IGNORECASE)
        )

    @classmethod
    def _is_section_question(cls, subgoal: str) -> bool:
        return bool(re.search(r"\b(?:which|what)\s+section\b", subgoal, flags=re.IGNORECASE))

    @classmethod
    def _section_answer_validation_failure(cls, answer: str) -> str:
        answer_norm = cls._normalize_for_match(answer)
        if not answer_norm:
            return "empty section answer"
        ordinals = {
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
        }
        tokens = set(answer_norm.split())
        if "section" in tokens or tokens & ordinals or re.search(r"\b\d+(?:st|nd|rd|th)?\b", answer_norm):
            return ""
        return "section question returned a non-section answer"

    @classmethod
    def _section_answer_correction(cls, documents: List[Document]) -> str:
        pattern = re.compile(
            r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th)?)\s+section\b",
            flags=re.IGNORECASE,
        )
        for doc in documents:
            text = f"{doc.title}. {doc.content}"
            match = pattern.search(text)
            if match:
                return match.group(0).strip(" .,")
        return ""

    @classmethod
    def _candidate_choice_validation_failure(cls, subgoal: str, answer: str) -> str:
        if not re.search(r"\b(which|who|what)\b", subgoal, flags=re.IGNORECASE):
            return ""
        candidates = cls._extract_or_candidates(subgoal)
        if len(candidates) != 2:
            return ""
        answer_norm = cls._normalize_for_match(answer)
        if not answer_norm:
            return "empty candidate-comparison answer"
        candidate_norms = [cls._normalize_for_match(candidate) for candidate in candidates]
        if answer_norm in candidate_norms:
            return ""
        return "candidate-comparison answer is not one of the requested candidates"

    @classmethod
    def _date_comparison_correction(cls, subgoal: str, answer: str, analysis: str) -> str:
        candidates = cls._extract_or_candidates(subgoal)
        if len(candidates) != 2:
            return ""
        text_norm = cls._normalize_for_match(subgoal)
        if not re.search(r"\b(younger|older|born|died|death|released|came out)\b", text_norm):
            return ""

        dates: Dict[str, tuple[int, int, int]] = {}
        for candidate in candidates:
            parsed = cls._date_near_candidate(candidate, analysis)
            if parsed:
                dates[candidate] = parsed
        if len(dates) != 2:
            return ""

        if re.search(r"\b(younger|born later|later birth)\b", text_norm):
            return max(dates.items(), key=lambda item: item[1])[0]
        if re.search(r"\b(older|born first|earlier birth)\b", text_norm):
            return min(dates.items(), key=lambda item: item[1])[0]
        if re.search(r"\b(died later|death later)\b", text_norm):
            return max(dates.items(), key=lambda item: item[1])[0]
        if re.search(r"\b(died first|death first|died earlier)\b", text_norm):
            return min(dates.items(), key=lambda item: item[1])[0]
        if re.search(r"\b(came out first|released first|earlier release)\b", text_norm):
            return min(dates.items(), key=lambda item: item[1])[0]
        if re.search(r"\b(came out later|released later|more recently)\b", text_norm):
            return max(dates.items(), key=lambda item: item[1])[0]
        return ""

    @classmethod
    def _date_near_candidate(cls, candidate: str, text: str) -> Optional[tuple[int, int, int]]:
        if not candidate or not text:
            return None
        match = re.search(re.escape(candidate), text, flags=re.IGNORECASE)
        if not match:
            return None
        window = text[match.start() : min(len(text), match.end() + 160)]
        return cls._parse_date(window)

    @classmethod
    def _birthplace_validation_failure(cls, subgoal: str, answer: str, documents: List[Document]) -> str:
        subject = cls._extract_birth_subject(subgoal)
        subject_norm = cls._normalize_for_match(subject)
        answer_norm = cls._normalize_for_match(answer)
        if not subject_norm or not answer_norm:
            return ""
        for doc in documents:
            text_norm = cls._normalize_for_match(f"{doc.title} {doc.content}")
            title_norm = cls._normalize_for_match(doc.title)
            if answer_norm not in text_norm:
                continue
            if subject_norm in text_norm or title_norm == subject_norm or title_norm in subject_norm:
                if re.search(r"\b(born|birth|place of birth)\b", text_norm):
                    return ""
        return "birthplace answer is not bound to the requested person/entity"

    @classmethod
    def _birthplace_correction(cls, subgoal: str, documents: List[Document]) -> str:
        subject = cls._extract_birth_subject(subgoal)
        subject_norm = cls._normalize_for_match(subject)
        if not subject_norm:
            return ""
        for doc in documents:
            title_norm = cls._normalize_for_match(doc.title)
            text = f"{doc.title}. {doc.content}"
            text_norm = cls._normalize_for_match(text)
            if not (subject_norm in text_norm or title_norm == subject_norm or title_norm in subject_norm):
                continue
            paren = re.search(
                r"\(([^()]{2,90}?),\s*(?:\d{1,2}\s+[A-Za-z]+\s+\d{3,4}|[A-Za-z]+\s+\d{1,2},\s*\d{3,4}|\d{3,4})\s*[–-]",
                text,
            )
            if paren:
                return paren.group(1).strip(" .")
            born = re.search(
                r"\bborn\s+(?:in|at)\s+([^.;()]{2,100}?)(?:,?\s+(?:on\s+)?(?:\d{1,2}\s+[A-Za-z]+\s+\d{3,4}|[A-Za-z]+\s+\d{1,2},\s*\d{3,4}|\d{3,4})|[.;])",
                text,
                flags=re.IGNORECASE,
            )
            if born:
                return born.group(1).strip(" .,")
        return ""

    @staticmethod
    def _extract_birth_subject(subgoal: str) -> str:
        match = re.search(r"\bwhere\s+was\s+(.+?)\s+born\b", subgoal, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .?")
        match = re.search(r"\bplace\s+of\s+birth\s+of\s+(.+?)(?:\?|$)", subgoal, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .?")
        return ""

    @classmethod
    def _numeric_answer_validation_failure(cls, subgoal: str, answer: str) -> str:
        text = subgoal.lower()
        if not re.search(
            r"\b(population|how many|number of|count of|what year|which year|what date|which date|how old)\b",
            text,
        ):
            return ""
        answer_norm = cls._normalize_for_match(answer)
        if not answer_norm:
            return "empty numeric/date answer"

        number_words = {
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
            "hundred",
            "thousand",
            "million",
            "billion",
        }
        months = {
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        }
        tokens = set(answer_norm.split())
        if any(ch.isdigit() for ch in answer_norm):
            return ""
        if tokens & number_words:
            return ""
        if tokens & months:
            return ""
        return "numeric/date sub-question returned a non-numeric entity"

    @classmethod
    def _descriptor_constraint_validation_failure(
        cls,
        subgoal: str,
        answer: str,
        documents: List[Document],
    ) -> str:
        constraints = cls._descriptor_constraints(subgoal)
        if len(constraints) < 2:
            return ""
        answer_norm = cls._normalize_for_match(answer)
        if not answer_norm:
            return "empty constrained-candidate answer"

        answer_tokens = answer_norm.split()
        constraint_set = set(constraints)
        for doc in documents:
            text_norm = cls._normalize_for_match(f"{doc.title} {doc.content}")
            if not cls._contains_token_sequence(text_norm.split(), answer_tokens):
                continue
            window = cls._window_around_answer(text_norm, answer_norm, width=70)
            if constraint_set.issubset(set(window.split())):
                return ""
        return "answer candidate is not explicitly supported with all descriptor constraints"

    @staticmethod
    def _descriptor_constraints(subgoal: str) -> List[str]:
        text = OperaPipeline._normalize_for_match(subgoal)
        match = re.search(
            r"\b(?:who|which|what)\s+(.{0,120}?)(?:person|man|woman|(?:film\s+)?actor|(?:film\s+)?actress|artist|singer|writer|author|poet|player|coach|director|politician)\b",
            text,
        )
        if not match:
            return []
        descriptor_text = match.group(1)
        descriptor_terms = {
            "african",
            "american",
            "argentine",
            "australian",
            "austrian",
            "belgian",
            "brazilian",
            "british",
            "canadian",
            "chinese",
            "dutch",
            "english",
            "french",
            "german",
            "greek",
            "indian",
            "irish",
            "italian",
            "japanese",
            "mexican",
            "norwegian",
            "pakistani",
            "polish",
            "russian",
            "scottish",
            "spanish",
            "swedish",
            "swiss",
            "turkish",
        }
        tokens = descriptor_text.split()
        constraints: List[str] = []
        for token in tokens:
            if token in descriptor_terms and token not in constraints:
                constraints.append(token)
        return constraints

    @staticmethod
    def _window_around_answer(text_norm: str, answer_norm: str, *, width: int) -> str:
        tokens = text_norm.split()
        answer_tokens = answer_norm.split()
        positions = OperaPipeline._find_token_sequence_positions(tokens, answer_tokens)
        if not positions:
            return ""
        start = positions[0]
        left = max(0, start - width)
        right = min(len(tokens), start + len(answer_tokens) + width)
        return " ".join(tokens[left:right])

    @classmethod
    def _characterizer_validation_failure(cls, subgoal: str, answer: str, documents: List[Document]) -> str:
        answer_norm = cls._normalize_for_match(answer)
        if not answer_norm:
            return "empty characterizer answer"
        subject = cls._extract_characterizer_subject(subgoal)
        subject_norm = cls._normalize_for_match(subject)
        for doc in documents:
            text_norm = cls._normalize_for_match(f"{doc.title} {doc.content}")
            if answer_norm not in text_norm:
                continue
            if subject_norm and subject_norm not in text_norm:
                continue
            if re.search(r"\b(characteri[sz]e|characteri[sz]es|described?|called?|named?|referred)\b", text_norm):
                return ""
        return "characterizer answer is not bound to the requested entity in the retrieved evidence"

    @staticmethod
    def _extract_characterizer_subject(subgoal: str) -> str:
        match = re.search(
            r"\b(?:who|what)\s+(?:characteri[sz]ed?|described?|called?|named?|referred\s+to)\s+(.+?)\s+as\b",
            subgoal,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(" .?")
        match = re.search(
            r"\b(.+?)\s+(?:was|is|were|are)?\s*(?:characteri[sz]ed?|described?|called?|named?|referred\s+to)\s+as\b",
            subgoal,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(" .?")
        return ""

    @classmethod
    def _characterizer_relation_correction(cls, subgoal: str, documents: List[Document]) -> str:
        subject = cls._extract_characterizer_subject(subgoal)
        subject_norm = cls._normalize_for_match(subject)
        source_patterns = [
            r"(?:\d{4}'s\s+)?\"([^\"]{3,100})\"\s+(?:characteri[sz]es|describes|calls|names|refers\s+to)\b",
            r"(?:\d{4}'s\s+)?'([^']{3,100})'\s+(?:characteri[sz]es|describes|calls|names|refers\s+to)\b",
        ]
        for doc in documents:
            text = f"{doc.title}. {doc.content}"
            text_norm = cls._normalize_for_match(text)
            if subject_norm and subject_norm not in text_norm and cls._normalize_for_match(doc.title) not in subject_norm:
                continue
            for pattern in source_patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip(" .,'\"")
                    if 3 <= len(candidate) <= 100:
                        return candidate
        return ""

    @classmethod
    def _alias_relation_correction(
        cls,
        subgoal: str,
        answer: str,
        documents: List[Document],
        analysis: str,
    ) -> str:
        answer_norm = cls._normalize_for_match(answer)
        sources = [analysis or ""]
        sources.extend(f"{doc.title}. {doc.content}" for doc in documents)
        for source in sources:
            for candidate in cls._extract_alias_candidates(source):
                candidate_norm = cls._normalize_for_match(candidate)
                if not candidate_norm or candidate_norm == answer_norm:
                    continue
                if answer_norm and answer_norm in candidate_norm:
                    continue
                return candidate
        return ""

    @classmethod
    def _extract_alias_candidates(cls, text: str) -> List[str]:
        candidates: List[str] = []
        patterns = [
            r"\b(?:also known as|known as|called|nicknamed|referred to as)\s+(?:the\s+)?[\"“']([^\"”']{2,120})[\"”']",
            r"\b(?:also known as|known as|called|nicknamed|referred to as)\s+(?:the\s+)?([^.;,\n]{2,120})",
            r"\b(?:another|alternative)\s+name\s+(?:for\s+[^.;,\n]{2,80}\s+)?(?:is|was)\s+(?:the\s+)?[\"“']?([^\"”'.;,\n]{2,120})",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text or "", flags=re.IGNORECASE):
                candidate = match.group(1).strip(" .,'\"“”")
                if not (2 <= len(candidate) <= 120):
                    continue
                if cls._normalize_for_match(candidate) in {"also", "known", "called", "the"}:
                    continue
                if candidate not in candidates:
                    candidates.append(candidate)
        return candidates

    @classmethod
    def _member_validation_failure(cls, subgoal: str, answer: str) -> str:
        answer_norm = cls._normalize_for_match(answer)
        if not answer_norm:
            return "empty member answer"
        group = cls._extract_relation_target(subgoal, [r"member of", r"part of", r"belongs to"])
        group_norm = cls._normalize_for_match(group)
        if group_norm and answer_norm == group_norm:
            return "member question returned the group itself"
        return ""

    @classmethod
    def _member_relation_correction(cls, subgoal: str, documents: List[Document]) -> str:
        group = cls._extract_relation_target(subgoal, [r"member of", r"part of", r"belongs to"])
        group_norm = cls._normalize_for_match(group)
        if not group_norm:
            return ""
        group_pattern = r"\s+".join(re.escape(token) for token in group_norm.split())
        member_patterns = [
            rf"\b(?:founding\s+)?members?\s+of\s+(?:the\s+)?{group_pattern}\b",
            rf"\b(?:formerly|previously)\s+in\s+(?:the\s+)?{group_pattern}\b",
            rf"\bpart\s+of\s+(?:the\s+)?{group_pattern}\b",
            rf"\bbelongs?\s+to\s+(?:the\s+)?{group_pattern}\b",
        ]
        for doc in documents:
            title = (doc.title or "").strip()
            title_norm = cls._normalize_for_match(title)
            if not title_norm or title_norm == group_norm:
                continue
            text_norm = cls._normalize_for_match(f"{doc.title} {doc.content}")
            if not any(re.search(pattern, text_norm) for pattern in member_patterns):
                continue
            if len(title_norm.split()) <= 6:
                return title
        return ""

    @classmethod
    def _counterpart_relation_validation_failure(cls, subgoal: str, answer: str) -> str:
        answer_norm = cls._normalize_for_match(answer)
        if not answer_norm:
            return "empty counterpart answer"
        anchors = []
        for pattern in [
            r"\bwith\s+([^,?]+)",
            r"\bbetween\s+([^,?]+)",
            r"\bshares?\s+a\s+border\s+with\s+([^,?]+)",
            r"\bconnected\s+to\s+([^,?]+)",
        ]:
            anchors.extend(match.group(1) for match in re.finditer(pattern, subgoal, flags=re.IGNORECASE))
        for anchor in anchors:
            anchor_norm = cls._normalize_for_match(anchor)
            if anchor_norm and answer_norm == anchor_norm:
                return "counterpart relation returned the anchor entity itself"
        return ""

    @classmethod
    def _counterpart_relation_correction(
        cls,
        subgoal: str,
        answer: str,
        documents: List[Document],
        analysis: str,
    ) -> str:
        answer_norm = cls._normalize_for_match(answer)
        if not answer_norm:
            return ""
        anchors = []
        for pattern in [
            r"\bwith\s+([^,?]+)",
            r"\bbetween\s+([^,?]+)",
            r"\bshares?\s+a\s+border\s+with\s+([^,?]+)",
            r"\bconnected\s+to\s+([^,?]+)",
        ]:
            anchors.extend(match.group(1).strip(" .?") for match in re.finditer(pattern, subgoal, flags=re.IGNORECASE))

        sources = [analysis or ""]
        sources.extend(f"{doc.title}. {doc.content}" for doc in documents)
        for anchor in anchors:
            anchor_norm = cls._normalize_for_match(anchor)
            if not anchor_norm or answer_norm != anchor_norm:
                continue
            for source in sources:
                candidate = cls._extract_counterpart_candidate(source, anchor)
                if candidate and cls._normalize_for_match(candidate) != anchor_norm:
                    return candidate
        return ""

    @classmethod
    def _extract_counterpart_candidate(cls, text: str, anchor: str) -> str:
        anchor_re = re.escape(anchor.strip())
        patterns = [
            rf"\bbetween\s+{anchor_re}\s+and\s+([A-Z][A-Za-z .'-]{{2,60}})",
            rf"\bbetween\s+([A-Z][A-Za-z .'-]{{2,60}})\s+and\s+{anchor_re}\b",
            rf"\bof\s+{anchor_re}\s+and\s+([A-Z][A-Za-z .'-]{{2,60}})",
            rf"\bof\s+([A-Z][A-Za-z .'-]{{2,60}})\s+and\s+{anchor_re}\b",
            rf"{anchor_re}\s*[–-]\s*([A-Z][A-Za-z .'-]{{2,60}})",
            rf"([A-Z][A-Za-z .'-]{{2,60}})\s*[–-]\s*{anchor_re}",
        ]
        stop_words = {
            "commission",
            "friendship",
            "government",
            "governments",
            "document",
            "president",
        }
        for pattern in patterns:
            match = re.search(pattern, text or "", flags=re.IGNORECASE)
            if not match:
                continue
            candidate = re.split(
                r"\b(?:Commission|Friendship|Document|government|governments|was|is|,|\.|\(|\))\b",
                match.group(1).strip(" .,'\""),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" .,'\"")
            norm = cls._normalize_for_match(candidate)
            if not norm or norm in stop_words:
                continue
            if len(norm.split()) <= 5:
                return candidate
        return ""

    @classmethod
    def _featured_entity_validation_failure(cls, subgoal: str, answer: str, documents: List[Document]) -> str:
        answer_norm = cls._normalize_for_match(answer)
        if not answer_norm:
            return "empty featured-entity answer"
        for doc in documents:
            text = cls._normalize_for_match(f"{doc.title} {doc.content}")
            if answer_norm not in text:
                continue
            if re.search(rf"\b(character|figure|topic|entity)\s+of\s+\w+.+\bby\s+{re.escape(answer_norm)}\b", text):
                return "featured-entity question returned the author/creator instead of the featured entity"
        return ""

    @classmethod
    def _answer_has_actor_role_support(cls, subgoal: str, answer: str, documents: List[Document]) -> bool:
        answer_norm = cls._normalize_for_match(answer)
        if not answer_norm:
            return False
        role = cls._extract_actor_role_target(subgoal)
        role_norm = cls._normalize_for_match(role)
        if not role_norm or len(role_norm.split()) < 1:
            return True
        role_tokens = [
            token
            for token in role_norm.split()
            if token not in {"the", "a", "an", "role", "character", "figure"}
        ]
        if not role_tokens:
            return True
        for doc in documents:
            text = cls._normalize_for_match(f"{doc.title} {doc.content}")
            if answer_norm not in text:
                continue
            if all(token in text for token in role_tokens):
                return True
        return False

    @staticmethod
    def _extract_actor_role_target(subgoal: str) -> str:
        match = re.search(
            r"\b(?:play|plays|played|portray|portrays|portrayed)\s+(.+?)\s+\b(?:in|on)\b",
            subgoal,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(" .?")
        return ""

    @classmethod
    def _answer_has_writer_support(cls, subgoal: str, answer: str, documents: List[Document]) -> bool:
        answer_norm = cls._normalize_for_match(answer)
        if not answer_norm:
            return False
        quoted_titles = [
            cls._normalize_for_match(item)
            for item in re.findall(r'"([^"]+)"|\'([^\']+)\'', subgoal)
            for item in item
            if item
        ]
        writer_cues = (
            "writer",
            "written by",
            "wrote",
            "author",
            "authored by",
            "composer",
            "composed by",
            "songwriter",
            "songwriter s",
            "lyrics by",
        )
        for doc in documents:
            raw_text = f"{doc.title} {doc.content}"
            norm_text = cls._normalize_for_match(raw_text)
            if quoted_titles and not any(title in norm_text for title in quoted_titles):
                continue
            for start in cls._find_token_sequence_positions(norm_text.split(), answer_norm.split()):
                prefix_tokens = norm_text.split()[max(0, start - 18) : start]
                prefix = " ".join(prefix_tokens)
                if any(cue in prefix for cue in writer_cues):
                    return True
        return False

    @classmethod
    def _answer_has_exact_relation_target_binding(
        cls,
        subgoal: str,
        answer: str,
        documents: List[Document],
    ) -> bool:
        target = cls._extract_relation_target(subgoal, [r"spouse of", r"wife of", r"husband of"])
        target_tokens = [
            token
            for token in cls._normalize_for_match(target).split()
            if len(token) > 1 and token not in {"the", "a", "an", "of"}
        ]
        if len(target_tokens) < 2:
            return True

        answer_norm = cls._normalize_for_match(answer)
        if not answer_norm:
            return False
        for doc in documents:
            text_norm = cls._normalize_for_match(f"{doc.title} {doc.content}")
            if answer_norm not in text_norm:
                continue
            if all(token in text_norm for token in target_tokens):
                return True
        return False

    @classmethod
    def _answer_has_parent_relation_support(
        cls,
        subgoal: str,
        answer: str,
        documents: List[Document],
    ) -> bool:
        target = cls._extract_relation_target(subgoal, [r"father of", r"mother of", r"parent of"])
        target_norm = cls._normalize_for_match(target)
        answer_norm = cls._normalize_for_match(answer)
        if not target_norm or not answer_norm:
            return False
        if target_norm == answer_norm:
            return False

        relation = cls._parent_relation_type(subgoal)
        for doc in documents:
            for sentence in cls._relation_sentences(doc):
                sentence_norm = cls._normalize_for_match(sentence)
                if target_norm not in sentence_norm or answer_norm not in sentence_norm:
                    continue
                if cls._sentence_supports_parent_relation(sentence_norm, target_norm, answer_norm, relation):
                    return True
        return False

    @classmethod
    def _parent_relation_correction(cls, subgoal: str, documents: List[Document]) -> str:
        target = cls._extract_relation_target(subgoal, [r"father of", r"mother of", r"parent of"])
        target_norm = cls._normalize_for_match(target)
        if not target_norm:
            return ""
        relation = cls._parent_relation_type(subgoal)
        for doc in documents:
            for sentence in cls._relation_sentences(doc):
                sentence_norm = cls._normalize_for_match(sentence)
                if target_norm not in sentence_norm:
                    continue
                candidate = cls._extract_parent_candidate_from_sentence(sentence, target, relation)
                if candidate:
                    return candidate
        return ""

    @staticmethod
    def _parent_relation_type(subgoal: str) -> str:
        text = (subgoal or "").lower()
        if "mother of" in text:
            return "mother"
        if "father of" in text:
            return "father"
        return "parent"

    @staticmethod
    def _relation_sentences(doc: Document) -> List[str]:
        text = re.sub(r"\s+", " ", f"{doc.title}. {doc.content}" if doc else "").strip()
        return [part.strip() for part in re.split(r"(?<=[.;])\s+", text) if part.strip()]

    @classmethod
    def _sentence_supports_parent_relation(
        cls,
        sentence_norm: str,
        target_norm: str,
        answer_norm: str,
        relation: str,
    ) -> bool:
        answer_re = r"\s+".join(re.escape(token) for token in answer_norm.split())
        target_re = r"\s+".join(re.escape(token) for token in target_norm.split())
        parent_terms = "father|mother|parent|parents"
        child_terms = "son|daughter|child|children"
        if relation == "father":
            parent_terms = "father|parents"
        elif relation == "mother":
            parent_terms = "mother|parents"
        patterns = [
            rf"\b{target_re}\b.{{0,120}}\b(?:{child_terms})\s+of\s+(?:the\s+)?\b{answer_re}\b",
            rf"\b{target_re}\b.{{0,120}}\b(?:{child_terms})\s+of\s+.{{0,80}}\b{answer_re}\b",
            rf"\b{target_re}\b.{{0,120}}\bborn\s+to\s+(?:the\s+)?\b{answer_re}\b",
            rf"\b{target_re}\b.{{0,120}}\bborn\s+to\s+.{{0,80}}\b{answer_re}\b",
            rf"\b{answer_re}\b.{{0,80}}\b(?:{parent_terms})\s+of\s+(?:the\s+)?\b{target_re}\b",
            rf"\b{answer_re}\b.{{0,80}}\b(?:had|has)\s+(?:a\s+)?(?:son|daughter|child).{{0,80}}\b{target_re}\b",
        ]
        return any(re.search(pattern, sentence_norm) for pattern in patterns)

    @classmethod
    def _extract_parent_candidate_from_sentence(cls, sentence: str, target: str, relation: str) -> str:
        target_re = re.escape(target.strip())
        parent_word = "father" if relation == "father" else "mother" if relation == "mother" else r"father|mother|parent"
        patterns = [
            rf"{target_re}[^.;]{{0,140}}\b(?:son|daughter|child)\s+of\s+([A-Z][A-Za-zÀ-ÖØ-öø-ÿ0-9 ,.'-]{{2,120}})",
            rf"([A-Z][A-Za-zÀ-ÖØ-öø-ÿ0-9 ,.'-]{{2,120}})\s+(?:was|is)?\s*(?:the\s+)?(?:{parent_word})\s+of\s+{target_re}",
        ]
        for pattern in patterns:
            match = re.search(pattern, sentence or "", flags=re.IGNORECASE)
            if not match:
                continue
            candidate = cls._clean_parent_candidate(match.group(1), relation)
            if candidate and cls._normalize_for_match(candidate) != cls._normalize_for_match(target):
                return candidate
        return ""

    @staticmethod
    def _clean_parent_candidate(candidate: str, relation: str) -> str:
        text = candidate.strip(" .,'\"")
        if relation in {"father", "mother"} and " and " in text:
            return ""
        text = re.split(
            r"\b(?:who|which|while|where|when|and their|and his|and her|with whom|;|\.|\))\b",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .,'\"")
        text = re.sub(r"\s+", " ", text)
        return text if 2 <= len(text) <= 120 else ""

    @staticmethod
    def _extract_relation_target(subgoal: str, relation_phrases: List[str]) -> str:
        for phrase in relation_phrases:
            match = re.search(
                rf"\b{phrase}\s+(.+?)(?:\?|,|\bthat\b|\bwho\b|\bwhich\b|\bwhere\b|$)",
                subgoal,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip(" .")
        return ""

    @staticmethod
    def _find_token_sequence_positions(tokens: List[str], needle: List[str]) -> List[int]:
        if not needle or len(needle) > len(tokens):
            return []
        width = len(needle)
        return [
            idx
            for idx in range(len(tokens) - width + 1)
            if tokens[idx : idx + width] == needle
        ]

    def _rewrite(
        self,
        *,
        original_question: str,
        subgoal: str,
        failure_info: str,
        documents: List[Document],
        previous_answers: Dict[int, str],
    ) -> RewriteResult:
        raw = self.rewrite_llm.generate(
            [
                {"role": "system", "content": REWRITE_SYSTEM},
                {
                    "role": "user",
                    "content": build_rewrite_prompt(
                        original_question,
                        subgoal,
                        failure_info,
                        documents,
                        previous_answers,
                        max_docs=min(3, self.config.max_docs_in_prompt),
                        max_chars=min(400, self.config.max_doc_chars),
                    ),
                },
            ],
            max_tokens=self.config.rewrite_max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        try:
            parsed = extract_json(raw)
            rewritten_query = str(parsed.get("rewritten_query") or "").strip()
            rewritten_query = self._augment_rewrite_with_candidate_entities(
                rewritten_query or subgoal,
                failure_info,
                documents,
            )
            keywords = parsed.get("keywords") or []
            if not isinstance(keywords, list):
                keywords = [str(keywords)]
            return RewriteResult(
                rewritten_query=rewritten_query,
                strategy=str(parsed.get("strategy") or ""),
                keywords=[str(k) for k in keywords],
                raw_response=raw,
            )
        except Exception:
            fallback = self._fallback_rewrite(raw, subgoal)
            fallback = self._augment_rewrite_with_candidate_entities(fallback, failure_info, documents)
            return RewriteResult(rewritten_query=fallback, strategy="fallback_parse", raw_response=raw)

    def _final_synthesis(
        self,
        question: str,
        answers: Dict[int, str],
        step_details: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        raw = self.analysis_llm.generate(
            [
                {"role": "system", "content": FINAL_SYSTEM},
                {"role": "user", "content": build_final_synthesis_prompt(question, answers, step_details)},
            ],
            max_tokens=self.config.final_max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        try:
            parsed = extract_json(raw)
            return {
                "answer": str(parsed.get("answer") or ""),
                "analysis": str(parsed.get("analysis") or ""),
                "raw_response": raw,
            }
        except Exception:
            recovered_answer = self._recover_string_field(raw, "answer")
            recovered_analysis = self._recover_string_field(raw, "analysis") or ""
            if recovered_answer:
                return {
                    "answer": recovered_answer,
                    "analysis": recovered_analysis,
                    "raw_response": raw,
                }
            return {"answer": raw.strip(), "analysis": "", "raw_response": raw}

    def _direct_answer_fallback(self, question: str) -> Dict[str, Any]:
        top_k = max(self.config.top_k_schedule or [self.config.top_k])
        docs, retrieval_queries = self._retrieve_with_expansions(
            [question],
            top_k=top_k,
            max_docs=max(self.config.max_docs_in_prompt, top_k),
        )
        docs = self._rerank_documents_for_subgoal(docs, question)
        analysis = self._analyze(question, question, docs, {})
        return {
            "retrieval_queries": retrieval_queries,
            "top_k": top_k,
            "documents": [doc.to_dict(max_chars=self.config.max_doc_chars) for doc in docs],
            "analysis": analysis.to_dict(),
        }

    @classmethod
    def _postprocess_final_answer(
        cls,
        question: str,
        final_answer: str,
        executed_steps: List[Dict[str, Any]],
        *,
        pipeline_success: bool = False,
    ) -> tuple[str, Dict[str, Any]]:
        if pipeline_success and "not found" in str(final_answer or "").lower():
            fallback_answer = cls._last_concrete_executed_answer(executed_steps)
            if fallback_answer:
                return fallback_answer, {
                    "type": "successful_chain_notfound_fallback",
                    "old_answer": final_answer,
                    "new_answer": fallback_answer,
                }

        same_country_answer = cls._postprocess_same_country(question, executed_steps)
        if same_country_answer and cls._normalize_for_match(same_country_answer) != cls._normalize_for_match(final_answer):
            return same_country_answer, {
                "type": "same_country",
                "old_answer": final_answer,
                "new_answer": same_country_answer,
            }

        comparison_answer = cls._postprocess_candidate_comparison(question, final_answer, executed_steps)
        if comparison_answer and cls._normalize_for_match(comparison_answer) != cls._normalize_for_match(final_answer):
            return comparison_answer, {
                "type": "candidate_comparison",
                "old_answer": final_answer,
                "new_answer": comparison_answer,
            }
        shortened = cls._strip_leading_source_year(final_answer)
        if shortened != final_answer:
            return shortened, {
                "type": "short_source_title",
                "old_answer": final_answer,
                "new_answer": shortened,
            }
        return final_answer, {}

    @staticmethod
    def _last_concrete_executed_answer(executed_steps: List[Dict[str, Any]]) -> str:
        for item in reversed(executed_steps or []):
            answer = str(item.get("answer") or "").strip()
            if answer and "not found" not in answer.lower():
                return answer
        return ""

    @classmethod
    def _postprocess_same_country(cls, question: str, executed_steps: List[Dict[str, Any]]) -> str:
        if not re.search(r"\bsame country\b", question, flags=re.IGNORECASE):
            return ""
        countries = []
        for item in executed_steps:
            subgoal = str(item.get("filled_subgoal") or item.get("subgoal") or "")
            if not re.search(r"\bcountry\b", subgoal, flags=re.IGNORECASE):
                continue
            country = cls._extract_country_component(str(item.get("answer") or ""))
            if country:
                countries.append(country)
        if len(countries) < 2:
            return ""
        normalized = [cls._normalize_for_match(country) for country in countries]
        return "yes" if len(set(normalized)) == 1 else "no"

    @classmethod
    def _extract_country_component(cls, answer: str) -> str:
        text = str(answer or "").strip(" .")
        if not text:
            return ""
        parts = [part.strip(" .") for part in text.split(",") if part.strip(" .")]
        return parts[-1] if parts else text

    @staticmethod
    def _strip_leading_source_year(answer: str) -> str:
        text = str(answer or "").strip()
        match = re.match(r"^\d{3,4}'s\s+\"([^\"]+)\"$", text)
        if match:
            return match.group(1).strip()
        match = re.match(r"^\d{3,4}'s\s+'([^']+)'$", text)
        if match:
            return match.group(1).strip()
        return text

    @classmethod
    def _postprocess_candidate_comparison(
        cls,
        question: str,
        final_answer: str,
        executed_steps: List[Dict[str, Any]],
    ) -> str:
        candidates = cls._extract_or_candidates(question)
        if len(candidates) != 2:
            return ""
        question_norm = cls._normalize_for_match(question)
        if not re.search(r"\b(which|what|who)\b", question_norm):
            return ""

        candidate_dates: Dict[str, tuple[int, int, int]] = {}
        candidate_related_answers: Dict[str, str] = {}
        for candidate in candidates:
            candidate_norm = cls._normalize_for_match(candidate)
            for item in executed_steps:
                subgoal = str(item.get("filled_subgoal") or item.get("subgoal") or "")
                subgoal_norm = cls._normalize_for_match(subgoal)
                answer = str(item.get("answer") or "")
                if candidate_norm not in subgoal_norm:
                    continue
                if cls._parse_date(answer):
                    candidate_dates[candidate] = cls._parse_date(answer)  # type: ignore[assignment]
                elif re.search(
                    r"\b(director|directed|author|authored|creator|created|composer|composed|writer|wrote|written)\b",
                    subgoal_norm,
                ):
                    candidate_related_answers[candidate] = answer

        final_norm = cls._normalize_for_match(final_answer)
        for candidate, related_answer in candidate_related_answers.items():
            if final_norm and final_norm == cls._normalize_for_match(related_answer):
                return candidate

        if len(candidate_dates) != 2:
            return ""

        if re.search(r"\b(younger|born later|later birth)\b", question_norm):
            return max(candidate_dates.items(), key=lambda item: item[1])[0]
        if re.search(r"\b(older|born first|earlier birth)\b", question_norm):
            return min(candidate_dates.items(), key=lambda item: item[1])[0]
        if re.search(r"\b(died later|death later|director died later)\b", question_norm):
            return max(candidate_dates.items(), key=lambda item: item[1])[0]
        if re.search(r"\b(died first|death first)\b", question_norm):
            return min(candidate_dates.items(), key=lambda item: item[1])[0]
        if re.search(r"\b(came out first|released first|earlier release|came first)\b", question_norm):
            return min(candidate_dates.items(), key=lambda item: item[1])[0]
        if re.search(r"\b(came out later|released later|more recently|recently)\b", question_norm):
            return max(candidate_dates.items(), key=lambda item: item[1])[0]
        return ""

    @staticmethod
    def _extract_or_candidates(question: str) -> List[str]:
        text = re.sub(r"\s+", " ", question or "").strip(" ?")
        match = re.search(r",\s*(.+?)\s+or\s+(.+?)\s*$", text, flags=re.IGNORECASE)
        if not match:
            return []
        return [match.group(1).strip(" .?\"'"), match.group(2).strip(" .?\"'")]

    @staticmethod
    def _parse_date(text: str) -> Optional[tuple[int, int, int]]:
        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        raw = str(text or "")
        match = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{3,4})\b", raw)
        if match and match.group(2).lower() in months:
            return (int(match.group(3)), months[match.group(2).lower()], int(match.group(1)))
        match = re.search(r"\b([A-Za-z]+)\s+(\d{1,2}),\s*(\d{3,4})\b", raw)
        if match and match.group(1).lower() in months:
            return (int(match.group(3)), months[match.group(1).lower()], int(match.group(2)))
        match = re.search(r"\b(\d{3,4})\b", raw)
        if match:
            return (int(match.group(1)), 1, 1)
        return None

    @staticmethod
    def _parse_plan(raw: str, question: str) -> List[PlanStep]:
        try:
            parsed = extract_json(raw)
        except Exception:
            parsed = None

        items: List[Dict[str, Any]] = []
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            if isinstance(parsed.get("sub_questions"), list):
                items = parsed["sub_questions"]
            elif isinstance(parsed.get("steps"), list):
                items = parsed["steps"]

        steps: List[PlanStep] = []
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            step_id = int(item.get("subgoal_id") or item.get("step_id") or idx)
            subgoal = str(item.get("subgoal") or item.get("sub_question") or item.get("question") or "").strip()
            deps_raw = item.get("dependencies") or []
            deps = []
            if isinstance(deps_raw, list):
                for dep in deps_raw:
                    try:
                        dep_int = int(dep)
                        if dep_int < step_id:
                            deps.append(dep_int)
                    except Exception:
                        continue
            if subgoal:
                steps.append(PlanStep(step_id=step_id, subgoal=subgoal, dependencies=deps))

        if steps:
            return sorted(steps, key=lambda x: x.step_id)

        xml_subquestions = re.findall(
            r"<subquestion>\s*(.*?)\s*</subquestion>",
            raw or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        if xml_subquestions:
            xml_steps: List[PlanStep] = []
            for idx, subquestion in enumerate(xml_subquestions, start=1):
                subgoal = re.sub(r"\s+", " ", subquestion).strip()
                if not subgoal:
                    continue
                deps = []
                for dep in re.findall(r"\[Answer\s*(\d+)\]", subgoal, flags=re.IGNORECASE):
                    dep_int = int(dep)
                    if dep_int < idx:
                        deps.append(dep_int)
                xml_steps.append(PlanStep(step_id=idx, subgoal=subgoal, dependencies=sorted(set(deps))))
            if xml_steps:
                return xml_steps

        fallback_steps = []
        for idx, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            match = re.match(r"^(?:step\s*)?(\d+)[.)\s:-]+(.+)$", line, flags=re.IGNORECASE)
            if match:
                fallback_steps.append(
                    PlanStep(step_id=int(match.group(1)), subgoal=match.group(2).strip(), dependencies=[])
                )
        if fallback_steps:
            return fallback_steps
        return [PlanStep(step_id=1, subgoal=question, dependencies=[])]

    @staticmethod
    def _fill_placeholders(text: str, answers: Dict[int, str]) -> str:
        filled = text
        for step_id, answer in sorted(answers.items()):
            if not answer:
                continue
            patterns = [
                rf"\[ANSWER_{step_id}\]",
                rf"\[Answer\s*{step_id}\]",
                rf"\[answer\s*{step_id}\]",
                rf"\[[^\]]+\s+from\s+step\s+{step_id}\]",
            ]
            for pattern in patterns:
                filled = re.sub(pattern, answer, filled, flags=re.IGNORECASE)
            duplicate = rf"{re.escape(answer)}\s*,\s*{re.escape(answer)}"
            filled = re.sub(duplicate, answer, filled, flags=re.IGNORECASE)
        return filled

    @staticmethod
    def _ensure_dependency_context(
        filled_subgoal: str,
        original_subgoal: str,
        dependencies: List[int],
        answers: Dict[int, str],
    ) -> str:
        if filled_subgoal != original_subgoal:
            return filled_subgoal
        dep_facts = []
        for dep in dependencies:
            answer = answers.get(dep, "").strip()
            if answer:
                patterns = [
                    rf"\b(?:the\s+)?(?:entity|answer|item|person|place|organization)\s+(?:identified|found|obtained)\s+in\s+step\s+{dep}\b",
                    rf"\b(?:the\s+)?(?:entity|answer|item|person|place|organization)\s+from\s+step\s+{dep}\b",
                    rf"\bstep\s+{dep}\s+answer\b",
                ]
                for pattern in patterns:
                    filled_subgoal = re.sub(pattern, answer, filled_subgoal, flags=re.IGNORECASE)
                dep_facts.append(f"Step {dep} answer: {answer}")
        if filled_subgoal != original_subgoal:
            return filled_subgoal
        if not dep_facts:
            return filled_subgoal
        return f"{filled_subgoal}\nDependency context: {'; '.join(dep_facts)}"

    @staticmethod
    def _split_candidate_answer(answer: str) -> List[str]:
        cleaned = answer.strip()
        if not cleaned:
            return []
        parts = re.split(r"\s+(?:and|or)\s+|[,;/]+", cleaned)
        candidates = [part.strip(" .") for part in parts if part.strip(" .")]
        if len(candidates) <= 1:
            return [cleaned]
        return candidates

    @staticmethod
    def _fallback_rewrite(raw: str, original: str) -> str:
        quoted = re.findall(r'"([^"]{5,200})"', raw)
        if quoted:
            return quoted[0].strip()
        for line in raw.splitlines():
            cleaned = line.strip(" -:\t")
            if len(cleaned) > 5:
                return cleaned
        return original

    @classmethod
    def _recover_analysis_result(cls, raw: str) -> Optional[AnalysisResult]:
        status = cls._recover_string_field(raw, "status").lower()
        answer = cls._recover_string_field(raw, "answer")
        if status not in {"yes", "no"}:
            if not answer:
                return None
            status = "yes"
        return AnalysisResult(
            status=status,
            answer=answer if status == "yes" else "",
            analysis=cls._recover_string_field(raw, "analysis") or raw,
            raw_response=raw,
        )

    @staticmethod
    def _recover_string_field(raw: str, field: str) -> str:
        match = re.search(rf'"{re.escape(field)}"\s*:', raw or "", flags=re.IGNORECASE)
        if not match:
            return ""
        idx = match.end()
        while idx < len(raw) and raw[idx].isspace():
            idx += 1
        if idx >= len(raw):
            return ""

        if raw[idx] == '"':
            try:
                value, _ = json.JSONDecoder().raw_decode(raw[idx:])
                return str(value).strip()
            except Exception:
                pass

            # Last-resort field recovery for malformed JSON. This keeps escaped
            # quotes intact and stops at the next likely JSON field boundary.
            idx += 1
            chars: List[str] = []
            escaped = False
            while idx < len(raw):
                ch = raw[idx]
                if escaped:
                    chars.append(ch)
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"' and re.match(r'\s*(?:,|\})', raw[idx + 1 :]):
                    break
                else:
                    chars.append(ch)
                idx += 1
            return "".join(chars).strip()

        match_value = re.match(r"([^,\r\n}]+)", raw[idx:])
        return match_value.group(1).strip() if match_value else ""

    @staticmethod
    def _is_not_found_answer(answer: str) -> bool:
        normalized = answer.lower().strip()
        return normalized in {
            "",
            "not found",
            "not found in retrieved documents",
            "not found in the retrieved documents",
            "not found in documents",
            "unknown",
        }
