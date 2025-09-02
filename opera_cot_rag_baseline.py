#!/usr/bin/env python3
"""
OPERA CoT+RAG Baseline Implementation
Chain-of-Thought + RAG baseline version based on paper and implementation document

Core features:
1. Single 7B model sequentially playing three Agent roles
2. TMC component with trajectory memory
3. Strictly following prompt templates from the paper
4. Placeholder mechanism ensuring logical consistency
"""

import json
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import time
from dataclasses import dataclass, field
import faiss
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SubQuestion:
    """Sub-question structure - fully consistent with the paper"""
    step_id: int
    sub_question: str
    goal: str
    dependencies: List[int] = field(default_factory=list)
    expected_info_type: str = 'general'
    result: Optional[str] = None


@dataclass
class ExecutionTrace:
    """Execution trace record - core data structure of TMC component"""
    timestamp: str
    agent_role: str  # PLAN / Analysis-Answer / Rewrite
    input_data: Dict
    output_data: Dict
    reasoning_process: str
    retrieval_info: Optional[Dict] = None
    success: bool = True


@dataclass 
class TrajectoryMemory:
    """Trajectory Memory Component - complete recording of reasoning process"""
    question: str
    strategic_plan: Dict
    execution_traces: List[ExecutionTrace] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_time: float = 0.0
    retrieval_rounds: int = 0
    
    def add_trace(self, trace: ExecutionTrace):
        """Add execution trace"""
        self.execution_traces.append(trace)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for storage"""
        return {
            'question': self.question,
            'strategic_plan': self.strategic_plan,
            'execution_traces': [
                {
                    'timestamp': t.timestamp,
                    'agent_role': t.agent_role,
                    'input_data': t.input_data,
                    'output_data': t.output_data,
                    'reasoning_process': t.reasoning_process,
                    'retrieval_info': t.retrieval_info,
                    'success': t.success
                } for t in self.execution_traces
            ],
            'final_answer': self.final_answer,
            'total_time': self.total_time,
            'retrieval_rounds': self.retrieval_rounds
        }


class OPERACoTRAGBaseline:
    """OPERA CoT+RAG Baseline - single model playing multiple roles"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = None):
        logger.info("Initializing OPERA CoT+RAG Baseline system...")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load single 7B model
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            pad_token='<|endoftext|>',
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        self.model.eval()
        logger.info("Model loaded successfully!")
        
        # Initialize retrieval system
        self._init_retrieval_system()
        
        # Trajectory memory
        self.trajectory_memory: Optional[TrajectoryMemory] = None
        
    def _init_retrieval_system(self):
        """Initialize retrieval system"""
        # Check index files
        index_paths = [
            Path("data/indexes/hotpotqa_bge_m3_flat.index"),
            Path("./data/indexes/hotpotqa_bge_m3_flat.index"),
        ]
        
        index_path = None
        meta_path = None
        
        for path in index_paths:
            if path.exists():
                index_path = path
                meta_path = path.with_suffix('.meta')
                break
        
        if index_path is None:
            logger.warning("FAISS index not found, will use simulated retrieval")
            self.retriever = None
            self.index = None
            return
        
        logger.info(f"Loading FAISS index: {index_path}")
        self.index = faiss.read_index(str(index_path))
        
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
            self.documents = metadata.get('documents', metadata.get('metadata', []))
        
        # Load embedding model
        self.embed_model = SentenceTransformer('BAAI/bge-m3', device=self.device)
        self.retriever = "faiss"  # Mark that we have a retriever
        logger.info(f"Retrieval system ready: {self.index.ntotal:,} vectors")
        
    def retrieve_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents"""
        if self.retriever is not None and hasattr(self, 'index'):
            # Use FAISS retrieval
            query_embedding = self.embed_model.encode([query], convert_to_numpy=True)
            scores, indices = self.index.search(query_embedding, k)
            
            documents = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    if isinstance(doc, dict):
                        documents.append({
                            'title': doc.get('title', ''),
                            'content': doc.get('text', doc.get('content', '')),
                            'score': float(score)
                        })
            return documents
        else:
            # Simulated retrieval
            return [
                {'title': 'Document 1', 'content': 'Sample content for testing.', 'score': 0.9},
                {'title': 'Document 2', 'content': 'Another sample document.', 'score': 0.8}
            ]
    
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate text - core inference function"""
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def role_plan_agent(self, question: str) -> Dict:
        """Play PLAN Agent role - question decomposition"""
        logger.info("🎯 PLAN Agent: Starting question decomposition")
        
        # Construct PLAN Agent prompt - strictly following the paper
        prompt = f"""You are a strategic planning agent. Given a complex multi-hop question, decompose it into a sequence of simpler sub-goals with dependency modeling.

Question: {question}

Please generate a plan with the following JSON format:
[
  {{
    "subgoal_id": 1,
    "subgoal": "First sub-question to answer",
    "dependencies": []
  }},
  {{
    "subgoal_id": 2, 
    "subgoal": "Second sub-question using [entity from step 1]",
    "dependencies": [1]
  }}
]

Requirements:
- Use placeholder mechanism: [entity from step X] for dependencies
- Each subgoal should be answerable with a small set of documents
- Maintain logical flow and clear dependencies

IMPORTANT: For dependencies, you MUST use placeholders like [entity from step 1], [location from step 2], etc. 
Example: If step 1 finds "Alexander Graham Bell", step 2 should be "Where was [entity from step 1] born?" not "Where was Alexander Graham Bell born?"

Return ONLY the JSON array, no other text."""
        
        # Generate response
        response = self.generate(prompt, max_new_tokens=512, temperature=0.1)
        
        # Record trajectory
        trace = ExecutionTrace(
            timestamp=datetime.now().isoformat(),
            agent_role="PLAN",
            input_data={"question": question},
            output_data={"raw_response": response},
            reasoning_process=f"Decomposing complex question into sub-questions"
        )
        
        # Parse response
        try:
            # Clean response and extract JSON array
            response = response.strip()
            # Remove any markdown code blocks
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            
            # Try to extract JSON array
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
            if json_match:
                subgoals = json.loads(json_match.group())
                # Convert to internal format
                plan_data = {
                    "reasoning": "Strategic decomposition completed",
                    "sub_questions": []
                }
                for subgoal in subgoals:
                    plan_data["sub_questions"].append({
                        "step_id": subgoal.get("subgoal_id", 1),
                        "sub_question": subgoal.get("subgoal", ""),
                        "goal": f"Collect information for: {subgoal.get('subgoal', '')}",
                        "dependencies": subgoal.get("dependencies", []),
                        "expected_info_type": "general"
                    })
                trace.output_data["parsed_plan"] = plan_data
                trace.success = True
            else:
                # Fallback to default plan
                plan_data = {
                    "reasoning": "Failed to parse, using default single-step plan",
                    "sub_questions": [{
                        "step_id": 1,
                        "sub_question": question,
                        "goal": "Answer the question",
                        "dependencies": [],
                        "expected_info_type": "general"
                    }]
                }
                trace.success = False
        except Exception as e:
            logger.error(f"Failed to parse PLAN: {e}")
            plan_data = {
                "reasoning": "Parse error, using default plan",
                "sub_questions": [{
                    "step_id": 1,
                    "sub_question": question,
                    "goal": "Answer the question",
                    "dependencies": [],
                    "expected_info_type": "general"
                }]
            }
            trace.success = False
        
        self.trajectory_memory.add_trace(trace)
        logger.info(f"Generated plan: {len(plan_data['sub_questions'])} sub-questions")
        
        return plan_data
    
    def role_analysis_answer_agent(self, sub_question: str, documents: List[Dict]) -> Dict:
        """Play Analysis-Answer Agent role - information analysis and answer extraction"""
        logger.info("🔍 Analysis-Answer Agent: Starting information analysis")
        
        # Format documents
        docs_text = self._format_documents(documents)
        
        # Construct Analysis-Answer Agent prompt - strictly following the paper
        prompt = f"""You are an analysis and answering agent. Given a sub-question and retrieved documents, determine if you can answer the question and provide analysis.

Sub-question: {sub_question}

Retrieved Documents: {docs_text}

Please respond in the following JSON format:
{{
  "status": "yes" or "no",
  "answer": "extracted answer if status is yes, empty if no",
  "analysis": "explain why you can/cannot answer based on the provided documents"
}}

Key principles:
- status="yes": Documents contain sufficient information
- status="no": Documents lack necessary information
- analysis: Always explain your reasoning"""
        
        # Generate response
        response = self.generate(prompt, max_new_tokens=512, temperature=0.1)
        
        # Record trajectory
        trace = ExecutionTrace(
            timestamp=datetime.now().isoformat(),
            agent_role="Analysis-Answer",
            input_data={
                "sub_question": sub_question,
                "num_documents": len(documents)
            },
            output_data={"raw_response": response},
            reasoning_process=f"Analyzing documents for: {sub_question}"
        )
        
        # Parse response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # Validate format
                if 'status' not in result:
                    result['status'] = 'no'
                if result['status'] == 'yes' and 'answer' not in result:
                    result['status'] = 'no'
                    result['analysis'] = 'Answer extraction failed'
                if 'analysis' not in result:
                    result['analysis'] = 'No analysis provided'
                    
                trace.output_data["parsed_result"] = result
                trace.success = True
            else:
                result = {
                    "status": "no",
                    "analysis": "Failed to parse response, information may be insufficient"
                }
                trace.success = False
        except Exception as e:
            logger.error(f"Failed to parse Analysis-Answer: {e}")
            result = {
                "status": "no",
                "analysis": f"Parse error: {str(e)}"
            }
            trace.success = False
        
        self.trajectory_memory.add_trace(trace)
        logger.info(f"Analysis result: status={result['status']}")
        
        return result
    
    def role_rewrite_agent(self, sub_question: str, failure_info: str) -> Dict:
        """Play Rewrite Agent role - query rewriting"""
        logger.info("✏️ Rewrite Agent: Starting query rewrite")
        
        # Construct Rewrite Agent prompt - strictly following the paper (3B model style but using 7B)
        prompt = f"""You are an expert query rewriter for information retrieval.

## Rewrite Task
Original Question: {sub_question}
Failure Reason: {failure_info}

## Instructions
1. Analyze why the current query failed to retrieve relevant information
2. Generate an improved search query using keyword expansion and synonyms
3. Focus on key entities, concepts, and alternative phrasings
4. Keep the rewritten query concise but comprehensive

## Output JSON Format
{{
  "rewritten_query": "improved search query with expanded keywords",
  "strategy": "brief explanation of rewrite approach",
  "keywords": ["key", "terms", "and", "synonyms"]
}}

Generate rewrite:"""
        
        # Generate response
        response = self.generate(prompt, max_new_tokens=256, temperature=0.1)
        
        # Record trajectory
        trace = ExecutionTrace(
            timestamp=datetime.now().isoformat(),
            agent_role="Rewrite",
            input_data={
                "sub_question": sub_question,
                "failure_info": failure_info
            },
            output_data={"raw_response": response},
            reasoning_process=f"Rewriting query due to: {failure_info[:100]}..."
        )
        
        # Parse response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if 'rewritten_query' not in result:
                    result['rewritten_query'] = sub_question + " detailed information"
                if 'strategy' not in result:
                    result['strategy'] = "keyword expansion"
                if 'keywords' not in result:
                    result['keywords'] = []
                    
                trace.output_data["parsed_result"] = result
                trace.success = True
            else:
                result = {
                    "rewritten_query": sub_question + " more information details",
                    "strategy": "default expansion",
                    "keywords": []
                }
                trace.success = False
        except Exception as e:
            logger.error(f"Failed to parse Rewrite: {e}")
            result = {
                "rewritten_query": sub_question + " additional context",
                "strategy": f"fallback due to error: {str(e)}",
                "keywords": []
            }
            trace.success = False
        
        self.trajectory_memory.add_trace(trace)
        logger.info(f"Rewrite strategy: {result['strategy']}")
        
        return result
    
    def _format_documents(self, documents: List[Dict]) -> str:
        """Format documents for prompt"""
        if not documents:
            return "No documents available"
        
        formatted = []
        for i, doc in enumerate(documents[:5], 1):  # Maximum 5 documents
            title = doc.get('title', f'Document {i}')
            content = doc.get('content', '')[:300]  # Maximum 300 characters per document
            formatted.append(f"[{i}] {title}: {content}...")
        
        return '\n'.join(formatted)
    
    def _resolve_placeholders(self, sub_question: str, previous_results: Dict[int, str]) -> str:
        """Resolve placeholders - core mechanism"""
        resolved = sub_question
        
        # Find all placeholders [entity from step X]
        placeholder_pattern = r'\[([^]]+) from step (\d+)\]'
        matches = re.findall(placeholder_pattern, sub_question)
        
        for info_type, step_id in matches:
            step_id = int(step_id)
            if step_id in previous_results:
                # Replace placeholder with actual result
                placeholder = f'[{info_type} from step {step_id}]'
                resolved = resolved.replace(placeholder, previous_results[step_id])
        
        return resolved
    
    def answer_question(self, question: str) -> Tuple[str, TrajectoryMemory]:
        """Main entry - complete CoT+RAG reasoning process"""
        start_time = time.time()
        logger.info(f"\n{'='*50}\nProcessing question: {question}\n{'='*50}")
        
        # Initialize trajectory memory
        self.trajectory_memory = TrajectoryMemory(question=question, strategic_plan={})
        
        # Step 1: PLAN Agent - question decomposition
        plan = self.role_plan_agent(question)
        self.trajectory_memory.strategic_plan = plan
        
        # Step 2: Execute sub-questions
        sub_results = {}  # step_id -> answer
        final_answer = None
        
        for sub_q_data in plan['sub_questions']:
            sub_q = SubQuestion(
                step_id=sub_q_data['step_id'],
                sub_question=sub_q_data['sub_question'],
                goal=sub_q_data['goal'],
                dependencies=sub_q_data.get('dependencies', []),
                expected_info_type=sub_q_data.get('expected_info_type', 'general')
            )
            
            # Check if all dependencies are satisfied
            missing_deps = [dep for dep in sub_q.dependencies if dep not in sub_results]
            if missing_deps:
                logger.warning(f"⏭️ Skipping sub-question {sub_q.step_id} due to missing dependencies: {missing_deps}")
                continue
            
            # Resolve placeholders
            resolved_question = self._resolve_placeholders(sub_q.sub_question, sub_results)
            logger.info(f"\nProcessing sub-question {sub_q.step_id}: {resolved_question}")
            
            # Initial retrieval
            documents = self.retrieve_documents(resolved_question, k=5)
            self.trajectory_memory.retrieval_rounds += 1
            
            # Analysis-Answer Agent
            analysis_result = self.role_analysis_answer_agent(resolved_question, documents)
            
            # If information is insufficient, trigger Rewrite Agent
            max_retries = 2
            retry_count = 0
            
            while analysis_result['status'] == 'no' and retry_count < max_retries:
                logger.info(f"Insufficient information, attempting query rewrite (attempt {retry_count + 1}/{max_retries})")
                
                # Rewrite Agent
                rewrite_result = self.role_rewrite_agent(
                    resolved_question, 
                    analysis_result['analysis']
                )
                
                # Retrieve again with rewritten query
                documents = self.retrieve_documents(rewrite_result['rewritten_query'], k=5)
                self.trajectory_memory.retrieval_rounds += 1
                
                # Analyze again
                analysis_result = self.role_analysis_answer_agent(resolved_question, documents)
                retry_count += 1
            
            # Record sub-question results
            if analysis_result['status'] == 'yes':
                sub_results[sub_q.step_id] = analysis_result['answer']
                logger.info(f"✅ Sub-question {sub_q.step_id} succeeded: {analysis_result['answer']}")
            else:
                logger.warning(f"❌ Sub-question {sub_q.step_id} failed: {analysis_result['analysis']}")
                # Don't record failed results - this will cause dependent steps to be skipped
        
        # Step 3: Synthesize answer
        if sub_results:
            # Use the answer from the last sub-question as final answer
            last_step_id = max(sub_results.keys())
            final_answer = sub_results[last_step_id]
        else:
            final_answer = "Unable to find answer"
        
        # Complete trajectory recording
        self.trajectory_memory.final_answer = final_answer
        self.trajectory_memory.total_time = time.time() - start_time
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Final answer: {final_answer}")
        logger.info(f"Total time: {self.trajectory_memory.total_time:.2f} seconds")
        logger.info(f"Retrieval rounds: {self.trajectory_memory.retrieval_rounds}")
        logger.info(f"Execution traces: {len(self.trajectory_memory.execution_traces)}")
        logger.info(f"{'='*50}\n")
        
        return final_answer, self.trajectory_memory


def main():
    """Main function - test CoT+RAG baseline"""
    # Initialize system
    opera = OPERACoTRAGBaseline()
    
    # Test questions
    test_questions = [
        "What is the capital of the country where the inventor of the telephone was born?",
        "Who directed the movie that won the Academy Award for Best Picture in 2010?",
        "What year was the company founded that acquired WhatsApp?",
    ]
    
    results = []
    for question in test_questions:
        answer, trajectory = opera.answer_question(question)
        results.append({
            'question': question,
            'answer': answer,
            'trajectory': trajectory.to_dict()
        })
    
    # Save results
    output_file = f"opera_cot_baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()