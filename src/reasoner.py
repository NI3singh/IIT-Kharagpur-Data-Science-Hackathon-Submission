"""
reasoner.py - Merged Evidence Retrieval + Consistency Judgment System

This file contains both Agent 2 (Evidence Retriever) and Agent 3 (Consistency Judge)
merged into a single coherent pipeline using dependency injection.

Architecture:
    Claims → retrieve_evidence_for_claims() → List[Evidence] → judge_consistency() → (prediction, rationale)

The Evidence class acts as the shared data structure, preserving:
- Text content
- Source metadata
- Similarity scores (distance)
- Claim origin tracking
"""

import os
import json
import re
import requests
import hashlib
from typing import List, Dict, Tuple, Set, Optional
from time import sleep

# Import configuration
from config import (
    get_openai_client,
    LLM_MODEL_REASONING,
    TOP_K_RETRIEVAL,
    PATHWAY_HOST,
    PATHWAY_PORT,
    TEMPERATURE_REASONING
)


# ============================================================================
# SHARED DATA STRUCTURES
# ============================================================================

class Evidence:
    """
    Shared evidence structure between retrieval and judgment.
    
    This is the "pipe" that connects Agent 2 output to Agent 3 input.
    By preserving metadata (distance, source, origin), we give the Judge
    full context about the quality and relevance of each evidence chunk.
    """
    
    def __init__(
        self,
        text: str,
        source: str,
        distance: float,
        claim_origin: str
    ):
        self.text = text.strip()
        self.source = source
        self.distance = distance  # Lower = more similar/relevant
        self.claim_origin = claim_origin  # Which claim generated this
        self.text_hash = self._compute_hash(self.text)
    
    def _compute_hash(self, text: str) -> str:
        """Compute normalized hash for deduplication."""
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "source": self.source,
            "distance": self.distance,
            "claim_origin": self.claim_origin
        }
    
    def __repr__(self) -> str:
        return f"Evidence(source={self.source}, len={len(self.text)}, dist={self.distance:.3f})"


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Vector Store API
VECTOR_STORE_URL = f"http://{PATHWAY_HOST}:{PATHWAY_PORT}/v1/retrieve"

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY_RETRIEVAL = 2
RETRY_DELAY_JUDGE = 5

# Request timeouts
TIMEOUT_RETRIEVAL = 30
TIMEOUT_JUDGE = 180  # 3 minutes for deep reasoning


# ============================================================================
# AGENT 2: EVIDENCE RETRIEVAL
# ============================================================================

def retrieve_evidence_for_claims(
    claims: List[str],
    top_k: int = TOP_K_RETRIEVAL,
    verbose: bool = True
) -> List[Evidence]:
    """
    Agent 2: The Evidence Retrieval Engine
    
    Performs multi-query retrieval with automatic deduplication.
    
    Args:
        claims: List of atomic claim strings from Agent 1
        top_k: Number of evidence chunks to retrieve per claim
        verbose: Print detailed progress
        
    Returns:
        List[Evidence]: Deduplicated evidence chunks with metadata
        
    Raises:
        ConnectionError: If vector store is not accessible
        ValueError: If claims list is invalid
    """
    
    # Validation
    if not claims or not isinstance(claims, list):
        raise ValueError("Claims must be a non-empty list of strings")
    
    if not all(isinstance(claim, str) and claim.strip() for claim in claims):
        raise ValueError("All claims must be non-empty strings")
    
    # Header
    if verbose:
        print("\n" + "="*70)
        print("🔍 AGENT 2: EVIDENCE RETRIEVAL ENGINE")
        print("="*70)
        print(f"📋 Claims to search: {len(claims)}")
        print(f"🎯 Top-K per claim: {top_k}")
        print(f"🌐 Vector store: {VECTOR_STORE_URL}")
    
    # Validate server connection
    _validate_vector_store_connection(verbose)
    
    # Multi-Query Retrieval with deduplication
    all_evidence: List[Evidence] = []
    evidence_hashes: Set[str] = set()
    
    for claim_idx, claim in enumerate(claims, 1):
        if verbose:
            print(f"\n{'─'*70}")
            print(f"🔎 Claim {claim_idx}/{len(claims)}")
            print(f"   📝 Query: {claim[:80]}{'...' if len(claim) > 80 else ''}")
        
        try:
            # Search for this specific claim
            claim_evidence = _search_single_claim(
                claim=claim,
                claim_number=claim_idx,
                top_k=top_k,
                verbose=verbose
            )
            
            # Deduplication
            unique_count = 0
            duplicate_count = 0
            
            for evidence in claim_evidence:
                if evidence.text_hash not in evidence_hashes:
                    evidence_hashes.add(evidence.text_hash)
                    all_evidence.append(evidence)
                    unique_count += 1
                else:
                    duplicate_count += 1
            
            if verbose:
                print(f"   ✅ Retrieved: {len(claim_evidence)} chunks")
                print(f"   ✓ Unique: {unique_count} | ⊗ Duplicates: {duplicate_count}")
        
        except Exception as e:
            if verbose:
                print(f"   ❌ Search failed: {e}")
                print(f"   ⏭️  Continuing with next claim...")
            continue
    
    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ RETRIEVAL COMPLETE")
        print(f"   📊 Total unique evidence: {len(all_evidence)}")
        print(f"   📚 Avg per claim: {len(all_evidence)/len(claims):.1f}")
        print(f"{'='*70}\n")
    
    return all_evidence


def _validate_vector_store_connection(verbose: bool) -> None:
    """Validate that Pathway vector store is accessible."""
    try:
        # Try minimal retrieve request
        response = requests.post(
            VECTOR_STORE_URL,
            json={"query": "test", "k": 1},
            timeout=5
        )
        
        if verbose:
            print(f"✓ Vector store connected")
    
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"❌ Cannot connect to vector store at {VECTOR_STORE_URL}\n\n"
            f"Make sure Pathway server is running:\n"
            f"  python pathway_app/app.py\n"
        )
    except Exception as e:
        if verbose:
            print(f"⚠️  Connection warning: {e}")


def _search_single_claim(
    claim: str,
    claim_number: int,
    top_k: int,
    verbose: bool
) -> List[Evidence]:
    """Search for a single claim with retry logic."""
    
    claim_label = f"Claim {claim_number}"
    last_error = None
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Make API request to Pathway
            response = requests.post(
                VECTOR_STORE_URL,
                json={"query": claim, "k": top_k},
                timeout=TIMEOUT_RETRIEVAL,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            # Parse response
            results = response.json()
            
            # Convert to Evidence objects
            evidence_list = _parse_search_results(
                results=results,
                claim_origin=claim_label
            )
            
            return evidence_list
        
        except requests.exceptions.Timeout:
            last_error = f"Timeout after {TIMEOUT_RETRIEVAL}s"
            if verbose and attempt < MAX_RETRIES:
                print(f"   ⚠️  Attempt {attempt} timed out, retrying...")
        
        except requests.exceptions.ConnectionError:
            last_error = "Connection failed"
            if verbose and attempt < MAX_RETRIES:
                print(f"   ⚠️  Connection failed, retrying...")
        
        except Exception as e:
            last_error = str(e)
            if verbose and attempt < MAX_RETRIES:
                print(f"   ⚠️  Error: {e}, retrying...")
        
        # Wait before retry
        if attempt < MAX_RETRIES:
            sleep(RETRY_DELAY_RETRIEVAL)
    
    # All retries failed
    error_msg = f"Failed after {MAX_RETRIES} attempts: {last_error}"
    if verbose:
        print(f"   ❌ {error_msg}")
    raise RuntimeError(error_msg)


def _parse_search_results(results: any, claim_origin: str) -> List[Evidence]:
    """Parse Pathway API response into Evidence objects."""
    
    evidence_list = []
    
    # Handle list responses
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            
            # Extract text
            text = item.get("text", "")
            if not text or not text.strip():
                continue
            
            # Extract source (try multiple keys)
            source = (
                item.get("source") or
                item.get("path") or
                item.get("metadata", {}).get("path") or
                item.get("metadata", {}).get("source") or
                "unknown"
            )
            
            # Extract distance/score
            distance = item.get("dist", item.get("distance", item.get("score", 0.0)))
            
            # Create Evidence object
            evidence = Evidence(
                text=text,
                source=source,
                distance=float(distance),
                claim_origin=claim_origin
            )
            evidence_list.append(evidence)
    
    # Handle nested dict responses
    elif isinstance(results, dict):
        if "results" in results:
            return _parse_search_results(results["results"], claim_origin)
        elif "data" in results:
            return _parse_search_results(results["data"], claim_origin)
    
    return evidence_list


# ============================================================================
# AGENT 3: CONSISTENCY JUDGMENT
# ============================================================================

# System prompt for the Judge
JUDGE_SYSTEM_PROMPT = """You are a strict literary consistency judge with expertise in narrative analysis.

**YOUR TASK:**
You will receive:
1. A CHARACTER BACKSTORY (hypothesis to verify)
2. EVIDENCE from the novel (actual text excerpts with relevance scores)

Determine: Is the backstory CONSISTENT with the evidence?

**CRITICAL ANALYSIS REQUIREMENTS:**

1. **Global Consistency**: Events must make sense across the timeline
   - Does the backstory contradict later events?
   - Would it make future actions illogical?

2. **Causal Reasoning**: Verify cause-and-effect
   - If backstory claims X, do later events support this?
   - Are causal chains preserved?

3. **Narrative Constraints**: Check for story-breaking elements
   - Does it violate established world rules?
   - Does it create impossible coincidences?

4. **Evidence Quality**: Consider relevance scores
   - High relevance (low distance) = strong evidence
   - Low relevance (high distance) = weak evidence
   - Use patterns across multiple chunks

**CONSISTENCY RULES:**

✓ CONSISTENT (1) if:
- Backstory aligns with character development
- No major contradictions with established facts
- Causal relationships make sense
- Could reasonably fit the narrative

✗ CONTRADICT (0) if:
- Direct contradictions (backstory says X, novel says NOT X)
- Creates logical impossibilities
- Makes character actions nonsensical
- Violates timeline or causality

**OUTPUT FORMAT:**
After analysis, output ONLY a JSON object:

{
  "prediction": 0,
  "rationale": "Brief explanation with specific evidence citations"
}

CRITICAL:
- prediction = 0 or 1 (integer)
- rationale = 2-4 sentences with evidence
- Think deeply - focus on logical consistency"""


def judge_consistency(
    backstory: str,
    evidence_list: List[Evidence],
    verbose: bool = True
) -> Tuple[int, str]:
    """
    Agent 3: The Consistency Judge
    
    Uses DeepSeek-R1 for deep reasoning to determine consistency.
    
    Args:
        backstory: The character backstory to verify
        evidence_list: List[Evidence] from Agent 2 (with metadata)
        verbose: Print reasoning process
        
    Returns:
        Tuple[int, str]: (prediction, rationale)
        - prediction: 0 (contradict) or 1 (consistent)
        - rationale: Explanation of decision
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If all retry attempts fail
    """
    
    # Validation
    if not backstory or not backstory.strip():
        raise ValueError("❌ Backstory cannot be empty")
    
    if not evidence_list:
        raise ValueError("❌ Evidence list cannot be empty")
    
    # Header
    if verbose:
        print("\n" + "="*70)
        print("⚖️  AGENT 3: CONSISTENCY JUDGE")
        print("="*70)
        print(f"🤖 Model: {LLM_MODEL_REASONING}")
        print(f"📝 Backstory length: {len(backstory)} chars")
        print(f"📚 Evidence chunks: {len(evidence_list)}")
    
    # Format evidence with metadata
    evidence_context = _format_evidence_with_metadata(evidence_list, verbose)
    
    # Construct judgment prompt
    user_prompt = _construct_judge_prompt(backstory, evidence_context)
    
    if verbose:
        print(f"📊 Total context: {len(user_prompt)} chars")
        print(f"\n🧠 Initiating deep reasoning...")
    
    # Call DeepSeek-R1 with retry logic
    prediction, rationale, thinking = None, None, None
    last_error = None
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if verbose:
                print(f"\n🔄 Attempt {attempt}/{MAX_RETRIES}: Calling {LLM_MODEL_REASONING}...")
            
            prediction, rationale, thinking = _call_judge_llm(
                user_prompt,
                verbose
            )
            
            # Success!
            break
        
        except Exception as e:
            last_error = str(e)
            if verbose:
                print(f"⚠️  Error: {e}")
            
            if attempt < MAX_RETRIES:
                if verbose:
                    print(f"⏳ Retrying in {RETRY_DELAY_JUDGE}s...")
                sleep(RETRY_DELAY_JUDGE)
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts: {last_error}"
                if verbose:
                    print(f"\n❌ {error_msg}")
                raise RuntimeError(error_msg)
    
    # Result
    if verbose:
        print(f"\n{'='*70}")
        print(f"⚖️  VERDICT: {'✓ CONSISTENT' if prediction == 1 else '✗ CONTRADICT'}")
        print(f"{'='*70}")
        print(f"\n📝 Rationale:")
        print(f"   {rationale}")
        
        if thinking:
            print(f"\n💭 Reasoning (excerpt):")
            excerpt = thinking[:300] + "..." if len(thinking) > 300 else thinking
            print(f"   {excerpt}")
        
        print(f"\n{'='*70}\n")
    
    return prediction, rationale


def _format_evidence_with_metadata(evidence_list: List[Evidence], verbose: bool) -> str:
    """
    Format evidence preserving metadata (distance, source, origin).
    
    This is CRITICAL - by including metadata, the Judge can weigh evidence quality.
    """
    RELEVANCE_THRESHOLD = 0.4  # Only use evidence with distance < 0.6
    
    filtered_evidence = [
        ev for ev in evidence_list 
        if ev.distance < (1 - RELEVANCE_THRESHOLD)
    ]
    
    if len(filtered_evidence) < 3:
        # Not enough quality evidence
        # Add a note to the Judge
        context_parts.append(
            "\n⚠️ WARNING: Limited high-quality evidence found. "
            "Consider this when judging - lack of evidence ≠ contradiction.\n"
        )
        # Use all evidence if we have too little
        filtered_evidence = evidence_list
    
    # Sort by relevance (distance)
    sorted_evidence = sorted(evidence_list, key=lambda e: e.distance)
    
    context_parts = []
    context_parts.append("EVIDENCE FROM THE NOVEL:")
    context_parts.append("=" * 60)
    
    for i, evidence in enumerate(sorted_evidence, 1):
        # Relevance score (inverse of distance, scaled 0-1)
        relevance = 1 - min(evidence.distance, 1.0)
        
        context_parts.append(f"\n[Evidence {i}]")
        context_parts.append(f"Source: {evidence.source}")
        context_parts.append(f"Relevance: {relevance:.2%} (distance: {evidence.distance:.3f})")
        context_parts.append(f"Related to: {evidence.claim_origin}")
        context_parts.append(f"\nText:\n{evidence.text}")
        context_parts.append("-" * 60)
    
    formatted = "\n".join(context_parts)
    
    if verbose:
        print(f"✓ Formatted {len(sorted_evidence)} evidence chunks with metadata")
    
    return formatted


def _construct_judge_prompt(backstory: str, evidence_context: str) -> str:
    """Construct the complete user prompt for the judge."""
    
    prompt = f"""Analyze consistency between this backstory and novel evidence.

{evidence_context}

CHARACTER BACKSTORY TO VERIFY:
{'=' * 60}
{backstory.strip()}
{'=' * 60}

ANALYZE:
1. Does evidence SUPPORT the backstory?
2. Does evidence CONTRADICT the backstory?
3. Are there logical inconsistencies?
4. Do causal relationships make sense?
5. Consider relevance scores - high relevance = stronger evidence

Output your verdict as JSON:
{{
  "prediction": 0,
  "rationale": "Your explanation"
}}

Remember:
- prediction = 1 (CONSISTENT) or 0 (CONTRADICT)
- Base decision on EVIDENCE and LOGIC"""
    
    return prompt


def _call_judge_llm(user_prompt: str, verbose: bool) -> Tuple[int, str, str]:
    """
    Call DeepSeek-R1 via OpenAI client from config.
    
    Returns:
        Tuple[int, str, str]: (prediction, rationale, thinking_process)
    """
    
    # Get OpenAI client from config
    client = get_openai_client()
    
    # Call the model
    response = client.chat.completions.create(
        model=LLM_MODEL_REASONING,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE_REASONING,
        max_tokens=16000,
        top_p=0.95
        # NOTE: Do NOT use response_format for DeepSeek-R1 - disables thinking!
    )
    
    # Extract content
    raw_content = response.choices[0].message.content
    
    if verbose:
        tokens = response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'
        print(f"   📊 Tokens used: {tokens}")
        print(f"   📥 Received {len(raw_content)} chars")
    
    # Parse response
    prediction, rationale, thinking = _parse_judge_response(raw_content, verbose)
    
    return prediction, rationale, thinking


def _parse_judge_response(raw_content: str, verbose: bool) -> Tuple[int, str, str]:
    """
    Parse DeepSeek-R1 response extracting prediction, rationale, and thinking.
    
    Handles <think>...</think> tags and multiple JSON formats.
    """
    
    # Extract thinking process
    thinking = _extract_thinking(raw_content)
    
    if verbose and thinking:
        print(f"   💭 Extracted {len(thinking)} chars of reasoning")
    
    # Remove thinking tags for clean answer
    answer_text = _remove_thinking_tags(raw_content)
    
    # Try multiple parsing strategies
    strategies = [
        ("Direct JSON", _parse_json_direct),
        ("Cleaned JSON", _parse_json_cleaned),
        ("Regex extraction", _parse_json_regex),
        ("Fallback", _parse_json_fallback)
    ]
    
    for strategy_name, strategy_func in strategies:
        try:
            prediction, rationale = strategy_func(answer_text)
            
            if verbose:
                print(f"   ✓ Parsed using: {strategy_name}")
            
            if prediction in [0, 1]:
                return prediction, rationale, thinking
        
        except:
            continue
    
    # All failed
    if verbose:
        print(f"   📄 Raw answer:\n{answer_text[:500]}...")
    
    raise ValueError("Could not extract valid prediction from LLM response")


def _extract_thinking(text: str) -> str:
    """Extract content from <think>...</think> tags."""
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _remove_thinking_tags(text: str) -> str:
    """Remove <think>...</think> tags."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _parse_json_direct(text: str) -> Tuple[int, str]:
    """Strategy 1: Direct JSON parsing."""
    parsed = json.loads(text)
    return int(parsed["prediction"]), str(parsed["rationale"])


def _parse_json_cleaned(text: str) -> Tuple[int, str]:
    """Strategy 2: Remove markdown code blocks first."""
    cleaned = re.sub(r'```json\s*', '', text)
    cleaned = re.sub(r'```\s*', '', cleaned).strip()
    parsed = json.loads(cleaned)
    return int(parsed["prediction"]), str(parsed["rationale"])


def _parse_json_regex(text: str) -> Tuple[int, str]:
    """Strategy 3: Regex extraction."""
    match = re.search(
        r'\{\s*"prediction"\s*:\s*(\d+)\s*,\s*"rationale"\s*:\s*"([^"]+)"\s*\}',
        text,
        re.DOTALL
    )
    if match:
        return int(match.group(1)), match.group(2)
    raise ValueError("No JSON pattern found")


def _parse_json_fallback(text: str) -> Tuple[int, str]:
    """Strategy 4: Find any JSON with prediction key."""
    for match in re.finditer(r'\{[^}]*"prediction"[^}]*\}', text, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            pred = int(parsed.get("prediction", -1))
            rat = str(parsed.get("rationale", "No rationale"))
            if pred in [0, 1]:
                return pred, rat
        except:
            continue
    raise ValueError("No valid JSON found")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_evidence_statistics(evidence_list: List[Evidence]) -> Dict:
    """Get statistics about retrieved evidence."""
    
    if not evidence_list:
        return {"count": 0}
    
    sources = [e.source for e in evidence_list]
    unique_sources = set(sources)
    
    return {
        "total_evidence": len(evidence_list),
        "unique_sources": len(unique_sources),
        "sources": list(unique_sources),
        "avg_text_length": sum(len(e.text) for e in evidence_list) / len(evidence_list),
        "total_characters": sum(len(e.text) for e in evidence_list),
        "avg_distance": sum(e.distance for e in evidence_list) / len(evidence_list)
    }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test the merged system"""
    
    print("\n" + "="*70)
    print("🧪 TESTING MERGED REASONER")
    print("="*70)
    
    # Test claims
    test_claims = [
        "Noirtier handed a conspiracy dossier to a British spy",
        "Villefort intended to denounce Noirtier to Louis XVIII",
        "The character experienced guilt about betraying his country"
    ]
    
    test_backstory = """
    Learning that Villefort meant to denounce him to Louis XVIII, Noirtier 
    pre-emptively handed the conspiracy dossier to a British spy named Harrington. 
    He believed this would protect his family from the king's wrath.
    """
    
    try:
        # Step 1: Retrieve evidence (Agent 2)
        print("\n" + "="*70)
        print("STEP 1: EVIDENCE RETRIEVAL")
        print("="*70)
        
        evidence = retrieve_evidence_for_claims(
            claims=test_claims,
            top_k=3,
            verbose=True
        )
        
        # Step 2: Judge consistency (Agent 3)
        print("\n" + "="*70)
        print("STEP 2: CONSISTENCY JUDGMENT")
        print("="*70)
        
        prediction, rationale = judge_consistency(
            backstory=test_backstory,
            evidence_list=evidence,
            verbose=True
        )
        
        # Final result
        print("\n" + "="*70)
        print("🎯 FINAL RESULT")
        print("="*70)
        print(f"Prediction: {prediction} ({'CONSISTENT' if prediction == 1 else 'CONTRADICT'})")
        print(f"Rationale: {rationale}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()