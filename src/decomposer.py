"""
Agent 1: The Claim Decomposer

This module extracts atomic, verifiable claims from character backstories.
It uses Llama-3.3-70B-Instruct for precise fact extraction.

Data Flow:
    Backstory Text → Llama 3.3 API → JSON List → Validated Claims
"""

import os
import json
import requests
import re
from typing import List, Dict, Optional
from time import sleep
from config import NEBIUS_BASE_URL, NEBIUS_API_KEY, LLM_MODEL_EXTRACTION

# ============================================================================
# CONFIGURATION
# ============================================================================

# NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
# NEBIUS_BASE_URL = "https://api.studio.nebius.ai/v1"
DECOMPOSER_MODEL = LLM_MODEL_EXTRACTION

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# ============================================================================
# SYSTEM PROMPT FOR CLAIM EXTRACTION
# ============================================================================

DECOMPOSER_SYSTEM_PROMPT = """You are a fact extractor for narrative verification.

**YOUR TASK:**
Extract 4-6 TESTABLE claims from the backstory at different specificity levels.

**EXTRACTION STRATEGY:**

For each major assertion, extract:
1. **Core factual claim** - What happened?
2. **Character action claim** - What did characters do?
3. **Relationship claim** - How did characters interact?
4. **Motivation/Trait claim** - Why did they act this way?
5. **Timeline claim** - When did this happen?

**CLAIM QUALITY CHECKLIST:**
✓ Can be verified against novel text
✓ Specific enough to be testable
✓ Not dependent on made-up names/details that may not appear in novel
✓ Captures the ESSENCE of the backstory assertion

**EXAMPLE:**

Backstory: "Noirtier handed the conspiracy dossier to a British spy named Harrington, 
            which led to Villefort's downfall due to his guilt."

Good Claims:
1. "Noirtier was involved in political conspiracies" (CORE FACT)
2. "Noirtier shared sensitive information with foreign parties" (ACTION)
3. "Noirtier's actions had negative consequences for Villefort" (RELATIONSHIP/CONSEQUENCE)
4. "Villefort experienced a downfall or suffered consequences" (OUTCOME)
5. "Noirtier made decisions that conflicted with his family's interests" (MOTIVATION)

Bad Claims (TOO SPECIFIC):
✗ "A spy named Harrington received the files" - name may not exist in novel
✗ "The dossier was about a specific conspiracy" - detail may not be mentioned

**OUTPUT:**
{
  "claims": [
    "Core factual claim",
    "Character action claim", 
    "Relationship or consequence claim",
    "Outcome or result claim",
    "Character motivation or trait claim"
  ]
}

Return ONLY the JSON object."""


# ============================================================================
# MAIN DECOMPOSITION FUNCTION
# ============================================================================

def decompose_backstory(backstory: str, verbose: bool = True) -> List[str]:
    """
    Decompose a backstory into atomic, verifiable claims using Llama 3.3.
    
    Args:
        backstory: The full backstory text to decompose
        verbose: If True, print detailed progress information
        
    Returns:
        List of atomic claim strings (3-5 claims)
        
    Raises:
        ValueError: If API key is missing, backstory is empty, or response is invalid
        requests.RequestException: If API call fails after retries
    """
    # Validation
    if not NEBIUS_API_KEY:
        raise ValueError(
            "❌ NEBIUS_API_KEY environment variable not set!\n"
            "Set it with: export NEBIUS_API_KEY='your-key-here'"
        )
    
    if not backstory or not backstory.strip():
        raise ValueError("❌ Backstory cannot be empty")
    
    # Log start
    if verbose:
        print("\n" + "="*70)
        print("🔍 AGENT 1: CLAIM DECOMPOSER")
        print("="*70)
        print(f"📝 Backstory length: {len(backstory)} characters")
        print(f"🤖 Model: {DECOMPOSER_MODEL}")
    
    # Construct user prompt
    user_prompt = f"""Analyze this backstory and extract 3-5 atomic, verifiable claims:

BACKSTORY:
{backstory.strip()}

Extract the claims now. Return ONLY the JSON object with the "claims" array."""
    
    # Attempt API call with retries
    claims = None
    last_error = None
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if verbose:
                print(f"\n🔄 Attempt {attempt}/{MAX_RETRIES}: Calling Llama 3.3 API...")
            
            claims = _call_llm_api(user_prompt, verbose)
            
            # If successful, break out of retry loop
            if claims:
                break
                
        except requests.RequestException as e:
            last_error = e
            if verbose:
                print(f"⚠️  API call failed: {e}")
            
            if attempt < MAX_RETRIES:
                if verbose:
                    print(f"⏳ Retrying in {RETRY_DELAY} seconds...")
                sleep(RETRY_DELAY)
            else:
                if verbose:
                    print(f"❌ All {MAX_RETRIES} attempts failed")
                raise
        
        except ValueError as e:
            last_error = e
            if verbose:
                print(f"⚠️  Parsing failed: {e}")
            
            if attempt < MAX_RETRIES:
                if verbose:
                    print(f"⏳ Retrying with adjusted prompt...")
                sleep(RETRY_DELAY)
            else:
                if verbose:
                    print(f"❌ Could not extract valid claims after {MAX_RETRIES} attempts")
                raise
    
    # Final validation
    if not claims:
        raise ValueError(f"Failed to extract claims: {last_error}")
    
    # Success!
    if verbose:
        print(f"\n✅ Successfully extracted {len(claims)} claims:")
        for i, claim in enumerate(claims, 1):
            print(f"   {i}. {claim[:100]}{'...' if len(claim) > 100 else ''}")
        print("="*70 + "\n")
    
    return claims


# ============================================================================
# API CALL HANDLER
# ============================================================================

def _call_llm_api(user_prompt: str, verbose: bool) -> List[str]:
    """
    Make the actual API call to Nebius/Llama 3.3.
    
    Args:
        user_prompt: The formatted user prompt
        verbose: Print progress info
        
    Returns:
        List of extracted claims
        
    Raises:
        requests.RequestException: If API call fails
        ValueError: If response cannot be parsed
    """
    headers = {
        "Authorization": f"Bearer {NEBIUS_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": DECOMPOSER_MODEL,
        "messages": [
            {"role": "system", "content": DECOMPOSER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,  # Low temperature for consistent extraction
        "max_tokens": 2000,  # Enough for several detailed claims
        "top_p": 0.95,
        "response_format": {"type": "json_object"}  # Force JSON output
    }
    
    # Make API request
    response = requests.post(
        f"{NEBIUS_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=90
    )
    response.raise_for_status()
    
    # Parse response
    result = response.json()
    
    if verbose:
        print(f"📥 Received response from LLM")
    
    # Extract content
    try:
        raw_content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Invalid API response structure: {e}")
    
    # Parse claims from JSON
    claims = _parse_claims_from_json(raw_content, verbose)
    
    return claims


# ============================================================================
# JSON PARSING WITH FALLBACKS
# ============================================================================

def _parse_claims_from_json(raw_content: str, verbose: bool) -> List[str]:
    """
    Parse claims from LLM response with multiple fallback strategies.
    
    Args:
        raw_content: Raw JSON text from LLM
        verbose: Print parsing info
        
    Returns:
        List of claim strings
        
    Raises:
        ValueError: If no valid claims can be extracted
    """
    
    # Strategy 1: Direct JSON parsing
    try:
        parsed = json.loads(raw_content)
        claims = _extract_claims_from_dict(parsed)
        
        if claims:
            if verbose:
                print("✓ Parsed using Strategy 1: Direct JSON")
            return _validate_claims(claims)
    
    except json.JSONDecodeError:
        if verbose:
            print("⚠️  Strategy 1 failed, trying fallbacks...")
    
    # Strategy 2: Remove markdown code blocks and retry
    try:
        cleaned = _clean_markdown(raw_content)
        parsed = json.loads(cleaned)
        claims = _extract_claims_from_dict(parsed)
        
        if claims:
            if verbose:
                print("✓ Parsed using Strategy 2: Markdown cleanup")
            return _validate_claims(claims)
    
    except json.JSONDecodeError:
        if verbose:
            print("⚠️  Strategy 2 failed, trying regex extraction...")
    
    # Strategy 3: Regex extraction of JSON object
    try:
        json_match = re.search(r'\{[^{}]*"claims"[^{}]*\[[^\]]*\][^{}]*\}', 
                               raw_content, re.DOTALL)
        
        if json_match:
            parsed = json.loads(json_match.group(0))
            claims = _extract_claims_from_dict(parsed)
            
            if claims:
                if verbose:
                    print("✓ Parsed using Strategy 3: Regex extraction")
                return _validate_claims(claims)
    
    except (json.JSONDecodeError, AttributeError):
        if verbose:
            print("⚠️  Strategy 3 failed, trying array extraction...")
    
    # Strategy 4: Extract just the array
    try:
        array_match = re.search(r'\[([^\]]+)\]', raw_content, re.DOTALL)
        
        if array_match:
            array_content = array_match.group(0)
            claims = json.loads(array_content)
            
            if isinstance(claims, list):
                if verbose:
                    print("✓ Parsed using Strategy 4: Array extraction")
                return _validate_claims(claims)
    
    except json.JSONDecodeError:
        pass
    
    # All strategies failed
    if verbose:
        print(f"📄 Raw response:\n{raw_content[:500]}...")
    
    raise ValueError(
        "Could not extract valid claims from LLM response. "
        "The model may not have returned proper JSON format."
    )


def _clean_markdown(text: str) -> str:
    """Remove markdown code blocks from text."""
    # Remove ```json and ``` markers
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()


def _extract_claims_from_dict(data: Dict) -> Optional[List[str]]:
    """
    Extract claims list from various dict structures.
    
    Args:
        data: Parsed JSON dictionary
        
    Returns:
        List of claims or None if not found
    """
    # Direct list
    if isinstance(data, list):
        return data
    
    # Standard format: {"claims": [...]}
    if "claims" in data:
        return data["claims"]
    
    # Alternative keys
    for key in ["facts", "items", "statements", "assertions", "list"]:
        if key in data and isinstance(data[key], list):
            return data[key]
    
    # First list value found
    for value in data.values():
        if isinstance(value, list):
            return value
    
    return None


def _validate_claims(claims: List) -> List[str]:
    """
    Validate and clean the extracted claims.
    
    Args:
        claims: Raw list of claims
        
    Returns:
        Validated list of claim strings
        
    Raises:
        ValueError: If claims are invalid
    """
    if not claims:
        raise ValueError("Claims list is empty")
    
    if not isinstance(claims, list):
        raise ValueError(f"Claims must be a list, got {type(claims)}")
    
    # Convert to strings and filter empty
    validated = []
    for claim in claims:
        claim_str = str(claim).strip()
        if claim_str:
            validated.append(claim_str)
    
    if not validated:
        raise ValueError("All claims were empty after validation")
    
    if len(validated) < 2:
        raise ValueError(
            f"Expected 3-5 claims, got {len(validated)}. "
            "The backstory may be too short or the LLM failed to extract enough claims."
        )
    
    if len(validated) > 5:
        # Truncate to 5 if we got too many
        print(f"⚠️  Warning: Got {len(validated)} claims, using first 5")
        validated = validated[:5]
    
    return validated


# ============================================================================
# COMMAND-LINE TESTING
# ============================================================================

if __name__ == "__main__":
    """Test the decomposer with sample backstories"""
    
    # Test backstory 1: Count of Monte Cristo
    test_backstory_1 = """
    Learning that Villefort meant to denounce him to Louis XVIII, Noirtier 
    pre-emptively handed the conspiracy dossier to a British spy named Harrington. 
    He believed this would protect his family from the king's wrath by removing 
    the evidence before it could be used against them. However, this act of 
    betrayal haunted him for years. He never forgave himself for putting foreign 
    interests above his country's safety, and this guilt contributed to his eventual 
    paralysis, which he privately believed was divine punishment for his actions.
    """
    
    # Test backstory 2: Simple example
    test_backstory_2 = """
    Captain Grant was a skilled navigator who had sailed the Pacific for fifteen years
    before his ship disappeared near the coast of Patagonia. He had three children 
    waiting for him in Scotland, and had promised to return before Christmas of 1862.
    His crew included his brother Thomas and a young sailor named Jenkins.
    """
    
    print("\n" + "="*70)
    print("🧪 TESTING CLAIM DECOMPOSER")
    print("="*70)
    
    # Test 1
    print("\n📖 TEST 1: Complex backstory (Count of Monte Cristo)")
    print("-" * 70)
    try:
        claims_1 = decompose_backstory(test_backstory_1, verbose=True)
        print("\n🎯 RESULT:")
        for i, claim in enumerate(claims_1, 1):
            print(f"   {i}. {claim}")
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
    
    # Test 2
    print("\n\n📖 TEST 2: Simple backstory (Search for Castaways)")
    print("-" * 70)
    try:
        claims_2 = decompose_backstory(test_backstory_2, verbose=True)
        print("\n🎯 RESULT:")
        for i, claim in enumerate(claims_2, 1):
            print(f"   {i}. {claim}")
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
    
    print("\n" + "="*70)
    print("✅ TESTING COMPLETE")
    print("="*70)