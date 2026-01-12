# Multi-Agent RAG System for Literary Consistency Analysis

## 1. Introduction

This project implements a sophisticated **Multi-Agent Retrieval-Augmented Generation (RAG) system** designed to verify the consistency of fictional character backstories against source novels. Built for the IIT Kharagpur Data Science Hackathon 2026, this system addresses a fundamental challenge in narrative analysis: determining whether a newly proposed character backstory contradicts, aligns with, or can plausibly coexist within an established literary work.

Traditional approaches to this problem suffer from two critical weaknesses. First, they attempt to process entire novels (100,000+ words) in a single LLM context window, leading to information loss and computational inefficiency. Second, they rely on surface-level text matching rather than deep causal reasoning about narrative constraints, character development, and temporal consistency. Our system overcomes these limitations through a novel three-agent architecture where each agent specializes in a distinct cognitive task: claim decomposition, evidence retrieval, and consistency judgment.

The core innovation lies in the **dependency injection pattern** that connects these agents. Rather than passing data through intermediate files or serialization layers (which cause metadata loss and introduce failure points), our agents communicate through structured Python objects that preserve critical information such as semantic similarity scores, source attribution, and claim provenance. This "zero data loss" architecture ensures that when the final Judge agent makes its verdict, it has full context not just about *what* the evidence says, but also *how relevant* that evidence is and *which specific claims* it addresses.

Powered by state-of-the-art language models—Llama 3.3-70B for precise fact extraction, DeepSeek-R1 for deep reasoning and causal analysis, and Qwen3-Embedding-8B for semantic search—the system achieves 98.3% processing success rate on real-world test cases. The architecture is production-ready, featuring automatic retries, graceful error handling, progress tracking, and intermediate result persistence to ensure robustness even when processing large batches of complex literary analysis tasks.

---

## 2. System Architecture

### Overview

The system employs a **three-agent pipeline** where each agent performs a specialized cognitive function, connected through a data dependency graph that ensures information fidelity across the entire analysis chain.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: test.csv                              │
│              (Character Backstories + Novel References)              │
└────────────────────────────────┬────────────────────────────────────┘
                                 ↓
                    ┌────────────────────────┐
                    │   main.py (Orchestrator)│
                    │   - Batch Processing    │
                    │   - Error Handling      │
                    │   - Progress Tracking   │
                    └────────────┬────────────┘
                                 ↓
        ┌────────────────────────┼────────────────────────┐
        ↓                        ↓                        ↓
┌───────────────┐      ┌──────────────────┐     ┌─────────────────┐
│  AGENT 1      │      │  AGENT 2         │     │  AGENT 3        │
│  Decomposer   │──────│  Evidence        │─────│  Consistency    │
│               │      │  Retriever       │     │  Judge          │
│  Llama 3.3    │      │                  │     │  DeepSeek-R1    │
│  70B-Instruct │      │  Pathway Vector  │     │  Deep Reasoning │
│               │      │  Store + Qwen3   │     │                 │
└───────────────┘      │  Embedding-8B    │     └─────────────────┘
        │              └──────────────────┘              │
        │                       │                        │
        │  List[Claims]         │  List[Evidence]        │  (Prediction,
        │  ["Claim 1",          │  [{text, source,       │   Rationale)
        │   "Claim 2",          │    distance,           │   (0 or 1,
        │   ...]                │    claim_origin},      │   "Because...")
        │                       │   ...]                 │
        └───────────────────────┴────────────────────────┘
                                 ↓
                    ┌────────────────────────┐
                    │   OUTPUT: results.csv   │
                    │   (ID, Prediction,      │
                    │    Rationale)           │
                    └────────────────────────┘
```

### Component Breakdown

#### **Pathway Vector Store (`pathway_app/app.py`)**
The foundation of our retrieval system. This persistent server indexes the complete text of both novels using a sliding-window chunking strategy (800 characters per chunk, 100-character overlap) and generates semantic embeddings via the Qwen3-Embedding-8B model. The server exposes a REST API that accepts natural language queries and returns the most semantically similar text passages using cosine similarity search in the embedding space.

**Key Innovation:** Unlike traditional vector databases that store only the embedding vectors, our implementation preserves complete metadata (source file path, chunk position, similarity scores) which is critical for the Judge to assess evidence quality.

#### **Agent 1: Claim Decomposer (`decomposer.py`)**
Transforms complex, narrative-style backstories into atomic, verifiable claims using Llama 3.3-70B-Instruct. The agent is specifically prompted to extract 3-5 falsifiable statements that can be independently verified against the source text.

**Example Transformation:**
```
Input: "Learning that Villefort meant to denounce him, Noirtier handed 
        the conspiracy dossier to a British spy named Harrington."

Output: [
  "Villefort intended to denounce Noirtier to Louis XVIII",
  "Noirtier handed a conspiracy dossier to a British spy",
  "The British spy was named Harrington"
]
```

**Technical Details:**
- Uses `temperature=0.1` for consistent extraction
- Forces JSON output via `response_format` parameter
- Implements 4-layer fallback parsing (direct JSON, markdown cleanup, regex extraction, structure search)
- Validates claim count (2-5) to reject both under-extraction and hallucination

#### **Agent 2: Evidence Retriever (`reasoner.py` - Part 1)**
Performs **multi-query retrieval** by searching for each claim independently, then deduplicates results using MD5 hashing of normalized text. This approach is superior to single-query retrieval because it ensures that each specific claim receives focused evidence, rather than getting generic passages about the character.

**Data Structure:**
```python
class Evidence:
    text: str          # The actual passage from the novel
    source: str        # Which book: "The Count of Monte Cristo.txt"
    distance: float    # Semantic similarity (0.0 = perfect match, 1.0 = unrelated)
    claim_origin: str  # "Claim 1", "Claim 2", etc.
    text_hash: str     # For deduplication
```

**Critical Feature:** By preserving the `distance` score and `claim_origin`, the Judge can assess evidence quality. For example, evidence with distance 0.15 (highly relevant) carries more weight than evidence with distance 0.75 (weakly related).

#### **Agent 3: Consistency Judge (`reasoner.py` - Part 2)**
Uses DeepSeek-R1, a reasoning-specialized model, to analyze whether the backstory creates contradictions with the evidence. DeepSeek-R1 generates internal `<think>...</think>` blocks containing thousands of tokens of reasoning before outputting the final verdict.

**Judgment Criteria:**
1. **Global Consistency:** Does the backstory make later events illogical?
2. **Causal Reasoning:** Do cause-effect chains remain intact?
3. **Narrative Constraints:** Does it violate world-building rules?
4. **Evidence Quality:** Are high-relevance chunks contradictory or supportive?

**Output Format:**
```json
{
  "prediction": 0,  // 0 = CONTRADICT, 1 = CONSISTENT
  "rationale": "The backstory claims Noirtier gave files to a British spy named 
                Harrington, but the novel never mentions this character. However, 
                it does describe Noirtier's involvement in conspiracies..."
}
```

### Data Flow & Zero Data Loss Architecture

The critical architectural decision is the **direct object passing** between agents:

```python
# Traditional (LOSSY) approach:
claims = decompose_backstory(text)
save_to_json(claims, "claims.json")          # ❌ Metadata lost
evidence = retrieve_evidence(load_json("claims.json"))
save_to_json(evidence, "evidence.json")      # ❌ Distance scores lost
prediction = judge(load_json("evidence.json")) # ❌ Claim provenance lost

# Our (LOSSLESS) approach:
claims = decompose_backstory(text)            # Returns List[str]
evidence = retrieve_evidence(claims)          # Returns List[Evidence]
prediction = judge(text, evidence)            # Receives full Evidence objects
```

The `Evidence` class acts as a **shared contract** between Agent 2 and Agent 3. When the Judge sees:
```
Evidence 1: distance=0.12, claim_origin="Claim 1", text="Noirtier was a Bonapartist..."
```

It knows this is **high-quality, directly relevant evidence** for a specific claim. Without this metadata, the Judge would treat all evidence equally, leading to poor decisions.

### Error Handling & Robustness

1. **Three-Layer Retry Strategy:**
   - API calls: 3 retries with exponential backoff
   - Individual backstories: 2 retries with 5-second delay
   - Graceful degradation: Continue processing even if one backstory fails

2. **Progress Persistence:**
   - Intermediate results saved every 10 rows
   - Prevents data loss during long batch runs
   - Allows resume from failure point

3. **Comprehensive Logging:**
   - Real-time ETA calculation based on rolling average
   - Success/failure counters updated after each row
   - Detailed error traces captured for debugging

---

## 3. Setup

### System Requirements

- **Python:** 3.10 or higher
- **Memory:** Minimum 4GB RAM (8GB recommended for large batches)
- **Storage:** ~500MB for dependencies + novel files
- **Network:** Stable internet connection for API calls to Nebius

### Installation Steps

#### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd IIT-Kharagpur-Data-Science-Hackathon-Submission
```

#### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv KDSH-venv
source KDSH-venv/bin/activate  # Linux/Mac
# OR
KDSH-venv\Scripts\activate  # Windows
```

#### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Required Packages:**
```txt
pathway-ai>=0.9.0
pandas>=2.0.0
python-dotenv>=1.0.0
openai>=1.12.0
requests>=2.31.0
```

#### Step 4: Verify Installation
```bash
python -c "import pathway; import pandas; import openai; print('✅ All packages installed successfully')"
```

#### Step 5: Prepare Data Directory Structure
```bash
# Create necessary directories
mkdir -p data/books
mkdir -p output

# Place your novel files in data/books/
# Expected files:
#   - The Count of Monte Cristo.txt
#   - In search of the castaways.txt

# Place test.csv in data/
# Expected columns: id, book_name, char, caption, content
```

#### Step 6: Validate Environment
```bash
python src/config.py
```

**Expected Output:**
```
✓ Configuration loaded successfully.
  - Reasoning Model: deepseek-ai/DeepSeek-R1-0528
  - Extraction Model: meta-llama/Llama-3.3-70B-Instruct
  - Embedding Model: Qwen/Qwen3-Embedding-8B
  - API Base: https://api.tokenfactory.nebius.com/v1
```

---

## 4. Environment Configuration (.env)

### Create Environment File

Create a file named `.env` in the project root directory:

```bash
# Navigate to project root
cd /path/to/IIT-Kharagpur-Data-Science-Hackathon-Submission

# Create .env file
touch .env
```

### Environment Variables

Copy and paste the following into your `.env` file, replacing placeholder values:

```bash
# ============================================================================
# NEBIUS API CONFIGURATION
# ============================================================================

# Your Nebius API key (REQUIRED)
# Obtain from: https://studio.nebius.ai/
NEBIUS_API_KEY=your_nebius_api_key_here

# Nebius API base URL
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Reasoning Model (Agent 3: Consistency Judge)
# DeepSeek-R1 for deep causal reasoning and chain-of-thought
LLM_MODEL_REASONING=deepseek-ai/DeepSeek-R1-0528

# Extraction Model (Agent 1: Claim Decomposer)
# Llama 3.3 for precise fact extraction and JSON output
LLM_MODEL_EXTRACTION=meta-llama/Llama-3.3-70B-Instruct

# Embedding Model (Agent 2: Vector Store)
# Qwen3 for high-quality semantic embeddings
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B


# ============================================================================
# PATHWAY SERVER CONFIGURATION
# ============================================================================

# Vector store server host (default: localhost)
PATHWAY_HOST=0.0.0.0

# Vector store server port (default: 8000)
PATHWAY_PORT=8000


# ============================================================================
# OPTIONAL: PERFORMANCE TUNING
# ============================================================================

# Number of evidence chunks to retrieve per claim (default: 3)
# Higher values = more context but slower processing
# TOP_K_RETRIEVAL=3

# Temperature for reasoning model (default: 0.1)
# Lower = more deterministic, Higher = more creative
# TEMPERATURE_REASONING=0.1

# Temperature for extraction model (default: 0.0)
# Keep at 0.0 for consistent claim extraction
# TEMPERATURE_EXTRACTION=0.0
```

### Important Notes

1. **NEBIUS_API_KEY is mandatory** - The system will fail immediately if this is not set
2. **Do NOT commit .env to version control** - Add `.env` to your `.gitignore` file
3. **Model names must be exact** - Nebius requires specific model identifiers
4. **API base URL** - Use the Nebius Token Factory endpoint for fastest response times

### Verify Configuration

After creating `.env`, verify it's loaded correctly:

```bash
python src/config.py
```

You should see all configuration values printed without errors.

---

## 5. Running the System

### Quick Start (3 Commands)

```bash
# Terminal 1: Start the Vector Store Server
python pathway_app/app.py

# Wait for "✨ Ready to serve queries!"

# Terminal 2: Run the Main Pipeline
export NEBIUS_API_KEY='your-api-key'  # Only needed if .env doesn't work
python src/main.py
```

### Detailed Execution Guide

#### **Phase 1: Start Pathway Vector Store**

The vector store must be running before executing the main pipeline.

```bash
# Terminal 1
cd /path/to/IIT-Kharagpur-Data-Science-Hackathon-Submission
source KDSH-venv/bin/activate  # Activate your virtual environment
python pathway_app/app.py
```

**Expected Output:**
```
🚀 Starting Pathway RAG Vector Store Server...

📚 Reading books from: ./data/books
🔍 Embedding model: Qwen/Qwen3-Embedding-8B
📦 Chunk size: 800 chars with 100 overlap
🌐 Server running on http://0.0.0.0:8000

💡 API Endpoints:
   POST http://localhost:8000/v1/retrieve
   Body: {'query': 'your search query', 'k': 5}

✨ Ready to serve queries!
```

**Keep this terminal running** - Do not close it while processing backstories.

#### **Phase 2: Validate System (Optional but Recommended)**

Before processing all 60 test cases, validate the pipeline:

```bash
# Terminal 2 (new terminal)
cd /path/to/IIT-Kharagpur-Data-Science-Hackathon-Submission
source KDSH-venv/bin/activate
python test_pipeline.py
```

**Expected Output:**
```
🧪 MULTI-AGENT RAG PIPELINE - VALIDATION TESTS
═══════════════════════════════════════════════

✅ Environment: PASS
✅ OpenAI Client: PASS
✅ Pathway Server: PASS
✅ Agent 1 (Decomposer): PASS
✅ Agent 2 (Retriever): PASS
✅ Agent 3 (Judge): PASS
✅ Full Pipeline: PASS

🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION
```

If any test fails, review the error messages and fix the issue before proceeding.

#### **Phase 3: Test Single Backstory (Optional)**

Test the pipeline on a single backstory to verify output format:

```bash
python src/main.py --test-single 95
```

This will process only backstory ID 95 with verbose output, allowing you to inspect the complete reasoning chain.

#### **Phase 4: Process Full Dataset**

Execute the main pipeline on all test cases:

```bash
python src/main.py
```

**Processing Flow:**
```
🚀 MULTI-AGENT RAG PIPELINE - MAIN EXECUTION
═══════════════════════════════════════════════
Started at: 2026-01-11 12:47:19

📋 Check 1: Environment Configuration ✓
📋 Check 2: Input Data ✓ (60 rows)
📋 Check 3: Pathway Vector Store ✓
📋 Check 4: Output Directory ✓

✅ ALL PRE-FLIGHT CHECKS PASSED

🔄 PROCESSING 60 BACKSTORIES

[1/60] Processing ID: 95
   ✅ CONTRADICT
   ⏱️  Elapsed: 30.7s | Avg: 30.7s/row | ETA: 30.2m

[2/60] Processing ID: 136
   ✅ CONTRADICT
   ⏱️  Elapsed: 46.7s | Avg: 23.3s/row | ETA: 22.6m

...

[60/60] Processing ID: 44
   ✅ CONTRADICT
   ⏱️  Elapsed: 883.4s | Avg: 14.7s/row | ETA: 0.0m

═══════════════════════════════════════════════
✅ BATCH PROCESSING COMPLETE
═══════════════════════════════════════════════
Total processed: 60
Successful: 59
Failed: 1
Success rate: 98.3%
Total time: 14.7 minutes
Average: 14.7s per backstory
═══════════════════════════════════════════════
```

**Output Files Generated:**

1. **`output/results.csv`** - Main submission file
   ```csv
   Story ID,prediction,rationale
   95,0,"The backstory claims Noirtier handed files to a British spy..."
   136,0,"Evidence contradicts the timeline of events..."
   ```

2. **`output/results_with_diagnostics.csv`** - Extended version with debugging info
   ```csv
   Story ID,prediction,rationale,status,processing_time,num_claims,num_evidence
   95,0,"...",success,30.7,4,9
   ```

3. **`output/intermediate_results_N.csv`** - Auto-saved checkpoints (every 10 rows)

### Advanced Usage Options

#### **Verbose Mode** (Show Detailed Logs for Each Backstory)
```bash
python src/main.py --verbose
```

Shows Agent 1, 2, and 3 outputs for every backstory. Useful for debugging but generates extensive logs.

#### **Dry Run** (Validation Only, No Processing)
```bash
python src/main.py --dry-run
```

Runs all pre-flight checks without processing any data.

#### **Quick Validation** (Test First 5 Rows Only)
```bash
# Manually edit test.csv to keep only first 5 rows
head -n 6 data/test.csv > data/test_small.csv
# Then run with modified config pointing to test_small.csv
```

### Performance Optimization Tips

1. **Reduce Top-K for Faster Processing:**
   ```python
   # In config.py
   TOP_K_RETRIEVAL = 2  # Instead of 3
   ```

2. **Use Fast Model Variants:**
   ```bash
   # In .env
   LLM_MODEL_REASONING=deepseek-ai/DeepSeek-R1-0528-fast
   LLM_MODEL_EXTRACTION=meta-llama/Llama-3.3-70B-Instruct-fast
   ```

3. **Enable Pathway Caching:**
   ```python
   # In pathway_app/app.py
   server.run_server(with_cache=True)
   ```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| **"Cannot connect to vector store"** | Ensure `python pathway_app/app.py` is running in another terminal |
| **"NEBIUS_API_KEY not found"** | Check `.env` file exists and contains the key, or export manually |
| **"Test CSV not found"** | Verify `data/test.csv` exists with correct columns |
| **Very slow processing (>60s/row)** | Check network connection; consider using `-fast` model variants |
| **All predictions are 0 (CONTRADICT)** | See "Analysis & Improvements" section - rebalance Judge prompt |
| **High failure rate (>5%)** | Enable verbose mode to diagnose specific errors |

---

## 6. Thank You

Thank you for exploring this Multi-Agent RAG System for Literary Consistency Analysis. This project represents the culmination of advanced techniques in natural language processing, retrieval-augmented generation, and production software engineering.

### Acknowledgments

- **IIT Kharagpur Data Science Hackathon 2026** - For providing the challenge and platform
- **Nebius AI** - For providing access to state-of-the-art language models (DeepSeek-R1, Llama 3.3, Qwen3)
- **Pathway.com** - For the robust streaming data framework that powers our vector store
- **The Open Source Community** - For the incredible tools and libraries that made this possible

### Key Innovations

This system demonstrates several novel contributions to the field of automated literary analysis:

1. **Zero Data Loss Architecture** - Structured evidence passing preserves semantic similarity scores and claim provenance across the pipeline
2. **Multi-Query Retrieval with Deduplication** - Each claim receives focused evidence, then MD5 hashing removes duplicates
3. **Metadata-Aware Judgment** - The Judge sees evidence quality scores, not just text, enabling nuanced reasoning
4. **Production-Grade Robustness** - Three-layer retry strategy, progress persistence, and graceful degradation ensure 98%+ success rates

### Future Enhancements

Potential areas for extension:
- **Adaptive Top-K** - Dynamically adjust retrieval count based on evidence quality
- **Claim-Level Confidence Scoring** - Probabilistic verdicts per claim, aggregated for final decision
- **Cross-Novel Analysis** - Extend to verify consistency across multiple related books
- **Interactive Explanation UI** - Web interface to visualize evidence chains and reasoning paths

### Contact & Contributions

For questions, suggestions, or collaboration opportunities, please reach out via the hackathon platform or submit an issue on the repository.

**Built with ❤️ for IIT Kharagpur Data Science Hackathon 2026**


