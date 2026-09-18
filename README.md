# Semantic Keys — Experiment Framework

Experimental framework for *"Semantic Keys: A Structured Language for Active
Inter-Agent Communication in Multi-Agent AI Systems."*

Three controlled experiments on HotpotQA multi-hop reasoning, run through a
three-agent pipeline (Researcher → Reasoner → Verifier), comparing the
Semantic Keys protocol against natural-language and JSON baselines.

**Design commitments**

- **Open weights.** Main models are Qwen 3.5 27B and Gemma 4 31B, served
  locally via vLLM. Weight hashes are pinned, so results remain reproducible
  after any proprietary API has been deprecated.
- **No LLM judge.** Every reported metric is computed by deterministic rules
  over experimentally-controlled ground truth. No language model
  participates in scoring, so no evaluator bias can enter the measurements.
- **Runs air-gapped.** Nothing in the experiment path requires network
  access. An online mode exists for development and for one optional
  validity check.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Confirm model repositories before downloading ~100 GB
export HF_TOKEN=hf_...                      # needed for gated repos (Gemma)
python3 scripts/verify_models.py

# 2. Five-minute pipeline check with a small model
./scripts/serve.sh dryrun                   # terminal A
./scripts/dryrun.sh                         # terminal B

# 3. Pilot on a real model
./scripts/serve.sh qwen                     # terminal A
./scripts/run.sh                            # terminal B

# 4. Repeat for the second model, then compare
./scripts/serve.sh gemma
./scripts/run.sh
python3 sk/compare.py --root results/pilot

# 5. Full run once the pilot looks sane
MODE=full ./scripts/run.sh
```

Add `--offline` to `serve.sh` and `run.sh` on an air-gapped machine.

---

## Layout

```
semantic-keys/
├── models.conf            Model identity — single source of truth
├── requirements.txt
├── docs/
│   ├── protocol.md        Judge-free measurement design (read first)
│   ├── methodology.md     Full measurement redesign and rationale
│   └── preregistration.md Frozen analysis plan for OSF
├── sk/                    Python modules
│   ├── runner.py          Experiment runner — the entry point
│   ├── scoring.py         Deterministic scoring rules
│   ├── metrics.py         EM/F1, embeddings, bootstrap, hypothesis tests
│   ├── protocols.py       Prompt templates for the six conditions
│   ├── injection.py       Entity substitution and attribute flip
│   ├── data.py            HotpotQA loading
│   ├── client.py          vLLM / API client
│   ├── analyze.py         Pre-registered hypothesis tests
│   ├── compare.py         Cross-model replication analysis
│   ├── sanity.py          Post-run sanity checks
│   ├── validation.py      Human annotation sheets and agreement scoring
│   └── judge_check.py     Rule-vs-judge agreement (online only)
├── scripts/
│   ├── verify_models.py   HuggingFace repo preflight
│   ├── serve.sh           Start vLLM (online or --offline)
│   ├── run.sh             Run experiments (online or --offline)
│   ├── dryrun.sh          Fast end-to-end pipeline check
│   ├── bundle_create.sh   Build an offline transfer bundle (run online)
│   └── bundle_install.sh  Install a bundle (run offline)
├── tests/
├── data/                  Dataset lands here
└── results/               Outputs land here, per mode and model
```

---

## The three experiments

| | Question | Primary metric |
|---|---|---|
| **1. Compression** | Does SK reduce tokens while preserving answerable content? | Output tokens; A-SRE via reconstruction scored with HotpotQA EM/F1 |
| **2. Error detection** | Does SK surface injected factual errors earlier and more specifically? | Propagation rate; detection rate (strict and lenient) |
| **3. Drift prevention** | Does SK's state anchoring resist topic drift across turns? | Embedding cosine drift; goal-maintenance EM/F1 |

**Conditions.** `NL-free`, `NL-concise`, `NL-CoT`, `JSON-free`, `SK`,
`SK-minimal`. The `JSON-free` baseline matters: it isolates whether SK's
effect is really about semantic structure rather than about JSON framing,
which is the first question a reviewer asks.

---

## How scoring avoids needing a judge

The experimental design already supplies ground truth, so scoring is rule
application rather than inference:

- **Propagation** — we inject the errors, so we know exactly which token was
  corrupted. An answer counts as propagated only if it is wrong against gold
  *and* carries the injected token. An answer that is wrong for an unrelated
  reason is a failure, not a propagation, and is excluded. This distinction
  is the main correction over naive substring scoring.
- **Detection** — verifiers emit a structured verdict, constrained by guided
  decoding. Strict detection requires naming the corrupted fact; lenient
  requires only a flag. Both are reported, and the gap between them is
  itself evidence about whether SK localises errors or merely raises
  suspicion.
- **Reconstruction (A-SRE)** — a model attempts the answer from the protocol
  message alone; HotpotQA's official EM/F1 scores the attempt. The model
  performs the task, the rule does the scoring.
- **Drift** — cosine distance between question and response embeddings. No
  judgment involved.

Full rationale in `docs/protocol.md`.

---

## Guided decoding

vLLM constrains **only the verifier verdict turn** to
`{"verified": bool, "reasons": [str]}`. That turn is a measurement
instrument shared identically by every condition, so constraining it
advantages no protocol.

Constraining the agents' primary outputs would be asymmetric — SK would gain
a reliability benefit that NL does not — so those stay unconstrained and the
comparison measures what each protocol naturally produces. State this in the
paper's implementation section.

`run.sh` smoke-tests guided decoding at startup and falls back with a
warning if the backend lacks it.

---

## Validation

Removing the LLM judge removes evaluator bias but not measurement error.
Two checks quantify how well the rules track human judgment.

**Human annotation (primary, required).**

```bash
python3 sk/validation.py --results-dir results/full/qwen3.5-27b \
    --output-dir validation
# Fill in the human_* columns in the three CSVs (~50 items each)
python3 sk/validation.py --score validation
```

Reports agreement and Cohen's κ separately for propagation and detection,
shows whether the rules over- or under-count, and calibrates the on-topic
thresholds. **Feed the calibrated thresholds back into the full run** using
the flags it prints.

**Judge cross-check (optional, online only).**

```bash
export ANTHROPIC_API_KEY=...
CROSS_VALIDATE=1 ./scripts/run.sh
```

Runs a frontier judge over the same observations and reports agreement with
the rules. The judge never feeds into a reported metric — this produces one
sentence for the validity section, of the form *"our deterministic
propagation rule agreed with an independent LLM judge in 94% of cases
(κ = 0.87)."*

If disagreements concentrate in a single condition, that is a measurement
problem rather than a finding, and the report flags it.

---

## Air-gapped workflow

**On an online machine:**

```bash
python3 scripts/verify_models.py --target-dir ~/bundle
./scripts/bundle_create.sh ~/bundle          # ~105 GB
```

Bundles both models, the embedder, HotpotQA, platform-matched Python wheels,
and this code. Writes `MANIFEST.txt` with config hashes for the paper's
reproducibility statement.

**Transfer, then on the offline machine:**

```bash
./scripts/bundle_install.sh ~/bundle ~/sk_experiments
cd ~/sk_experiments && source env_offline.sh
./scripts/serve.sh qwen --offline
./scripts/run.sh --offline
```

`HF_HUB_OFFLINE=1` is set throughout so a stray network call fails fast
instead of hanging on a DNS timeout.

---

## Reading pilot results before scaling up

**Both models show the same pattern** → proceed to the full run.

**Effect on one model only** → debug first. Usual causes: one model's
instruction following is weaker (check `report.txt` for unparsed verdicts),
or a prompt in `sk/protocols.py` assumes behaviour only one model exhibits.

**Neither model shows the effect** → the honest outcome. Do not scale N
hoping the effect emerges from noise reduction; an effect that is tiny at
N=20 will still be tiny at N=100. Revisit the protocol design, or report
the null result.

---

## Statistical plan

Seven pre-registered hypotheses, tested per model with Holm-Bonferroni
correction within each model's family, then classified across models as
REPLICATED / PARTIAL / NEITHER. Fisher-pooled p-values are reported as a
secondary summary only — the per-model plus replication verdict is the
primary claim. Details in `docs/preregistration.md`.

---

## Known limitations to report

1. **Strict detection under-credits vague verifiers.** A verifier that
   identifies a problem without naming the token scores as not-detected.
   The lenient variant covers this; quantify the gap from human validation.2. **Propagation requires token overlap.** An answer paraphrasing the
   injected entity rather than quoting it is not counted. Report the rate.
3. **On-topic is not independent of drift.** Both derive from the same
   cosine similarity. Present it as an interpretable re-expression, not as
   corroboration.
4. **Entity substitution uses heuristic type matching.** Person and place
   pools with preposition filtering, not full NER. Occasional surname/place
   confusions remain; spaCy NER would close the gap if reviewers press.
5. **Both main models are multimodal and ship thinking modes.** We use them
   text-only and disable thinking so that token counts reflect the
   communication protocol rather than hidden reasoning. `run.sh` checks for
   reasoning tokens and warns. State both facts in the paper.
6. **The two main models differ architecturally.** Qwen 3.5 is a hybrid
   Mamba/attention model; Gemma 4 31B is dense attention. This strengthens
   the cross-model replication argument — an effect surviving both is
   unlikely to be an artifact of one attention mechanism — but it also means
   a *failure* to replicate could reflect architecture rather than protocol.
   Report the architectures and treat any divergence accordingly rather than
   attributing it to model quality.

---

## Pitfalls that invalidate a run silently

Three failure modes reached a completed run with no error, producing numbers
that looked plausible and meant nothing. `sanity.py` now checks for all
three; they are recorded here because each cost a full pilot.

**Context truncation dropping evidence.** HotpotQA scatters the two
supporting passages anywhere among ten distractors. Slicing the joined
context to a character budget deleted them on most samples. Every condition
then correctly answered "the passages do not contain this information", and
accuracy went to zero across the board — indistinguishable, at a glance,
from a weak model. `extract_context_text` now orders supporting passages
first, and each row records `evidence_complete`.

**Reasoning tokens counted as protocol output.** Qwen 3.x and Gemma 4 emit
chain-of-thought as ordinary output unless disabled. Those tokens are billed
to `completion_tokens`, so Experiment 1 attributed them to protocol
verbosity, and long reasoning pushed the real answer past `max_tokens`,
zeroing accuracy. The client sends `enable_thinking=false`; `run.sh` checks
the effect rather than trusting the setting.

**Structured-output parameters silently ignored.** vLLM removed
`guided_json`; servers without it accept the request and return free prose.
Verdict parsing collapsed, and unevenly — SK emits JSON natively while NL
must append it, so the detection comparison became a test of format
compliance. Use `response_format` with a `json_schema`, and verify that
output actually conforms.

The pattern common to all three: a parameter or assumption failed quietly,
and the resulting numbers were self-consistent. Prefer checks that confirm
an *effect* over checks that confirm a *setting was sent*.

---

## Reproducibility statement material

Collect from `MANIFEST.txt` and each run's `run_metadata.json`:

- Model config SHA-256 for both main models, and the vLLM version
- Embedder model name and revision
- HotpotQA split and record count
- Seed, sample count, condition list, turn count
- Whether guided decoding was active
- Frozen on-topic thresholds from calibration