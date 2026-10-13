# Implementation plan

*Companion to `PROPOSAL.md`, which remains the source of truth for **what** is run and measured. This document says **how** to build it: the module designs, the external-library integration points (verified against their sources — see §12), the hard technical problems and their checks, and the build order. Where this document makes a choice the proposal leaves open, the choice is flagged in §10 and either confirmed with Paul or recorded in the preregistration. If this document and the proposal disagree, the proposal wins and this document is wrong.*

Design stance: the smallest correct system. Every module below is traceable to a proposal requirement; §9 lists what is deliberately not built. Target: one 48 GB GPU (L40S class), bf16 throughout.

## 1. Environment and pins

- **Python ≥ 3.10, managed with uv.** One `pyproject.toml` + `uv.lock` at the root governs everything, including the submodule's imports (the submodule's own lock applies only to `uv sync` inside it).
- **Pinned external code** (every pin echoed in `configs/base.yaml`):
  - `assistant-axis` — git submodule at `external/assistant-axis`, checked out at commit `a989619` (verified public; **default branch is `master`**, so pin the commit, never a branch). Installed editable via `[tool.uv.sources]` (hatchling; the wheel ships only the `assistant_axis/` package — `pipeline/` scripts and `data/` are reached by path).
  - `steering-vectors==0.12.2` (PyPI; requires `transformers<5`; torch is not a dependency of it — ours governs).
  - `strong_reject` from git at a commit fixed at setup (HEAD verified 2026-10-13: `7a551d5`). Its deps (litellm, peft, …) are unpinned — the lock resolves them.
  - `lm_eval[ifeval]` at the latest release (v0.4.13 verified; re-check at setup). The `ifeval` extra (langdetect, immutabledict, nltk) is required or ifeval fails at import.
  - `transformers≥4.57` (assistant-axis's `ProbingModel` passes `dtype=`, not `torch_dtype=`), `peft`, `accelerate`, `vllm`, `datasets`, `scikit-learn`, `statsmodels`, `matplotlib`, `pytest`.
  - The HF **model revision** of `meta-llama/Llama-3.1-8B-Instruct` is pinned in `base.yaml` and passed everywhere (pipeline step 2 re-applies the chat template in a separate run; an upstream template change would silently shift spans).
- **Secrets** via `.env` (already gitignored): `HF_TOKEN` (must have accepted **both** the Llama 3.1 license and the Gemma terms — the StrongREJECT evaluator is a LoRA on gated `google/gemma-2b`), `OPENAI_API_KEY`. The assistant-axis pipeline and strong_reject both read `.env` themselves (python-dotenv).
- **`phase0.py --step setup-check`** gates everything (run before any long job): `uv lock` resolves; `huggingface_hub.auth_check` on both gated repos; the gated tokenizer's chat template contains `{%- set date_string = "26 Jul 2024" %}` (byte-compare against the verified mirror); `python -c "from strong_reject.jailbreaks import registered_jailbreaks; assert 'aim' in registered_jailbreaks"` (package-data smoke); `get_config("meta-llama/Llama-3.1-8B-Instruct").target_layer == 16`; `lm_eval` ifeval imports; env var `TESTING` unset (it silently swaps the harm scorer for pythia-14m); submodule commit `a989619` checked out.

## 2. Repository layout

Exactly as in `PROPOSAL.md` §Repository layout, with **one proposed addition**: `src/persona_robustness/splits.py`, the guarded split loader and attack assembler. The proposal mandates the behavior (ID-only split files, the `FINAL_EVAL` refusal, attack construction) but names no module for it. **Per repo rules this needs a one-line PROPOSAL.md update approved by Paul before Phase 0 ends**; the fallback is folding it into `generate.py`. No other layout changes. Config loading and run manifests live in `generate.py` (~100 lines), not a new module.

## 3. Data layer

### 3.1 `data/download.py`

Idempotent fetchers (skip if present), each writing `data/raw/<name>/` plus a `source.json` manifest (URL/repo, commit or revision, sha256, license, fetch date). Verified sources:

| Source | How | Check |
| --- | --- | --- |
| HarmBench behaviors | raw CSV `data/behavior_datasets/harmbench_behaviors_text_all.csv` at a pinned commit of `centerforaisafety/HarmBench` | parse with `csv.DictReader` (contextual rows contain multi-line quoted fields); filter `FunctionalCategory=="standard"` → exactly 200 rows; key `BehaviorID`. **Never** load HarmBench's own val/test CSVs (41/159 standard — a different split than ours) |
| AdvBench | raw CSV `data/advbench/harmful_behaviors.csv` from `llm-attacks/llm-attacks` | 520 rows; use column `goal` |
| XSTest | raw CSV `xstest_prompts.csv` from `paul-rottger/exaggerated-safety` (CC-BY-4.0) | 450 rows; filter `label=="safe"` → 250; key `id` |
| In-the-wild | `datasets.load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25")["train"]` (raw URLs 404 — use the datasets API) | 1,405 rows; text in `prompt`; the card itself says to dedup on it |
| OR-Bench | `load_dataset("bench-llm/or-bench", "or-bench-hard-1k")["train"]` | 1,319 rows (not 1,000); fields `prompt`, `category`; no id column → key sha1(prompt) |
| UltraChat | `load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")` | 207,865 rows; first turn is the `prompt` field; key `prompt_id` |
| Alpaca | `load_dataset("tatsu-lab/alpaca")["train"]` (**CC-BY-NC-4.0** — release-checklist item) | 52,002 rows; filter `input==""` first (~60%) so every prompt is a clean single-turn instruction; assert ≥ 820 remain |
| Roleplay wrappers | stub `data/raw/roleplay_wrappers.json`: the four Table-2 wrappers with a `{request}` placeholder and `"user_checked": false` | Phase 0 refuses to use them until Paul flips the flag |
| Qwen capping fixture | `hf_hub_download("lu-christina/assistant-axis-vectors", "qwen-3-32b/capping_config.pt", repo_type="dataset")` → `tests/fixtures/` (gitignored; fetched on demand, never committed or redistributed) | **test fixture only**; a guard test asserts no runtime module imports it |

Role data (275 roles + `default.json`, 240 extraction questions) ships with the submodule — nothing to download.

### 3.2 `data/make_splits.py`

Writes every split once to `data/splits/*.jsonl` — **IDs only**, committed to git. Refuses to overwrite a split file whose content would change. All sampling is deterministic from seed 0 via per-step derived seeds (`stable_hash("0:"+step_name)`) so that multiple draws from the same pool are not accidentally correlated; every secondary draw (template–request pairings, the Phase-0 label sample, the Phase-2 pilot attacks) is **materialized as a frozen file**, never recomputed. Splits: `harmbench_{dev,test}` (40/160), `roles_{train,dev,test}` (220/15/40), `roleplay_prompts_{dev,test}` (60/200), `itw_templates_{dev,test}` (20/80 by template, built after the Phase-0 judge labeling pass), `attack_pairs_{dev,test}` (frozen rows `{attack_id, source, template_ref, request_id, attack_type}`), `alpaca_{direction,calibration,check}` (520/200/100 disjoint), `advbench_direction` (520, index-paired with `alpaca_direction`), `orbench_dev` (200), `ultrachat_train` (10,000), `xstest_test` (250). In-the-wild dedup is exact match after whitespace normalization, applied **before** the 20/80 template split (so no template text can straddle dev/test); fuzzy near-dup collapse is a noted limitation.

### 3.3 `src/persona_robustness/splits.py`

```python
def load_split(dataset, split) -> list[dict]      # IDs -> full rows from data/raw/
def load_attacks(attack_type, split) -> list[dict]
def load_roleplay_prompts(split) -> list[dict]
```

`load_split` raises on `split=="test"` unless **all three** hold: `FINAL_EVAL=1`, the git tag `prereg-v1` exists, and `sha256(configs/preregistration.yaml)` matches the hash recorded at tag time (the proposal mandates both the env guard and no-test-before-tag; checking the tag closes the gap). The guard has **no bypass**: guard tests assert the exception and run on dev splits or synthetic fixture files. `load_attacks` keeps the two construction paths separate: StrongREJECT templates substitute `{forbidden_prompt}` via `str.replace`; in-the-wild templates get `template + "\n\n" + request`; wrappers substitute `{request}`.

## 4. Prompts and spans — `prompts.py`

Single owner of tokenization. Every prompt in the repo — engine, vLLM, training, and a rendered-sample assert inside capability runs — goes through it.

- `build_messages(system: str|None, user: str)` — `system=None` means the template's bare header block (the Llama 3.1 template always emits `Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n`; its own default `date_string` is `"26 Jul 2024"`, but we pass `date_string="26 Jul 2024"` explicitly anyway).
- `render(messages, tok) -> RenderedPrompt` with `input_ids`, `text`, and `a_span`: the trailing `<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n` token ids (`A_SUFFIX_IDS`, asserted at import against the pinned tokenizer). Rendering **raises** if the prompt does not end with exactly those ids — spans can never silently drift.
- `strip_bos(...)` / `render_direction_prompt(user, tok)` — steering-vectors tokenizes with `add_special_tokens=True` (verified), so pre-templated prompts passed to it must have the leading `<|begin_of_text|>` stripped, and refusal-direction prompts must end exactly at the A span.
- `ASSISTANT_PROMPT` and `NEUTRAL_PROMPT` as frozen constants (proposal text, verbatim).

Unit tests: identical tokens across runs; byte-exact header; A-span position for system/no-system/long-user prompts (Phase 0 step 9).

## 5. The generation + readout engine — `generate.py`, `readouts.py`

The heart of the repo. Every evaluated generation runs through HF `model.generate` with forward hooks; vLLM appears in exactly two call sites (pipeline step 1, training answers), because steering/capping/readout hooks are HF-only.

**`ArmContext`** = model (base or `PeftModel` with adapter active), tokenizer, optional system prompt, and a hook context (`ActivationSteering` for steer/cap/random, a steering-vectors handle for refusal edits, or `nullcontext()`).

Per batch (one pass, greedy, KV cache on):

1. **Build** ids via `prompts.render` (attack text at the start of the user turn; system message iff the arm has one). Runtime assert: every row's non-pad suffix equals `A_SUFFIX_IDS`.
2. **Left-pad** to length L (`padding_side="left"`, pad = eos). Left padding puts every row's A span at fixed suffix positions and generation start at L.
3. **Hook order (invariant, tested):** enter the intervention context **first**, then register readout hooks (`register_forward_hook` on each `model.model.layers[ℓ]`, resolved through `get_base_model()` under PEFT, reading `output[0]` — the post-block residual, equal to `hidden_states[ℓ+1]`, the same object assistant-axis extracts). Same-module hooks run in registration order, so readouts see **post-edit** activations; the readout hook returns `None` (pure reader). The engine asserts at attach time that its readout hooks are last in each module's `_forward_hooks`. Contexts are re-entered per batch and torn down in `finally`.
4. **Generate** (`do_sample=False`, `max_new_tokens` 256 for evals / 65 for resting position — see step 5, `use_cache=True`, `return_dict_in_generate=True`). The readout hook distinguishes prefill (seq dim > 1: accumulate `A_sum[ℓ] = out[:, -n_A:, :].sum(1)` and `A_last[ℓ] = out[:, -1, :]`) from decode steps (seq dim == 1: write `out[:, 0, :]` into a preallocated f16 buffer `R_buf[ℓ][:, j]`, j = 1..64; ≈ 270 MB at batch 16 — fine).
5. **After generate:** mask `R_buf` to tokens strictly before the first `<|eot_id|>`/`<|eom_id|>` in the returned sequences (single source of truth), capped at 64; means in f32, stored f16. One HF-generate subtlety is load-bearing here: **N new tokens get only N−1 decode forwards** — the final sampled token is never fed back, so the last token of a generation that hits `max_new_tokens` without an eot has no recorded activation. Eval rows are unaffected (the 64-token R cap is well below 256; eot-terminated rows always have every pre-eot activation), but a 64-token resting-position decode would record only 63 R activations — so **resting-position runs decode 65 tokens and read the first 64** (flagged, §10 A11/B5). **Per-batch cross-check:** buffer-derived counts must equal `min(64, gen_len, n_decode_steps)` derived from the returned sequences; hard fail on mismatch (this catches the off-by-one between "token generated" and "step that computes its activation").
6. **Write:** answer text → `restricted/<run_id>/<row_id>.txt` (file mode 0600, dir 0700); one metadata row → `generations.jsonl` (the proposal's exact schema; `harm_score`/judge fields null until back-filled by atomic rewrite), fsynced per batch; readout arrays → `readouts_shards/*.npz`, consolidated by `readouts.finalize` into `readouts.npz` (shards deleted only after a successful re-read).

**`readouts.npz`** (canonical float16 store; row-aligned with `generations.jsonl`): `A_mean (N,32,4096)`, `A_last (N,32,4096)`, `R_mean (N,32,4096)`, `R_len (N,)`. `A_last` is a third array beyond the proposal's wording because the refusal reading is defined at the **last** A-span token and is otherwise not recomputable from the store (flagged, §10; raises ~0.5 → ~0.75 MB/generation, inside the 150 GB budget). `readouts.py` is pure numpy over this store — any reading layer, including each Phase-1 sweep window's, is a post-hoc projection since all layers are logged.

**Correctness oracles (tests):** (a) engine A/R/A_last means equal a reference computed with `output_hidden_states=True` on a full no-cache forward over prompt+output, restricted to the same masked token set; (b) padding invariance — a prompt alone vs inside a mixed-length batch gives identical readings and greedy output; (c) hook order — with capping active, recorded band-layer projections never fall below τ, and the deliberately wrong registration order must fail.

**Run protocol** (owned by `generate.py`): `load_config` deep-merges `base.yaml ← arms/<arm>.yaml ← CLI flags`; `config_hash` = sha256 of canonical JSON, 8 hex; `new_run` creates `results/<phase>/<run_id>/` with `mkdir(exist_ok=False)` (never overwrite), writing `config.yaml` and `git_sha.txt` (repo sha + dirty flag + submodule sha + uv.lock sha; a dirty tree is a hard error for any test-split run). Incomplete runs (no `metrics.json`) may be **resumed** via the `generations.jsonl` ledger; completed runs are immutable.

**Memory/throughput:** batch 16 at ≤ 2,048 ctx ≈ 22 GB (weights 16 GB + KV ≈ 4.3 GB + buffers); batch 32 fits for short sets. ≈ 1.5–3 h per fully evaluated 2,100–2,900-generation setting; Phase 1's 18 test settings ≈ 35–55 GPU-h total — run as per-setting jobs, with the phase total flagged to Paul up front (24 GPU-hour rule).

## 6. Directions, constants, interventions — `directions.py` + adapters

### 6.1 Persona-space pipeline (adapter `assistant_axis.py`)

Run the five scripts by subprocess from the pinned submodule with **absolute** `--roles_dir/--questions_file/--output_dir` (their defaults are relative to the CWD and point outside the repo), `--question_count 120` (takes the *first* 120 questions in file order → 276 files × 5 prompts × 120 = 165,600 generations), outputs under `artifacts/persona_space/llama31_8b/`. Known semantics the adapter wraps:

- Steps 1/2 skip a role whose output **file** exists — a crash-truncated file silently blocks regeneration; `validate_partial_outputs` finds bad last-line JSON and deletes after asking.
- Step 3 (`gpt-4.1-mini` default, pinned in `base.yaml`; temperature 1 and `max_completion_tokens=10` are hardcoded) merges per-key and silently omits unparsed/errored keys — **rerun it once** to fill gaps; `--dry_run` first for the budget sign-off (~165k calls); lower `--requests_per_second/--batch_size` from the shipped 100/50 to the account tier.
- Step 4 drops roles with < 50 score-3 answers (`--min_count 50`) — parse and persist the `Summary: N successful … N failed` line; the failed count is the proposal's dropped-roles report.
- Step 5 writes the axis as a **raw unnormalized tensor** `(32, 4096)` = default mean − role mean; unit-normalize per layer ourselves.
- Steps 4/5 classify "default" by filename **substring** — no artifact filename may contain `default` (unit-tested).
- On a multi-GPU node, steps 1/2 auto-spawn one worker per GPU unless `CUDA_VISIBLE_DEVICES`/`--tensor_parallel_size` pin them.

### 6.2 Steering and capping (verified `ActivationSteering` semantics)

`ActivationSteering(model, vectors, coefficients=…, layer_indices=…, intervention_type="addition"|"capping", positions="all")` is a context manager registering forward hooks on `model.model.layers[i]`, rewriting `output[0]` — it works through `model.generate` with the KV cache (prefill tokens enter the cache already edited; each decode step edits the new token).

- **Addition does not normalize the vector.** Pass the **unit** axis v̂ per band layer and fold the scale into the coefficient: `coefficients = α·(m_asst,ℓ − m_roles,ℓ)` (R-span constants, per layer). Random steer: same, with the seed-0 random unit vector.
- **Capping normalizes internally**: `h ← h − clamp(⟨h, v̂⟩ − τ, 0)·v̂` — τ is in unit-direction units regardless of the stored vector's norm. Use `load_capping_config` + `build_capping_steerer` (forces `positions="all"`, coefficient 0).
- `positions="all"` always (`"last"` edits only the final prompt token during prefill).

### 6.3 Capping-config builder

Emit the **exact published format** (verified by downloading and disassembling the Qwen3-32B config): `{"vectors": {f"layer_{l}/contrast_role_pos3_default1": {"vector": bf16 (4096,), "layer": l}}, "experiments": [{"id": "layers_<lo>:<hi+1>-p<q/100>", "interventions": [{"vector": name, "cap": -τ_l}, …]}]}` — window end **exclusive** (inclusive band [22,25] → `layers_22:26`), stored vector = role mean − default mean (**the negative of the axis**; cosine with the published Qwen axis verified = −1.0 exactly), τ_l = q-th percentile of all per-answer step-2 means (all 276 roles incl. default, any judge score) projected onto the **unit** axis at ℓ. Tests: the Qwen fixture sign/format check; a builder → `build_capping_steerer` round trip counting hooked layers (the loader silently skips malformed interventions); a behavioral assert that the min observed unit-axis projection ≈ τ on a dev batch. Note the published percentile *population* is a reconstruction — no cap-computation script exists upstream; the fixture validates format and sign only (flagged, §10).

### 6.4 Refusal direction (adapter `steering_vectors.py`)

`train_steering_vector(model, tok, pairs, layers=None, layer_type="decoder_block", read_token_index=-1, aggregator=mean_aggregator(), batch_size=8, move_to_cpu=True)` — pairs are **(harmful, harmless) in that order** (the aggregator computes pos − neg; wrong order flips the sign), prompts rendered by `render_direction_prompt` (end at the A span, BOS stripped). Applying:

- **Removal** (ablate r̂_ℓ\* from every layer): hand-build `SteeringVector({l: r_lstar for l in range(32)})` — applying the trained vector directly would ablate each layer's *own* direction. `ablation_operator()` **ignores the multiplier** (it projects out the normalized vector).
- **Addition** at ℓ\* only: `SteeringVector({l_star: r_hat}).apply(model, multiplier=m_harmful − m_harmless)` with the default addition operator.
- Keep `min_token_index=0`: the hook's token mask is rebuilt per forward pass, and with the KV cache every post-prompt pass has seq-len 1.

Two intervention unit tests the proposal mandates directly (Phase 0 step 9): **addition writes exactly** `h + α·(m_asst,ℓ − m_roles,ℓ)·v̂` — asserted against a hand computation on a toy batch (this is the test that catches passing the raw-norm axis, which would silently steer by α·‖axis‖); and **capping leaves activations already above the floor untouched** (bit-identical above τ).

### 6.5 `artifacts/directions/llama31_8b.pt`

One torch dict: unit + raw axis, per-layer PCA (all components, ~1 GB, MeanScaler-centered), axis-vs-PC1 cosines, random unit directions (+ cosine with v̂), refusal directions + ℓ\*, and the scaling constants — **R-span: m_asst = mean projection of the default answers' step-2 per-answer means, m_roles = mean projection of the step-4 role vectors** (score-3 filtered, dropped roles excluded; the proposal deliberately uses this population for the reading constants and the all-answers step-2 population only for the capping percentile, §6.3); A-span constants from one hooked forward-only pass over the same persona-space prompts; refusal m_harmless/m_harmful at the last A token — plus provenance meta. Phase-1 outputs (band B, k, U_ℓ, reading layer) go to a **new** file `llama31_8b_subspace.pt` — artifacts are append-only like runs.

## 7. Judges — `judges.py`

One `JudgeClient` (OpenAI, pinned model/version from `base.yaml`, temperature 0, small max_tokens, token-bucket rate limit, backoff; unparseable → null, retried once, counted) behind task functions: `judge_refusal`, `judge_role_adoption`, `judge_persona_agreement` (blind; reference order seeded per row_id and logged), `label_itw_persona`, `judge_incoherent`; plus `keyword_refusal` (versioned list, cross-check only) and `broken_output` (4-word sequence repeated ≥ 5 times).

- **Cache:** sqlite at `artifacts/judge_cache.sqlite`; key = sha256(judge model ‖ prompt-file sha ‖ canonical input fields); value = the parsed label plus the judge reply truncated to 100 characters — never request payloads, so no answer text is stored outside `restricted/` (judge replies are single labels; the truncation guards against a judge echoing input). Editing a judge prompt invalidates exactly its entries.
- Judge functions take **paths** to answer files and read internally; only labels surface. (Acknowledged tension, as in the proposal's design: judge API calls transmit answer text in the request body; the agent never prints or reads it — text flows file → request through code only.)
- The persona-space judge is *not* this client: pipeline step 3 runs as shipped (§6.1).
- Prompts in `configs/judge_prompts/`, versioned; the Phase-0 validation protocol (≥ 80% blind identification; κ ≥ 0.7 on three pairings) gates their use.

## 8. Training — `train/`

### 8.1 `make_train_data.py`

vLLM, Assistant system prompt, the 10,000 UltraChat first turns (prompts > 2,048 tokens dropped before sampling — flagged), temperature 0.7, top_p 1.0 (flagged: the proposal pins only temperature), ≤ 512 tokens; prompts pre-rendered by `prompts.py` and passed as **token ids** (`prompt_token_ids`/`TokensPrompt`) — vLLM re-tokenizes raw strings with `add_special_tokens=True`, which would prepend a second `<|begin_of_text|>` and sample the training answers from a slightly different prompt than the evaluated Prompt arm sees; a setup assert encodes one rendered prompt through the vLLM path and compares ids against `prompts.render`. vLLM's own chat path is never used. Output `artifacts/lora/train_data/ultrachat_10k.jsonl` — benign, not restricted. Sharded/restartable.

### 8.2 Tokenization alignment (load-bearing)

The answer is tokenized **once** (`add_special_tokens=False`) → `y_ids`; teacher input = `render(ASSISTANT_PROMPT, user).input_ids + y_ids + [eot]`; student/base input = `render(None, user).input_ids + y_ids + [eot]`. Identical answer ids and known offsets by construction. Unit test: concatenation equals full-template tokenization on 100 real examples including multi-token unicode; a deliberately shifted-by-one case fails.

### 8.3 Teacher-coordinate cache

Built once per (band, k, U) and shared by all seeds: base model, no adapter, `torch.no_grad()`, forward hooks on band layers only (safe in a no-grad pass). Shards store per example `answer_ids` (int32) and `coords (|B|, |y|, k)` f16 (projected in fp32); `index.json` maps prompt_id → (shard, y_len, sha1(y_ids)). ≈ 0.8–1.3 GB at k ≈ 20–40; ~1–1.5 h. Training-time **hard assert**: the student batch's answer ids equal the cached `answer_ids` (sha1 match) — this is the check that the coordinate loss is aligned token-for-token.

**Z_ℓ:** over 500 fixed examples, one no-grad base pass on student inputs (LoRA B = 0 ⇒ student ≡ base at init): `Z_ℓ = mean over answer tokens of ‖U_ℓᵀ(h_base − h_teacher)‖²`. Assert `Z_ℓ > 1e-6` per layer (a near-zero Z_ℓ means the prompt doesn't move that layer's coordinates — abort loudly) and that the real step-0 loss ≈ 1 per layer within 10%.

### 8.4 `identity_lora.py`

One hand-written training loop (~200 lines, shared skeleton with Plain FT — no HF `Trainer`: the loss needs `disable_adapter()` passes Trainer fights). Adapter and optimization exactly as the proposal (r=16, α=32, dropout 0.05, 7 proj modules, `layers_to_transform = 0..band_end`; AdamW 1e-4, 3% warmup, micro-batch 4 × accum 8, 2 epochs ≤ 625 steps, bf16, gradient checkpointing, seeds 0/1/2 — a seed also reshuffles data order, flagged). Per micro-batch:

1. **Student pass** (adapter on, grad): `gradient_checkpointing_enable(use_reentrant=False)` + `output_hidden_states=True` — **not hooks**; hooks under checkpointing can double-fire or capture detached tensors, and the failure mode (coordinate loss silently training nothing while KL still moves) is invisible. Keep `hidden_states[ℓ+1]` for ℓ ∈ B and the final hidden states; drop the rest.
2. **Coordinate loss** in fp32 with fp32 `U_ℓ` on GPU: `mean_ℓ (1/(Z_ℓ|y|)) Σ_t ‖U_ℓᵀ h_student − c_teacher‖²`, masked to answer tokens.
3. **Base pass**: `with model.disable_adapter(), torch.no_grad():` same inputs; keep final hidden states at answer positions. (Two live passes + cached teacher = the proposal's accounting.)
4. **Chunked KL**: never materialize `(B, T, 128256)` logits; per 1,024-token chunk (config knob), `lm_head` on student (grad) and base (no-grad) hidden states, KL(student ‖ base) accumulated fp32 over answer tokens. Total `L = L_sub + λ·KL`, λ = 0.1 (0.01 fallback is dev-only).

**Probes (tests):** loss ≡ 0 (both terms) when teacher inputs = student inputs; nonzero adapter grads at λ = 0 after one step with teacher ≠ student, and `hidden_states[ℓ+1].requires_grad` under non-reentrant checkpointing; a 20-step overfit smoke on 8 examples drives the coordinate loss below 0.1.

**Checkpointing:** `peft.save_pretrained` → `artifacts/lora/identity_s{seed}/step_{n:04d}/` every 50 steps + final; `latest/` holds optimizer/scheduler/RNG state + data cursor for resume; `train_log.jsonl` per step (loss terms, KL, lr — no text); an existing `step_{n}` refuses. Checkpoint evaluation loads the base model once and swaps adapters. Memory ≈ 25–28 GB peak at micro-batch 4 × 2,048 tokens (halve on OOM). Identity ≈ 4.5–7 h/run, Plain FT ≈ 3–4 h/run; 6 runs total.

### 8.5 `context_distill.py` (Plain FT)

Same loop skeleton, data, adapter, optimizer, checkpoint cadence and seeds; loss = token cross-entropy on answer tokens; no system prompt, no teacher cache, no base pass, no KL. One forward pass per example.

## 9. Evaluations, stats, figures

### 9.1 Capability (`evals/capability.py`, verified lm-eval API)

Wrap the already-prepared model: `HFLM(pretrained=model_obj, tokenizer=tok, batch_size=8)` — the caller attaches the PEFT adapter / enters hook contexts **before** wrapping (HFLM never moves an object model; `peft=`/`device=` are silently ignored for object inputs). **Three separate `simple_evaluate` calls**: `mmlu` with `num_fewshot=5` (its YAML has no default), `gsm8k` and `ifeval` with `num_fewshot=None` (their YAMLs pin 5 and 0). Always pass `apply_chat_template=True, fewshot_as_multiturn=True, log_samples=False, bootstrap_iters=1000`; `system_instruction` = Assistant prompt for the Prompt arm **only**. Assert the exact result keys on first run (`acc,none`, `exact_match,strict-match` + `exact_match,flexible-extract`, `prompt_level_strict_acc,none`); store `results['results']` in `metrics.json`. GSM8K headline = strict-match, with a pre-tag dev-style smoke (`limit=20`, `write_out`) deciding whether `gsm8k_cot_llama` must be preregistered instead (chat-templated models can fail the `#### N` format wholesale; both filters always logged).

### 9.2 Over-refusal and persona evals

`overrefusal.py`: XSTest (every setting) and OR-Bench dev (Phase-1 sweep) through the engine → `judge_refusal` + keyword cross-check. `persona_evals.py`: role adoption (judge); persona agreement — blind judge comparing the arm's answer to cached greedy Prompt/Baseline references (`restricted/refs/`), gate ≥ 0.6 on dev for Identity LoRA; plus the Phase-0 judge-validation protocol.

### 9.3 `stats.py` and `plots.py`

`cluster_bootstrap_ci` (resample request ids with replacement, all of a request's attacks moving together, 10,000 draws, percentile CI); `paired_diff_ci`; per-seed H1/H2 (criterion must hold for **every** seed; pooling across seeds would violate the plan); `detectable_difference` (2.8 × clustered SE × √(40/160), attack mix fixed) + the in-the-wild power curve; `mixedlm` + request-bootstrap CIs (CPU batch job, `n_proc=cpu_count()`); `grouped_auroc` (LogisticRegression, GroupKFold by request); `build_summary` recomputes every aggregate from `generations.jsonl` + `readouts.npz` (never trusts `metrics.json`) → `results/summary.csv`, with a toy-run test asserting rebuild == snapshot. `plots.py`: Figures 1–4 + Table 1 (markdown + csv), matplotlib only, rebuildable from `summary.csv` + raw rows alone.

## 10. Flagged choices

**Layout/engineering (A) — confirm with Paul before Phase 0 ends:** A1 `splits.py` as the one layout addition (one-line PROPOSAL.md update first). A2 `A_last` third array in `readouts.npz`. A3 readout hooks after intervention hooks (post-edit values). A4 *all* answer text (benign included) under `restricted/` — one write path can never misroute a harmful answer. A5 ledger resume of incomplete runs; completed runs immutable. A6 judge cache = labels only, in `artifacts/`. A7 config/run helpers in `generate.py`; summary builder in `stats.py`. A8 `config_hash` 8 hex. A9 KL chunk 1024, engine batch 16/32 (knobs, not preregistered). A10 the test guard also checks the prereg tag + file hash. A11 resting-position runs decode 65 tokens (proposal says 64) so that all of the first 64 generated tokens have recorded activations — see §5 step 5.

**Preregistration (B) — fixed in `configs/preregistration.yaml` before the tag:** B1 GSM8K metric (strict-match, or `gsm8k_cot_llama` after the dev smoke). B2 steering coefficients per band layer: α·(m_asst,ℓ − m_roles,ℓ), R-span. B3 random steer = one seed-0 Gaussian unit direction per layer, same coefficients. B4 capping percentile population = all per-answer step-2 means projected on the unit axis (a reconstruction; upstream publishes no cap-computation script). B5 R span = generated tokens strictly before the first eot/eom, capped at 64 (resting-position runs decode 65 so all 64 activations exist). B6 the four † Identity-LoRA items (proposal defaults; Paul confirms before the tag). B7 judge prompts/versions, matching rule, band B, reading layer, k, α/q grids, thresholds. B8 training seeds also reshuffle data order.

**Ask-user (C):** the proposal's list, plus: C6 UltraChat split = `train_sft` + the 2,048-token prompt cutoff; C7 top_p = 1.0 for training answers; C8 in-the-wild config = `jailbreak_2023_12_25`, dedup = whitespace-normalized exact; C9 Alpaca filtered to `input==""`; C10 "seed 0" read as derived per-step seeds + frozen files for every secondary draw; C11 ℓ\* screened across all 32 layers with the keyword check, winner confirmed with the judge.

**Setup-time verifications (D):** the §1 setup-check list, plus: lock co-resolution (vllm pins torch tightly; steering-vectors needs transformers < 5; litellm constrains openai/httpx); lm-eval release + its `dataset_path` values at the pinned release (older releases differed); `HFLM(pretrained=PeftModel)` smoke including `disable_adapter()`; steering-vectors 0.12.2 against the resolved transformers (untested upstream after Feb 2025) — run the refusal extraction as an early smoke; the strongreject adapter's first download with the project token; Gemma tokenizer maps '1'..'5' to single tokens and a known refusal scores ≈ 0; measured evaluator VRAM/throughput before sizing jobs; one chat-templated MMLU render inspected (`write_out`, `limit=2`); Alpaca `input==""` count ≥ 820.

## 11. Deliberately not built

- Anything under **Later experiments** (random-subspace LoRA, Identity LoRA + Cap, stiffness LoRA + the stiffness metric, long conversations, k=1 axis-only, other training methods, Qwen replication, PAIR).
- Frameworks: no Hydra/OmegaConf/pydantic (plain YAML + asserts); no W&B/MLflow/TensorBoard (`train_log.jsonl` + run folders); no HF `Trainer`/`accelerate` launcher; no CLI framework, registry pattern, DAG runner, or dashboard.
- No multi-GPU/DDP/FSDP; no steered vLLM (impossible with this hook stack); no torch.compile/quantization.
- No dataset class hierarchy (lists of dicts); no database beyond the sqlite judge cache; no Docker/CI service (uv.lock + pytest; GPU tests behind `@pytest.mark.gpu`).
- Unused library paths, never called: `strong_reject.generate`/litellm, `apply_jailbreaks_to_dataset` (its `jailbroken_prompt` column is a messages list, wrong shape for our engine), lm-eval CLI, assistant-axis README's `load_model`/`extract_response_activations` (do not exist at the pin).
- No fuzzy dedup of in-the-wild templates; no caching of base-pass logits (the live no-grad pass is cheap and exact).

## 12. How the external work is used (one table)

| Library (pin) | What we call | What we never do |
| --- | --- | --- |
| assistant-axis `a989619` | pipeline 1–5 by subprocess (absolute paths, `--question_count 120`); `ActivationSteering` (addition: unit vector + scale-in-coefficient; capping: `build_capping_steerer`); `load_axis`, `project`, `compute_pca` + `MeanScaler`; `ProbingModel` conventions | edit it; use its README quick-start names (stale); pass raw-norm vectors to addition; run its scripts from inside `pipeline/` (CWD trap); let filenames contain `default` |
| steering-vectors `0.12.2` | `train_steering_vector(..., aggregator=mean_aggregator(), read_token_index=-1)`; hand-built `SteeringVector` dicts + `apply()` with `ablation_operator()`/addition | pass (harmless, harmful) order; rely on multiplier with ablation; set `min_token_index>0` during generation; forget BOS stripping |
| strong_reject `@<sha>` | `registered_jailbreaks[name](request)` (string path); `decode(response, template)` on every row; `evaluate_dataset(ds, ["strongreject_finetuned"], batch_size=32, empty_model_cache=False)` on in-memory Datasets built from `restricted/` files, with `forbidden_prompt` = the **plain** HarmBench request, never the jailbroken prompt | its generation loop; `apply_jailbreaks_to_dataset`; scoring undecoded base64; running with `TESTING` set in production |
| lm-eval `v0.4.13` | `HFLM(pretrained=<prepared model object>, tokenizer=…, batch_size=…)`; three `simple_evaluate` calls with explicit template flags | pass `num_fewshot` to ifeval; rely on `model_args`/`device` with an object model; leave `log_samples=True` (copies test data into run folders) |

## 13. Build order

| # | Step | Accept when |
| --- | --- | --- |
| 1 | pyproject + lock; submodule pinned; `.env`; `phase0.py setup-check` | Lock resolves; every §1 smoke passes |
| 2 | `prompts.py` + tests | Green against the pinned revision |
| 3 | `download.py`, `make_splits.py`, `splits.py` guard + tests (counts; dev∩test=∅; guard raises; frozen-bytes regeneration) | Split files committed; **layout addition confirmed with Paul** |
| 4 | Engine + `readouts.py` + adapters' hook paths + tests (dual oracle, padding invariance, hook order, intervention exactness, restricted hygiene, run collision/resume) | 20-benign-prompt smoke writes a complete, schema-valid run; shards→npz round trip; < 40 GB at batch 16 |
| 5 | Pipeline steps 1–5 (**ask first**: ~165k generations ≈ 8–14 h + ~165k judge calls; `--dry_run` sign-off) | Axis (32,4096) saved; dropped-role count captured; axis-vs-PC1 cosine computed |
| 6 | `directions.py` (PCA, random, refusal, constants, capping builder) + fixture tests | `llama31_8b.pt` written; ‖U_kᵀv̂‖ curve at layer 16 for k=1..50; Qwen fixture cosine = −1.0 |
| 7 | `judges.py` + cache + `adapters/strongreject.py` + tests (`TESTING=1` pythia smoke; cache invalidation; decode-before-score) | Known refusal scores ≈ 0 on the real evaluator; judge prompts v1 authored |
| 8 | `calibrate.py`, `stats.py` + tests (synthetic bootstrap coverage; summary rebuild == snapshot) | Green |
| 9 | Phase-0 checks (refusal ℓ\*, steering check, itw labeling + splits, label sample → **stop for Paul**, κ, judge validation, power) → `reports/phase0.md`, **stop** | Every Gate-0 input recorded; κ ≥ 0.7 ×3; judge ≥ 80% |
| 10 | Phase 1 dev sweep → band/q/reading layer/k; preregistration filled, † confirmed, approved, **tagged** | `prereg-v1` exists; guard opens |
| 11 | Phase 1 test runs + scoring + matching + capability + diagnosis → `reports/phase1.md`, **stop** | RP\* fixed; Gate-1 table filled |
| 12 | Phase 2 train data; teacher cache + alignment tests; Identity pilot s0 + gates | Pilot gates recorded; step-0 loss ≈ 1 |
| 13 | Plain FT s0; seeds 1–2 both arms; curves; full matched eval; analysis → `reports/phase2.md`, **stop** | 6 runs matched; Figures 1–3 + Table 1 rebuild from `summary.csv` alone |
| 14 | Phase 3 figures/tables/release manifest | License checklist (Alpaca CC-BY-NC flagged); nothing from `restricted/`/`results/` staged |

## 14. Top risks and their designed checks

1. **R-span misalignment under KV caching** → buffered capture masked by returned sequences; per-batch count cross-check; dual-oracle test.
2. **Readouts recorded pre-intervention** after a refactor → hook-order test (capped readings never below τ; wrong order must fail); attach-time assert.
3. **Capping sign/units/off-by-one** → Qwen fixture (cosine −1.0); builder round trip; behavioral τ assert on a dev batch.
4. **Identity-LoRA token misalignment or silent gradient detachment** → cached `y_ids` + sha1 train-time assert; gradient probe; overfit smoke.
5. **Dependency lock failure / library drift** → build step 1 is the gate; HF revision pinned; span tests against the pin.
6. **Silent data loss in the external pipeline** (truncated files, unparsed judge keys, dropped roles) → `validate_partial_outputs`; step-3 rerun; step-4 summary parsed and reported.
7. **Unscored base64 looking like robustness** → `decode` on every row keyed by template, unit-tested.
8. **Padding contamination** (edited pad positions; lm-eval's right-padded loglikelihood batches) → padding-invariance test; hook shape test on both layouts; rendered-header assert inside capability runs.
