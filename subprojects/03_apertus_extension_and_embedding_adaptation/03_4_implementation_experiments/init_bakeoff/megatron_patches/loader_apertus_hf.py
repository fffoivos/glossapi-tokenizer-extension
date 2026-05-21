"""HF → Megatron checkpoint loader for Apertus.

Round-2 reviewer flagged that the previous version of this loader emitted
Apertus-specific tensor keys (`mlp xielu alpha p/n`, `q norm weight`,
`k norm weight`) that `saver_core.py`'s `check_message()` rejects (the
saver protocol only consumes standard transformer keys). With default
checking, conversion fails; with `--no-checking`, those values get
silently dropped — silent-loss either way.

This revised loader sends **only** the saver_core-consumed standard keys
("word embeddings", "input norm weight", "post norm weight", "qkv weight",
"dense weight", "mlp l0 weight", "mlp l1 weight", "final norm" / "weight",
"output layer" / "weight"). It does NOT send xIELU or QK-Norm trained
tensors — saver_core has no slot for them in its `params_dict` mapping.

**Fidelity caveat (new risk R17 in RISKS.md):** the saved Megatron
checkpoint will have xIELU αp / αn / β / ε and QK-Norm q_norm/k_norm
at their **default init values** (αp=αn=0.8, β=0.5; q/k_norm at all-ones)
rather than Apertus's pretraining-trained values. For the bakeoff this
is a same-loss-across-all-three-arms compromise — comparison validity
holds, but absolute scores will differ from running unmodified Apertus.
For production CPT, a follow-up patcher is needed (see README.md
"Patching xIELU + QK-Norm post-conversion").

Tensor-name mapping (HF → Megatron message keys) follows the inverse of
saver_swissai_hf.py L237-345 (commit c92402e3...). Apertus-specific
architectural bits we still handle:

  * Bias-free everywhere (Apertus removes all bias terms — no bias keys
    sent; `--disable-bias-linear` declared in Megatron args).
  * GQA (num_heads = 32, num_kv_heads = 8). Megatron's `qkv_weight` is
    INTERLEAVED across kv groups; `_interleave_qkv` builds it from HF's
    separate q/k/v projections.
  * MLP is xIELU (single up_proj, no SwiGLU gate) — we send `mlp l0 weight`
    = HF's `up_proj.weight` (the "fc1"), NO l0_W / l0_V split.
  * Untied E / U (`--untie-embeddings-and-output-weights`).
  * RoPE θ = 500,000.

Wires into convert.py via the standard `add_arguments` + `load_checkpoint`
contract. Drop this file into `swiss-ai/Megatron-LM/tools/checkpoint/`
(via `install.sh`), then:

    python3 tools/checkpoint/convert.py \\
        --model-type GPT \\
        --loader apertus_hf \\
        --saver core \\
        --load-dir /path/to/Apertus-8B-2509-hf \\
        --save-dir /path/to/Apertus-8B-2509-megatron \\
        --tokenizer-model /path/to/Apertus-8B-2509-hf \\
        --bf16

Note `--model-type GPT` is REQUIRED by convert.py:114 (reviewer round-2 fix).

[Refs:
 - references/repos/swiss-ai_Megatron-LM/tools/checkpoint/saver_swissai_hf.py
 - references/repos/swiss-ai_Megatron-LM/tools/checkpoint/loader_llama_mistral.py
 - references/repos/swiss-ai_Megatron-LM/tools/checkpoint/saver_core.py L357-443 (check_message)
 - references/papers/apertus_2509.14233.pdf §2.1 (architecture)]
"""
import os
import sys
import types

import torch


def add_arguments(parser):
    group = parser.add_argument_group(title="Apertus HF loader.")
    group.add_argument('--true-vocab-size', type=int, default=None,
                       help="Original (pre-padding) vocab size. If set, padding rows beyond this are dropped by the saver.")
    group.add_argument('--vocab-file', type=str, default=None)
    group.add_argument('--tokenizer-model', required=True,
                       help="Path to HF tokenizer dir (typically same as --load-dir)")
    group.add_argument('--megatron-path', type=str, default=None,
                       help="Base directory of Megatron-LM-Swiss-AI repo (added to sys.path)")
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help="Apertus pretraining used 128 (submit_apertus_8b.sh:L193)")
    group.add_argument('--loader-transformer-impl', default='local',
                       choices=['local', 'transformer_engine'])
    # convert.py's top-level parser does NOT add --bf16/--fp16 — each loader
    # registers its own dtype flags. (Verified empirically against
    # tools/checkpoint/convert.py at commit c92402e3 — argparse error
    # "unrecognized arguments: --bf16" if these are omitted.)
    group.add_argument('--bf16', action='store_true', help='Load weights as bf16.')
    group.add_argument('--fp16', action='store_true', help='Load weights as fp16.')


def _interleave_qkv(q, k, v, num_heads, num_kv_heads, head_dim, hidden_size):
    """HF separate {q,k,v}_proj → Megatron interleaved qkv_weight.

    Apertus is GQA: num_heads / num_kv_heads = heads_per_group (e.g., 4).
    Megatron groups Q+K+V heads per kv-group, in this order per group:
        Q_0, Q_1, ..., Q_{heads_per_group-1}, K_0, V_0
    `saver_swissai_hf.py` L185-200 computes the corresponding q_slice,
    k_slice, v_slice. The slicing pattern is inverted below.
    """
    heads_per_group = num_heads // num_kv_heads
    qkv_total_heads = num_heads + 2 * num_kv_heads

    q_heads = q.reshape(num_heads, head_dim, hidden_size)
    k_heads = k.reshape(num_kv_heads, head_dim, hidden_size)
    v_heads = v.reshape(num_kv_heads, head_dim, hidden_size)

    qkv = torch.empty(qkv_total_heads, head_dim, hidden_size, dtype=q.dtype, device=q.device)
    for g in range(num_kv_heads):
        base = (heads_per_group + 2) * g
        qkv[base : base + heads_per_group] = q_heads[g * heads_per_group : (g + 1) * heads_per_group]
        qkv[base + heads_per_group] = k_heads[g]
        qkv[base + heads_per_group + 1] = v_heads[g]
    return qkv.reshape(qkv_total_heads * head_dim, hidden_size).contiguous()


def _load_checkpoint(queue, args):
    # Allow convert.py's directory ancestor as fallback megatron path.
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  os.path.pardir, os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        import transformers
        from transformers import AutoModelForCausalLM
        # Prefer the native ApertusForCausalLM if available (transformers >= 4.45-ish);
        # otherwise AutoModelForCausalLM with trust_remote_code=True loads Apertus's
        # `modeling_apertus.py` from the HF repo. Functionally equivalent for our
        # state_dict mapping.
        try:
            from transformers import ApertusForCausalLM
            ApertusLoaderCls = ApertusForCausalLM
            APERTUS_NATIVE = True
        except ImportError:
            ApertusLoaderCls = AutoModelForCausalLM
            APERTUS_NATIVE = False
    except ImportError as exc:
        print(f"transformers unavailable: {exc}", file=sys.stderr)
        queue.put("exit")
        raise

    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_global_variables
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy.model import module
        from megatron.legacy import fused_kernels
    except ModuleNotFoundError as exc:
        print(f"Megatron unavailable: {exc}; pass --megatron-path. Exiting.", file=sys.stderr)
        queue.put("exit")
        raise

    # === Load HF Apertus ===
    print(f"[loader_apertus_hf] loading HF Apertus from {args.load_dir} (native={APERTUS_NATIVE})", file=sys.stderr)
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    load_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if not APERTUS_NATIVE:
        load_kwargs["trust_remote_code"] = True
    hf_model = ApertusLoaderCls.from_pretrained(args.load_dir, **load_kwargs)
    hf_config = hf_model.config
    sd = hf_model.state_dict()
    print(f"[loader_apertus_hf] loaded: vocab={hf_config.vocab_size}, layers={hf_config.num_hidden_layers}, "
          f"hidden={hf_config.hidden_size}, heads={hf_config.num_attention_heads}/{hf_config.num_key_value_heads}",
          file=sys.stderr)

    # === Build Megatron args by mocking the CLI ===
    # This is the same pattern loader_llama_mistral.py uses: we declare all
    # arguments programmatically, then let Megatron's argparse build the
    # `margs` namespace that's downstream of validate_args(). We need
    # margs to populate `md.checkpoint_args` which the saver consumes.
    rope_scaling = hf_config.rope_scaling or {}
    rope_scaling_factor = rope_scaling.get("factor")

    sys_argv = [
        'loader_apertus_hf.py',
        '--no-masked-softmax-fusion',
        '--no-bias-gelu-fusion',
        '--no-bias-dropout-fusion',
        '--no-async-tensor-model-parallel-allreduce',
        '--use-cpu-initialization',
        '--micro-batch-size', '1',
        '--no-load-optim',
        '--no-load-rng',
        '--no-save-optim',
        '--no-save-rng',
        '--mock-data',
        '--no-initialization',
        '--load', args.load_dir,
        '--no-one-logger',
        # Apertus architecture (mirrors submit_apertus_8b.sh L181-199)
        '--num-layers', str(hf_config.num_hidden_layers),
        '--hidden-size', str(hf_config.hidden_size),
        '--ffn-hidden-size', str(hf_config.intermediate_size),
        '--num-attention-heads', str(hf_config.num_attention_heads),
        '--num-query-groups', str(hf_config.num_key_value_heads),
        '--group-query-attention',
        '--max-position-embeddings', str(hf_config.max_position_embeddings),
        '--seq-length', str(hf_config.max_position_embeddings),
        '--position-embedding-type', 'rope',
        '--rotary-base', str(int(hf_config.rope_theta)),
        '--make-vocab-size-divisible-by', str(args.make_vocab_size_divisible_by),
        '--normalization', 'RMSNorm',
        '--xielu',
        '--qk-layernorm',
        '--qknorm-impl', 'apex',
        '--disable-bias-linear',  # Apertus has no bias on linear layers
        '--tokenizer-type', 'HuggingFaceTokenizer',
        '--tokenizer-model', args.tokenizer_model,
    ]
    if not hf_config.tie_word_embeddings:
        sys_argv.append('--untie-embeddings-and-output-weights')
    if rope_scaling_factor is not None:
        sys_argv.extend(['--use-rope-scaling',
                         '--rope-scaling-factor', str(int(rope_scaling_factor))])
    if args.bf16:
        sys_argv.append('--bf16')
    elif args.fp16:
        sys_argv.append('--fp16')

    sys.argv = sys_argv
    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size
    margs = validate_args(margs)
    margs.use_legacy_models = True
    margs.transformer_impl = args.loader_transformer_impl
    margs.position_embedding_type = "rope"
    margs.params_dtype = dtype

    module.MegatronModule.embedding_warning_printed = True
    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    fused_kernels.load(margs)

    # === Build metadata ===
    md = types.SimpleNamespace()
    md.model_type = "GPT"
    md.num_layers = hf_config.num_hidden_layers
    md.hidden_size = hf_config.hidden_size
    md.seq_length = hf_config.max_position_embeddings
    md.num_attention_heads = hf_config.num_attention_heads
    md.max_position_embeddings = hf_config.max_position_embeddings
    md.tokenizer_type = "HuggingFaceTokenizer"
    md.iteration = 0
    md.params_dtype = dtype
    md.bert_binary_head = False
    md.output_layer = not hf_config.tie_word_embeddings
    md.position_embedding_type = "rope"
    md.linear_bias = False
    md.qkv_bias = bool(getattr(hf_config, "attention_bias", False))
    md.norm_has_bias = False
    md.swiglu = False                         # Apertus uses xielu, not swiglu
    md.previous_tensor_parallel_size = 1
    md.previous_pipeline_parallel_size = 1
    md.make_vocab_size_divisible_by = args.make_vocab_size_divisible_by
    md.true_vocab_size = args.true_vocab_size or hf_config.vocab_size
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0

    queue.put(md)

    def queue_put(name, msg):
        print(f"[loader_apertus_hf] sending {name}", file=sys.stderr)
        msg["name"] = name
        queue.put(msg)

    # === Send embeddings ===
    queue_put("embeddings", {
        "word embeddings": sd["model.embed_tokens.weight"].clone()
    })

    # === Per-layer ===
    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = hf_config.hidden_size // num_heads
    hidden = hf_config.hidden_size

    for i in range(hf_config.num_hidden_layers):
        p = f"model.layers.{i}"
        # ONLY saver_core-consumable standard keys here. saver_core.py:L357-443
        # pops exactly these keys per transformer layer; emitting anything else
        # triggers check_message() failure (with --checking) or silent drop
        # (with --no-checking). For Apertus-specific tensors (xIELU αp/αn,
        # QK-Norm q_norm/k_norm) the saved Megatron model uses XIELU.__init__
        # / RMSNorm.__init__ defaults — see README.md "Patching xIELU +
        # QK-Norm post-conversion" + RISKS.md R17 for the fidelity caveat.
        message = {
            "input norm weight":  sd[f"{p}.attention_layernorm.weight"].clone(),
            "post norm weight":   sd[f"{p}.feedforward_layernorm.weight"].clone(),
            "dense weight":       sd[f"{p}.self_attn.o_proj.weight"].clone(),
            "mlp l1 weight":      sd[f"{p}.mlp.down_proj.weight"].clone(),
            # xIELU MLP: single up_proj is Megatron's "mlp l0 weight" (no SwiGLU split).
            "mlp l0 weight":      sd[f"{p}.mlp.up_proj.weight"].clone(),
        }

        # Interleaved QKV
        q = sd[f"{p}.self_attn.q_proj.weight"]
        k = sd[f"{p}.self_attn.k_proj.weight"]
        v = sd[f"{p}.self_attn.v_proj.weight"]
        message["qkv weight"] = _interleave_qkv(q, k, v, num_heads, num_kv_heads, head_dim, hidden)

        if md.qkv_bias and f"{p}.self_attn.q_proj.bias" in sd:
            qb = sd[f"{p}.self_attn.q_proj.bias"]
            kb = sd[f"{p}.self_attn.k_proj.bias"]
            vb = sd[f"{p}.self_attn.v_proj.bias"]
            # Same interleave pattern on the bias axis
            qb_heads = qb.reshape(num_heads, head_dim)
            kb_heads = kb.reshape(num_kv_heads, head_dim)
            vb_heads = vb.reshape(num_kv_heads, head_dim)
            heads_per_group = num_heads // num_kv_heads
            qkv_total = num_heads + 2 * num_kv_heads
            qkv_b = torch.empty(qkv_total, head_dim, dtype=qb.dtype, device=qb.device)
            for g in range(num_kv_heads):
                base = (heads_per_group + 2) * g
                qkv_b[base : base + heads_per_group] = qb_heads[g * heads_per_group : (g + 1) * heads_per_group]
                qkv_b[base + heads_per_group] = kb_heads[g]
                qkv_b[base + heads_per_group + 1] = vb_heads[g]
            message["qkv bias"] = qkv_b.reshape(-1).contiguous()

        # NOTE: q_norm / k_norm (QK-Norm) and xIELU alpha_p / alpha_n / beta / eps
        # are intentionally NOT sent here. saver_core's check_message() at
        # L443 would reject them. See R17 in RISKS.md for the fidelity caveat
        # + post-conversion patcher plan.

        queue_put(f"transformer layer {i}", message)

    # === Final norm + output ===
    queue_put("final norm", {"weight": sd["model.norm.weight"].clone()})

    if md.output_layer:
        queue_put("output layer", {"weight": sd["lm_head.weight"].clone()})

    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except Exception:
        queue.put("exit")
        raise
