from dataclasses import dataclass

from transformers import LlamaConfig, LlamaForCausalLM


@dataclass
class ModelArgs:
    vocab_size: int
    pad_token_id: int
    sep_token_id: int
    eos_token_id: int
    num_hidden_layers: int
    hidden_size: int
    max_position_embeddings: int = 256
    intermediate_size: int = 2048
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    attention_dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    use_cache: bool = True


def build_llama_config(args: ModelArgs) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_position_embeddings,
        pad_token_id=args.pad_token_id,
        bos_token_id=args.sep_token_id,
        eos_token_id=args.eos_token_id,
        hidden_act="silu",
        attention_dropout=args.attention_dropout,
        rms_norm_eps=args.rms_norm_eps,
        rope_theta=args.rope_theta,
        tie_word_embeddings=False,
        use_cache=args.use_cache,
    )


def build_model(args: ModelArgs) -> LlamaForCausalLM:
    config = build_llama_config(args)
    return LlamaForCausalLM(config)


def count_parameters(
    model: LlamaForCausalLM,
    trainable_only: bool = False,
) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)
