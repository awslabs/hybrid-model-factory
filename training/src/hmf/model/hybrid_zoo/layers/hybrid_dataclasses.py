from dataclasses import dataclass, asdict
from typing import Optional

import torch


@dataclass
class HybridMamba2Config:
    """
    dataclass for our custom Mamba2 class with state expansion.

    mamba2:
        use_qk_norm: Whether to apply QK (CB) norm to Mamba2.
        use_pos_emb: Whether to use positional embeddings.
    """

    use_qk_norm: bool = True
    use_pos_emb: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class HybridSWAConfig:
    """
    dataclass for sliding window attention. This allows us to specify parameters specific to
    SWA in the hybridization config as:

    swa:
        window_size: Sliding window size.
    """

    window_size: int

    def to_dict(self):
        return asdict(self)


@dataclass
class HybridBMOJOConfig:
    """
    dataclass for B'MOJO. This allows us to specify parameters specific to
    B'MOJO in the hybridization config as:

    bmojo:
        window_size: Specifies the fading context size and the local context size; hence, the total
            window size is window_size*2.
        tie_attn_weights: Whether projections for in-context and fading tokens should be tied.
        ssm_mixer: Which SSM module to use for processing the fading tokens.
    """

    window_size: int = 2048
    tie_attn_weights: bool = True
    ssm_mixer: str = "mamba2"

    def to_dict(self):
        return asdict(self)


@dataclass
class HybridGatedDeltaNetConfig:
    """
    dataclass for Gated DeltaNet (GDN). This allows us to specify parameters specific to
    GDN in the hybridization config as:

    gated_deltanet:
        use_gate: Whether to use output gating (FusedRMSNormGated).
        use_short_conv: Whether to apply short convolutions to q/k/v projections.
        allow_neg_eigval: Allow negative eigenvalues by scaling beta by 2.
        conv_size: Kernel size for short convolutions.
        conv_bias: Whether to use bias in short convolutions.
        norm_eps: Epsilon value for RMSNorm layers.
    """

    use_gate: bool = True
    use_short_conv: bool = True
    allow_neg_eigval: bool = False
    conv_size: int = 4
    conv_bias: bool = False
    norm_eps: float = 1.0e-5
    kv_proj_rank: int = None
    kv_learnable_residual: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class HybridGKAConfig:
    """
    dataclass for GKA. This allows us to specify parameters specific to
    GKA in the hybridization config as:

    gka:
        use_alpha_connection: Whether to use alpha connection for residual paths.
        use_v_conv: Whether to apply convolution to value tensors.
        use_forgetting_gate: Whether to use forgetting gate mechanism.
        gla_rescale: Whether to apply GLA-style rescaling.
        solver_type: Type of solver for regression problem (e.g., "chebyshev").
        bp_lambda: Whether to backpropagate through lambda parameters.
        num_iter: Number of iterations for iterative solvers.
        ridge_strength: Regularization strength for ridge regression.
        use_gate: Whether to use output gating.
        conv_size: Size of the convolution kernel.
        norm_eps: Epsilon value for RMSNorm layers.
        use_forgetting_gate_kk: Whether to use forgetting gate for key-key interactions.
        use_beta_gate: Whether to use beta gating mechanism.
        chunk_size: Triton block size for chunked computation.
    """

    use_alpha_connection: bool = True
    use_v_conv: bool = True
    use_forgetting_gate: bool = True
    gla_rescale: bool = True
    solver_type: str = "chebyshev"
    bp_lambda: bool = True
    num_iter: int = 30
    ridge_strength: float = 0.02
    use_gate: bool = True
    conv_size: int = 4
    norm_eps: float = 1.0e-6
    use_forgetting_gate_kk: bool = True
    use_beta_gate: bool = True
    chunk_size: int = 64
    kv_proj_rank: int = None
    kv_learnable_residual: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class HybridConfig:
    """
    dataclass for hybrid models. All hybrid model configs must at least specify:
        - base_model_name_or_path: The path to the base Transformer model.
        - hybrid_override_pattern: A string, such as "*-M2-M2-M2-*-..." which specifies
            the layers to replace in the base model. "-" separates decoder layers; the following
            are supported layers:
            - "*": Attention
            - "*DA": Dual path Attention. This is an Attention layer with two inputs streams,
                and Attention stream and an SSM stream. Each input gets processed through the
                same Attention parameters and returns two corresponding outputs. Works for any 
                of our supported SSM layers.
            - "M2": Mamba2
            - "GDN": Gated DeltaNet.
            - "GKA": Gated KalmaNet
            - "BMF": B'MOJO-F
            - "<M2/GDN/GKA/BMF>*": E.g., 'M2*'. A coupled <SSM> + Attention layer used for 
                stage 1 (layerwise distillation). This has both a <SSM> and an Attention layer.
                It takes as input a <SSM> and Attention input from the previous layer and process
                each through their respective layers.
        - output_dir: The folder in which the hybrid model will be saved.
        - dtype: The torch dtype to use for the hybrid model. Also used as the target dtype
            when dequantizing FP8 models.
        - copy_auxiliary_files: Whether to copy auxiliary files (e.g., system prompts, tokenizer
            files) from the base model directory to the output directory. Defaults to False.
    """

    base_model_name_or_path: str
    hybrid_override_pattern: str
    output_dir: str
    dtype: Optional[torch.dtype] = torch.bfloat16
    copy_auxiliary_files: bool = False
    swa: HybridSWAConfig = None
    bmojo: HybridBMOJOConfig = None
    mamba2: HybridMamba2Config = None
    gka: HybridGKAConfig = None
    gdn: HybridGatedDeltaNetConfig = None
