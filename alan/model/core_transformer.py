"""
ALAN v4 — Core Transformer Architecture
ECONX GROUP (PTY) LTD

Modular transformer where specialized attention heads handle different
cognitive functions: reasoning, creativity, curiosity, emotional intelligence,
memory retrieval, and meta-reasoning.

CRITICAL: Nothing is hardcoded. All behavior emerges from training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================
# DEVICE AUTO-DETECTION
# ============================================================

def get_device() -> torch.device:
    """
    Auto-detect the best available compute device.
    Priority: CUDA GPU > Apple MPS > CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"[ALAN] CUDA detected: {gpu_count} GPU(s) | {gpu_name} | {total_mem:.1f} GB")
        return device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("[ALAN] Apple MPS detected — using Metal GPU acceleration")
        return torch.device("mps")
    else:
        logger.info("[ALAN] No GPU detected — running on CPU")
        return torch.device("cpu")


def print_device_info():
    """Print full device information at startup."""
    print("\n" + "="*50)
    print("  ALAN v4 — Device Detection Report")
    print("="*50)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    print(f"  CUDA version    : {torch.version.cuda}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}           : {props.name}")
            print(f"  GPU {i} Memory    : {props.total_memory / 1e9:.2f} GB")
            print(f"  GPU {i} Compute   : {props.major}.{props.minor}")
    else:
        print("  GPU             : None (CPU mode)")
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"  Apple MPS       : {mps}")
    device = get_device()
    print(f"  Active device   : {device}")
    print("="*50 + "\n")
    return device


# ============================================================
# ALAN CONFIGURATION
# ============================================================

@dataclass
class AlanConfig:
    """Complete configuration for ALAN v4 architecture."""
    # Core dimensions
    vocab_size: int = 50257
    max_seq_len: int = 8192
    hidden_dim: int = 2048
    num_layers: int = 32
    num_heads: int = 32
    intermediate_dim: int = 8192
    dropout: float = 0.1
    tie_embeddings: bool = True

    # Specialized cognitive module heads (must sum to <= num_heads)
    num_reasoning_heads: int = 8
    num_creativity_heads: int = 6
    num_curiosity_heads: int = 4
    num_ei_heads: int = 4         # emotional intelligence
    num_memory_heads: int = 6
    num_meta_heads: int = 4

    # Scratchpad (internal chain-of-thought)
    scratchpad_max_tokens: int = 1024
    scratchpad_start_id: int = 50258

    # Vision (image understanding)
    vision_enabled: bool = True
    patch_size: int = 16
    image_size: int = 224
    vision_hidden_dim: int = 768

    # Context tracker
    recency_decay: float = 0.7
    reference_boost: float = 1.5
    topic_threshold: float = 0.2

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, \
            "hidden_dim must be divisible by num_heads"
        total_module_heads = (
            self.num_reasoning_heads + self.num_creativity_heads +
            self.num_curiosity_heads + self.num_ei_heads +
            self.num_memory_heads + self.num_meta_heads
        )
        assert total_module_heads <= self.num_heads, \
            f"Module heads ({total_module_heads}) exceed total heads ({self.num_heads})"


# ============================================================
# ROTARY POSITIONAL ENCODING (RoPE)
# ============================================================

class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — better long-range attention
    than absolute positional encodings.
    """
    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :])

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        cos = self.cos_cache[:, :, :seq_len, :].to(q.device)
        sin = self.sin_cache[:, :, :seq_len, :].to(q.device)
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


# ============================================================
# MODULAR MULTI-HEAD ATTENTION
# ============================================================

class ModularMultiHeadAttention(nn.Module):
    """
    Multi-head attention where different heads are assigned to specific
    cognitive modules. During training, module-specific losses encourage
    specialization. All behaviors are LEARNED, not hardcoded.
    """

    def __init__(self, config: AlanConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.hidden_dim = config.hidden_dim

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # Rotary positional encoding
        self.rope = RotaryPositionalEncoding(self.head_dim, config.max_seq_len)

        # Module-specific temperature scaling (learned)
        self.temperature_scales = nn.Parameter(torch.ones(config.num_heads))

        # Head-to-module assignment
        self.head_roles = self._assign_heads(config)

    def _assign_heads(self, config: AlanConfig) -> Dict[str, list]:
        """Map attention heads to cognitive modules."""
        roles = {}
        idx = 0
        for module, count in [
            ("reasoning",  config.num_reasoning_heads),
            ("creativity", config.num_creativity_heads),
            ("curiosity",  config.num_curiosity_heads),
            ("emotional",  config.num_ei_heads),
            ("memory",     config.num_memory_heads),
            ("meta",       config.num_meta_heads),
        ]:
            roles[module] = list(range(idx, idx + count))
            idx += count
        # Remaining heads are general-purpose
        roles["general"] = list(range(idx, config.num_heads))
        return roles

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        module_weights: Optional[Dict[str, torch.Tensor]] = None,
        context_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        Q, K = self.rope(Q, K)

        # Scale with module-specific learned temperatures
        scale = (self.head_dim ** -0.5) * self.temperature_scales.view(1, -1, 1, 1)
        attn = (Q @ K.transpose(-2, -1)) * scale

        # Causal mask
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        # Apply context bias (Attention-to-Context: recency weighting)
        if context_bias is not None:
            attn = attn + context_bias.unsqueeze(1)

        # Apply module activation weights
        if module_weights is not None:
            for module, heads in self.head_roles.items():
                if module in module_weights:
                    weight = module_weights[module].view(-1, 1, 1, 1)
                    attn[:, heads] = attn[:, heads] * weight

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ============================================================
# FEED-FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):
    """Expanded feed-forward network with GELU activation (more neurons)."""

    def __init__(self, config: AlanConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# MODULAR TRANSFORMER BLOCK
# ============================================================

class ModularTransformerBlock(nn.Module):
    """
    Single transformer block with modular attention heads.
    Pre-norm architecture for training stability.
    """

    def __init__(self, config: AlanConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = ModularMultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        module_weights: Optional[Dict] = None,
        context_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        h = self.norm1(x)
        h = self.attention(h, mask=mask, module_weights=module_weights, context_bias=context_bias)
        x = x + self.dropout(h)

        # Pre-norm feed-forward
        h = self.norm2(x)
        h = self.ff(h)
        x = x + h

        return x


# ============================================================
# TASK ROUTER (Module Activation)
# ============================================================

class TaskRouter(nn.Module):
    """
    Classifies the input and determines which cognitive modules to activate.
    Outputs activation weights (0-1) for each module.
    All routing behavior is LEARNED from training data.
    """

    def __init__(self, config: AlanConfig):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 6),   # 6 cognitive modules
            nn.Sigmoid()         # Each module gets 0-1 activation
        )
        self.module_names = ["reasoning", "creativity", "curiosity", "emotional", "memory", "meta"]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Use mean of first 8 tokens as task summary
        summary = x[:, :min(8, x.shape[1])].mean(dim=1)
        activations = self.classifier(summary)
        return {
            name: activations[:, i]
            for i, name in enumerate(self.module_names)
        }


# ============================================================
# VISION ENCODER (Image Understanding)
# ============================================================

class VisionEncoder(nn.Module):
    """
    Patch-based vision encoder that converts images into token embeddings
    compatible with the main transformer. Enables ALAN to understand images.
    """

    def __init__(self, config: AlanConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        num_patches = (config.image_size // config.patch_size) ** 2
        patch_dim = 3 * config.patch_size * config.patch_size  # RGB patches

        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, config.vision_hidden_dim),
            nn.LayerNorm(config.vision_hidden_dim),
            nn.GELU(),
            nn.Linear(config.vision_hidden_dim, config.vision_hidden_dim),
        )

        # Learnable position embeddings for patches
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, config.vision_hidden_dim) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.vision_hidden_dim) * 0.02)

        # Project vision features to main hidden_dim
        self.projection = nn.Linear(config.vision_hidden_dim, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)

        # Simple transformer for vision processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.vision_hidden_dim,
            nhead=8,
            dim_feedforward=config.vision_hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.vision_transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to patch sequences."""
        B, C, H, W = images.shape
        p = self.patch_size
        assert H == W == self.image_size, f"Expected {self.image_size}x{self.image_size}, got {H}x{W}"
        patches = images.reshape(B, C, H // p, p, W // p, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.reshape(B, -1, C * p * p)
        return patches

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) normalized image tensors
        Returns:
            vision_tokens: (B, num_patches+1, hidden_dim) — ready to prepend to text tokens
        """
        B = images.shape[0]
        patches = self.patchify(images)
        patch_emb = self.patch_embedding(patches)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patch_emb], dim=1)
        x = x + self.pos_embedding

        # Vision transformer encoding
        x = self.vision_transformer(x)

        # Project to main model dimension
        x = self.projection(x)
        x = self.norm(x)
        return x


# ============================================================
# EXTERNAL MEMORY INTERFACE
# ============================================================

class ExternalMemoryInterface(nn.Module):
    """
    Vector-based external memory for long-term pattern storage.
    Allows ALAN to retrieve relevant past patterns during inference.
    """

    def __init__(self, config: AlanConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.query_proj = nn.Linear(config.hidden_dim, 512)
        self.value_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.gate = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)

        # In-context memory bank (for session-level memory)
        self.memory_bank = None

    def store(self, key_vectors: torch.Tensor):
        """Store new patterns in memory."""
        self.memory_bank = key_vectors.detach()

    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> Optional[torch.Tensor]:
        """Retrieve top-k most relevant memories."""
        if self.memory_bank is None:
            return None
        q = self.query_proj(query.mean(dim=1))  # (B, 512)
        keys = self.query_proj(self.memory_bank)  # (M, 512)
        scores = torch.matmul(q, keys.T)  # (B, M)
        top_scores, top_idx = scores.topk(min(top_k, scores.shape[1]), dim=-1)
        retrieved = self.memory_bank[top_idx]  # (B, k, hidden_dim)
        return retrieved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate retrieved memories with current representation."""
        retrieved = self.retrieve(x)
        if retrieved is None:
            return x
        # Attend over retrieved memories
        mem_summary = retrieved.mean(dim=1)  # (B, hidden_dim)
        gate_input = torch.cat([x.mean(dim=1), mem_summary], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input)).unsqueeze(1)
        memory_contribution = self.value_proj(mem_summary).unsqueeze(1)
        x = x + gate * memory_contribution
        return self.norm(x)


# ============================================================
# ALAN v4 — COMPLETE MODEL
# ============================================================

class Alan(nn.Module):
    """
    ALAN v4 — The complete modular transformer language model.
    
    Architecture:
    - Modular multi-head attention (cognitive specialization)
    - Rotary positional encoding (RoPE)
    - Task router (module activation)
    - Vision encoder (image understanding)
    - External memory interface
    - Context tracker integration
    - Scratchpad (internal chain-of-thought)
    
    CRITICAL: Nothing is hardcoded. All behavior emerges from training.
    """

    def __init__(self, config: AlanConfig):
        super().__init__()
        self.config = config

        # Extended vocab size: base + 100 special tokens (scratchpad, system, etc.)
        self.extended_vocab_size = config.vocab_size + 100

        # Token embedding (extended)
        self.token_embedding = nn.Embedding(self.extended_vocab_size, config.hidden_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            ModularTransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        # Language model head (matches extended vocab)
        self.lm_head = nn.Linear(config.hidden_dim, self.extended_vocab_size, bias=False)

        # Tie embeddings (weight sharing)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Cognitive module router
        self.router = TaskRouter(config)

        # External memory
        self.memory = ExternalMemoryInterface(config)

        # Vision encoder
        if config.vision_enabled:
            self.vision_encoder = VisionEncoder(config)
        else:
            self.vision_encoder = None

        # Initialize weights
        self.apply(self._init_weights)
        logger.info(f"[ALAN] Model initialized: {self.count_parameters():,} parameters")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask (lower triangular)."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        tokens: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        memory_context: Optional[torch.Tensor] = None,
        context_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through ALAN v4.
        
        Args:
            tokens: (B, T) input token IDs
            images: (B, 3, H, W) optional image inputs
            memory_context: optional pre-retrieved memory embeddings
            context_bias: (B, T, T) attention bias from context tracker
        
        Returns:
            logits: (B, T, vocab_size) next-token prediction logits
            metadata: dict with module_weights, routing info
        """
        B, T = tokens.shape
        device = tokens.device

        # Token embeddings
        x = self.token_embedding(tokens)

        # Prepend vision tokens if image provided
        if images is not None and self.vision_encoder is not None:
            vision_tokens = self.vision_encoder(images)  # (B, num_patches+1, hidden_dim)
            x = torch.cat([vision_tokens, x], dim=1)
            T = x.shape[1]

        # Prepend memory context if provided
        if memory_context is not None:
            x = torch.cat([memory_context, x], dim=1)
            T = x.shape[1]

        # Causal attention mask
        mask = self.make_causal_mask(T, device)

        # Determine module activation weights via task router
        module_weights = self.router(x)

        # Process through transformer layers
        for layer in self.layers:
            x = layer(
                x,
                mask=mask,
                module_weights=module_weights,
                context_bias=context_bias,
            )

        # Integrate external memory
        x = self.memory(x)

        # Final normalization
        x = self.final_norm(x)

        # Language model head — only over original token positions
        if images is not None and self.vision_encoder is not None:
            num_vision_tokens = (self.config.image_size // self.config.patch_size) ** 2 + 1
            x_text = x[:, num_vision_tokens:, :]
        else:
            x_text = x

        logits = self.lm_head(x_text)

        metadata = {
            "module_weights": {k: v.detach() for k, v in module_weights.items()},
            "num_parameters": self.count_parameters(),
        }

        return logits, metadata

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        images: Optional[torch.Tensor] = None,
        context_bias: Optional[torch.Tensor] = None,
        stop_tokens: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with nucleus sampling.
        Temperature, top_p, and top_k are all dynamically adjustable.
        """
        self.eval()
        generated = tokens.clone()

        for _ in range(max_new_tokens):
            # Truncate to max context window
            ctx = generated[:, -self.config.max_seq_len:]

            logits, _ = self.forward(ctx, images=images, context_bias=context_bias)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_idx_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_idx_to_remove] = float("-inf")
                next_logits = torch.zeros_like(next_logits).scatter_(
                    1, sorted_idx, sorted_logits
                )

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop on stop tokens
            if stop_tokens and next_token.item() in stop_tokens:
                break

        return generated


# ============================================================
# FACTORY FUNCTION
# ============================================================

def build_alan(
    size: str = "medium",
    device: Optional[torch.device] = None,
    vision: bool = True,
    vocab_size: Optional[int] = None,
) -> Tuple["Alan", AlanConfig]:
    """
    Build ALAN v4 with preset sizes.
    
    Sizes:
        "small"  — 125M params  (fast, for testing)
        "medium" — 350M params  (balanced)
        "large"  — 1.3B params  (full spec)
    """
    if device is None:
        device = get_device()

    size_configs = {
        "small": dict(
            hidden_dim=512, num_layers=6, num_heads=8, intermediate_dim=2048,
            num_reasoning_heads=2, num_creativity_heads=1, num_curiosity_heads=1,
            num_ei_heads=1, num_memory_heads=2, num_meta_heads=1,
        ),
        "medium": dict(
            hidden_dim=1024, num_layers=16, num_heads=16, intermediate_dim=4096,
            num_reasoning_heads=4, num_creativity_heads=3, num_curiosity_heads=2,
            num_ei_heads=2, num_memory_heads=3, num_meta_heads=2,
        ),
        "large": dict(
            hidden_dim=2048, num_layers=32, num_heads=32, intermediate_dim=8192,
            num_reasoning_heads=8, num_creativity_heads=6, num_curiosity_heads=4,
            num_ei_heads=4, num_memory_heads=6, num_meta_heads=4,
        ),
    }

    cfg_overrides = size_configs.get(size, size_configs["medium"])
    config_kwargs = dict(vision_enabled=vision, **cfg_overrides)
    if vocab_size is not None:
        config_kwargs["vocab_size"] = vocab_size
    config = AlanConfig(**config_kwargs)
    model = Alan(config).to(device)

    param_count = model.count_parameters()
    print(f"[ALAN] Built '{size}' model: {param_count:,} parameters on {device}")
    return model, config


if __name__ == "__main__":
    # Quick architecture test
    logging.basicConfig(level=logging.INFO)
    device = print_device_info()

    print("Building ALAN v4 (small) for architecture test...")
    model, config = build_alan(size="small", device=device, vision=True)

    # Test forward pass
    B, T = 2, 32
    tokens = torch.randint(0, config.vocab_size, (B, T)).to(device)
    images = torch.randn(B, 3, config.image_size, config.image_size).to(device)

    logits, meta = model(tokens, images=images)
    print(f"Input tokens : {tokens.shape}")
    print(f"Input images : {images.shape}")
    print(f"Output logits: {logits.shape}")
    mw = {k: round(v.mean().item(), 3) for k, v in meta['module_weights'].items()}
    print(f"Module weights: {mw}")
    print(f"Parameters   : {meta['num_parameters']:,}")
    print("\nALAN v4 architecture test PASSED!")
