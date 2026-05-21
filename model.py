import torch
import torch.nn as nn
import math

from logging_config import get_logger

logger = get_logger(__name__)


class Inputs(nn.Module):
  def __init__(self, dModel: int, vocabSize: int): #d_model = 512
    super().__init__()
    if dModel <= 0:
      raise ValueError(f"dModel must be positive, got {dModel}")
    if vocabSize <= 0:
      raise ValueError(f"vocabSize must be positive, got {vocabSize}")
    self.dModel = dModel
    self.vocabSize = vocabSize
    self.embedding = nn.Embedding(vocabSize, dModel)

  def forward(self, x):
    return self.embedding(x) * math.sqrt(self.dModel)

class PositionalEncoding(nn.Module):

  def __init__(self, dModel: int, sequenceLength: int, dropout: float) -> None:
    super().__init__()
    if sequenceLength <= 0:
      raise ValueError(f"sequenceLength must be positive, got {sequenceLength}")
    self.dModel = dModel
    self.sequenceLength = sequenceLength
    self.dropout = nn.Dropout(dropout)

    # Matrix of shape (sequenceLength, dModel)
    pe = torch.zeros(sequenceLength, dModel)
    # Vector of shape
    pos = torch.arange(0, sequenceLength, dtype=torch.float).unsqueeze(1) # Position will be referred to as 'pos'
    divTerm = torch.exp(torch.arange(0, dModel, 2).float() * (-math.log(10000.0) / dModel)) #PE(pos, 2i) = sin(pos/10000^2i/dModel) (3.5 Positional Encoding)
    pe[:, 0::2] = torch.sin(pos * divTerm)
    pe[:, 1::2] = torch.cos(pos * divTerm)
    pe = pe.unsqueeze(0)

    self.register_buffer('pe', pe)


  def forward(self, x):
    if x.shape[1] > self.sequenceLength:
      raise ValueError(
        f"Input sequence length ({x.shape[1]}) exceeds max "
        f"positional encoding length ({self.sequenceLength}). "
        f"Truncate input or increase seqLength in config."
      )
    x = x + (self.pe[:, :x.shape[1]]).requires_grad_(False)
    return self.dropout(x)

class LayerNormalization(nn.Module):

  def __init__(self, eps: float = 10**-6) -> None:
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1)) # Multiplyed
    self.bias = nn.Parameter(torch.zeros(1)) # Added

  def forward(self, x):
    mean = x.mean(dim = -1, keepdim=True)
    std = x.std(dim = -1, keepdim=True)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module): # max(0,xW1 + b1)W2 + b2
  def __init__(self, dModel: int, dFF: int, dropout: float) -> None:
    super().__init__()
    self.linearOne = nn.Linear(dModel, dFF) # (xW1 & B1)
    self.dropout = nn.Dropout(dropout)
    self.linearTwo = nn.Linear(dFF, dModel) # (W2 & B2)

  def forward(self, x): # (Batch, sequenceLength, dModel) --> (Batch, sequenceLength, dFF) --> (Batch, sequenceLength, dModel)
    return self.linearTwo(self.dropout(torch.relu(self.linearOne(x))))


class MultiHeadAttentionBlock(nn.Module): # 3.2.2 Multi-Head Attention (Unknown Territory...)
    
    def __init__(self, dModel: int, h: int, dropout: float) -> None:
      super().__init__()
      self.dModel = dModel
      self.h = h
      assert dModel % h == 0, "dModel not divisible by h"

      self.d_k = dModel // h
      self.w_q = nn.Linear(dModel, dModel) # WQ
      self.w_k = nn.Linear(dModel, dModel) # WK
      self.w_v = nn.Linear(dModel, dModel) # WV

      self.w_o = nn.Linear(dModel, dModel)
      self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
      d_k = query.shape[-1]
      attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

      if mask is not None:
          attention_scores.masked_fill_(mask == 0, -1e4)
      attention_scores = attention_scores.softmax(dim = -1)
      if dropout is not None:
          attention_scores = dropout(attention_scores)

          
      return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # Batch, sequenceLength, dModel --> Batch, sequenceLength, dModel
        key = self.w_k(k) # Batch, sequenceLength, dModel --> Batch, sequenceLength, dModel
        value = self.w_v(v) # Batch, sequenceLength, dModel --> Batch, sequenceLength, dModel

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # Batch, sequenceLength, dModel --> Batch, sequenceLength, dModel
        return self.w_o(x)
        

class ResidualConnection(nn.Module):
  
  def __init__(self, dropout:float) -> None:
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNormalization()

  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
  
  def __init__(self, self_attentionBlock: MultiHeadAttentionBlock, feedForward: FeedForward, dropout: float) -> None:
    super().__init__()
    self.selfAttentionBlock = self_attentionBlock
    self.feedForward = feedForward
    self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

  def forward(self, x, src_mask):
    x = self.residual_connections[0](x, lambda x: self.selfAttentionBlock(x,x,x,src_mask))
    x = self.residual_connections[1](x, self.feedForward)
    return x

class Encoder(nn.Module):

  def __init__(self, layers: nn.ModuleList) -> None:
    super().__init__()
    self.layers = layers
    self.norm = LayerNormalization()

  def forward(self, x, mask):
    for layer in self.layers:
        x = layer(x, mask)
    return self.norm(x)
    
class DecoderBlock(nn.Module):

  def __init__(self, selfAttention: MultiHeadAttentionBlock, crossAttention: MultiHeadAttentionBlock, feed_forward: FeedForward, dropout: float) -> None:
    super().__init__()
    self.selfAttention = selfAttention
    self.crossAttention = crossAttention
    self.feed_forward = feed_forward
    self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

  def forward(self, x, encoder_output, sourceMask, targetMask):
      x = self.residual_connections[0](x, lambda x: self.selfAttention(x, x, x, targetMask))
      x = self.residual_connections[1](x, lambda x: self.crossAttention(x, encoder_output, encoder_output, sourceMask))
      x = self.residual_connections[2](x, self.feed_forward)
      return x

class Decoder(nn.Module):
  def __init__(self, layer: nn.ModuleList) -> None:
    super().__init__()
    self.layers = layer
    self.norm = LayerNormalization()

  def forward(self, x, encoder_output, sourceMask, targetMask):
    for layer in self.layers:
        x = layer(x, encoder_output, sourceMask, targetMask)
    return self.norm(x)

class ProjectionLayer(nn.Module): #ADD __INIT__
  def __init__(self, dModel: int, vocabSize: int) -> None:
    super().__init__()
    self.proj = nn.Linear(dModel, vocabSize)

  def forward(self, x):
    return self.proj(x)

class TransformerBlock(nn.Module):

  def __init__(self, encoder: Encoder, decoder: Decoder, sourceEmbed: Inputs, sourcePosition: PositionalEncoding, targetEmbed: Inputs, targetPosition: PositionalEncoding, projectLayer: ProjectionLayer) -> None:
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.sourceEmbed = sourceEmbed
    self.sourcePosition = sourcePosition
    self.targetEmbed = targetEmbed
    self.targetPosition = targetPosition
    self.projectLayer = projectLayer
  def encode(self, source, sourceMask):
    source = self.sourceEmbed(source)
    source = self.sourcePosition(source)
    return self.encoder(source, sourceMask)

  def decode(self, encoderOut, sourceMask, target, targetMask):
    target = self.targetEmbed(target)
    target = self.targetPosition(target)
    return self.decoder(target, encoderOut, sourceMask, targetMask)

  def projection(self, x):
    return self.projectLayer(x)

def buildTransformer(source_vocabSize: int, target_vocabSize: int, source_sequenceLength: int, target_sequenceLength: int, N: int = 6, dModel: int = 512, dFF: int = 2048, h: int = 8, dropout: float = 0.1) -> TransformerBlock:
    """
    Constructs a Transformer model with the given hyperparameters.

    Args:
        source_vocabSize: Size of the source vocabulary.
        target_vocabSize: Size of the target vocabulary.
        source_sequenceLength: Maximum source sequence length.
        target_sequenceLength: Maximum target sequence length.
        N: Number of encoder/decoder layers.
        dModel: Model embedding dimension.
        dFF: Feed-forward hidden dimension.
        h: Number of attention heads.
        dropout: Dropout probability.

    Returns:
        TransformerBlock: The constructed model.

    Raises:
        ValueError: If hyperparameters are invalid.
    """
    # ── Validate hyperparameters ──────────────────────────────────────────
    if dModel % h != 0:
        raise ValueError(
            f"dModel ({dModel}) must be divisible by h ({h}). "
            f"Got remainder {dModel % h}."
        )
    if source_vocabSize <= 0 or target_vocabSize <= 0:
        raise ValueError(
            f"Vocab sizes must be positive. Got source={source_vocabSize}, "
            f"target={target_vocabSize}."
        )

    logger.info(
        f"Building Transformer: N={N}, dModel={dModel}, dFF={dFF}, "
        f"h={h}, dropout={dropout}, src_vocab={source_vocabSize}, "
        f"tgt_vocab={target_vocabSize}, src_seq={source_sequenceLength}, "
        f"tgt_seq={target_sequenceLength}"
    )

    # Create the source and target embeddings
    sourceEmbed = Inputs(dModel, source_vocabSize)
    targetEmbed = Inputs(dModel, target_vocabSize)
    # Create the positional encodings
    sourcePosition = PositionalEncoding(dModel, source_sequenceLength, dropout)
    targetPosition = PositionalEncoding(dModel, target_sequenceLength, dropout)

    encoder_blocks = []
    for _ in range(N):
      encoder_slef_attentionBlock = MultiHeadAttentionBlock(dModel, h, dropout)
      feedForwardBlock = FeedForward(dModel, dFF, dropout)
      encoder_block = EncoderBlock(encoder_slef_attentionBlock, feedForwardBlock, dropout)
      encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
      decoder_self_attentionBlock = MultiHeadAttentionBlock(dModel, h, dropout)
      decoder_cross_attentionBlock = MultiHeadAttentionBlock(dModel, h, dropout)
      decoder_feed_forwardBlock = FeedForward(dModel, dFF, dropout)
      decoder_block = DecoderBlock(decoder_self_attentionBlock, decoder_cross_attentionBlock, decoder_feed_forwardBlock, dropout)
      decoder_blocks.append(decoder_block)

    #Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    #Create the projection layer
    projectionLayer = ProjectionLayer(dModel, target_vocabSize)

    #Create the transformer block
    transformer = TransformerBlock(encoder, decoder, sourceEmbed, sourcePosition, targetEmbed, targetPosition, projectionLayer)

    # ── Weight Tying ─────────────────────────────────────────────────────
    # Share embedding weights: source embed = target embed = projection
    # Reduces param count by ~vocabSize × dModel and regularizes learning.
    # Standard in GPT-2, BERT, T5, etc.
    transformer.targetEmbed.embedding.weight = transformer.sourceEmbed.embedding.weight
    transformer.projectLayer.proj.weight = transformer.sourceEmbed.embedding.weight

    # ── Xavier Uniform Init ──────────────────────────────────────────────
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # ── GPT-2 Scaled Residual Init ───────────────────────────────────────
    # For Pre-LN Transformers, residual paths accumulate variance across
    # N layers. Scale output projections by 1/√(2N) to stabilize training.
    residualScale = 0.02 / math.sqrt(2 * N)
    for name, p in transformer.named_parameters():
        if name.endswith('w_o.weight') or name.endswith('linearTwo.weight'):
            nn.init.normal_(p, mean=0.0, std=residualScale)

    paramCount = sum(p.numel() for p in transformer.parameters())
    logger.info(f"Transformer built: {paramCount:,} parameters")

    return transformer
        
