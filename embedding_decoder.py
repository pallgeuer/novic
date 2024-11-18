# Embedding decoder model classes

# Imports
from __future__ import annotations
import math
import fractions
import dataclasses
from typing import Callable, Optional, Any
import torch
import torch.nn as nn
import utils
import embedders
import embedding_dataset

#
# Embedding decoder
#

# Embedding decoder model
class EmbeddingDecoder(nn.Module):

	@classmethod
	def get_target_config_kwargs(cls, **target_kwargs) -> dict[str, Any]:
		# Allow the model to adjust the target tokenization specification to its needs
		# The required target_kwargs are exactly the kwargs of embedders.Embedder.create_target_config(), i.e. exactly all those arguments after the * in the method signature
		raise NotImplementedError

	@classmethod
	def get_data_config_kwargs(cls, **data_kwargs) -> dict[str, Any]:
		# Allow the model to adjust the data specification to its needs
		# The possible data_kwargs are exactly the fields of embedding_dataset.DataConfig, where a missing value is equivalent to None, and a value of None means don't care (embedding dataset can decide based on what data it has on offer)
		raise NotImplementedError

	max_seq_len: int                              # Maximum sequence length that is ever provided to the transformer
	vocab_size_quant: int                         # Vocab size that has optionally been increased to the nearest multiple of 64
	embed_mlp: EmbeddingVectorMLP                 # Embedding vector MLP module
	token_embedding: Optional[nn.Embedding]       # Token embedding module (if required)
	embed_tokens: Optional[Callable[[torch.Tensor], torch.Tensor]]  # Callable that applies token embedding (could be via logits linear if weight tying)
	pos_embedding: Optional[LearnedPosEmbedding]  # Learned position embedding module
	logits_linear: nn.Linear                      # Linear logits calculation module
	unused_map: dict[nn.Parameter, int]           # Map of module parameters that contain unused elements to the corresponding integer number of unused elements

	def __init__(
		self,
		*,
		embedder: embedders.Embedder,               # Embedder to associate the model with (target_config must be already configured)
		data_config: embedding_dataset.DataConfig,  # Dataset data configuration (NOTE: The model state dict should not be directly affected by data_config, so that you can for example train in multi-target mode and evaluate in single target mode)
		vocab_quant: bool,                          # Whether to quantize the vocab size to the next highest multiple of a power of 2 for alleged computational efficiency purposes
		num_end_loss: int,                          # Number of trailing end tokens to include in the non-generation prediction loss (must be >= 1)
		label_smoothing: float,                     # Amount of label smoothing when computing the cross entropy loss (0 = Disable label smoothing)
		hidden_dim: int,                            # Transformer hidden dimension E
		feedfwd_scale: Any,                         # Any string, int, float, decimal, etc that can be converted to a clean fraction specifying how much multiplicatively larger the feedforward dimension is than the hidden dimension
		mlp_seq_len: int,                           # SPECIAL: Number of sequence elements P the embedding vector is transformed to using an MLP in order to input it to the transformer
		mlp_hidden_layer: str,                      # See hidden_layer in EmbeddingVectorMLP
		mlp_hidden_bias: bool,                      # See hidden_bias in EmbeddingVectorMLP
		mlp_hidden_norm: bool,                      # See hidden_norm in EmbeddingVectorMLP
		mlp_hidden_activation: str,                 # See hidden_activation in EmbeddingVectorMLP
		input_dropout: float,                       # Dropout probability of input signal to first transformer layer (0 = No dropout but layer(s) are still there)
		num_layers: int,                            # Number of transformer layers
		num_heads: int,                             # Number of transformer attention heads
		layer_dropout: float,                       # Dropout probability within each transformer layer (0 = No dropout but layers are still there)
		layer_activation: str,                      # Activation function to use within the transformer layers specified as a string (e.g. 'relu' or 'gelu', see utils.get_activation_gain)
		layer_norm_first: bool,                     # Whether the layer norm is done prior to the attention and feedforward operations respectively in each transformer layer
		layer_bias: bool,                           # Whether to incorporate biases throughout the entire transformer layers
		logits_bias: bool,                          # Whether to include a bias in the logits linear layer
		init_bias_zero: bool,                       # Whether to initialise all bias parameters in all parts of the model to zero
		init_mlp_mode: str,                         # Initialisation mode of the MLP (default = Default PyTorch init, balanced = Balanced init to promote unitness)
		init_mlp_unit_norm: bool,                   # Initialise the MLP and embedding to work with unit norm (True) as opposed to unit standard deviation (False)
		init_tfrm_mode: str,                        # Initialisation mode of the transformer (default = Default PyTorch init, open = Init strategy used in OpenCLIP, balanced = Balanced init to promote unitness for layer norm first case)
		init_tfrm_unit_norm: bool,                  # Initialise the transformer to work with unit norm (True) as opposed to unit standard deviation (False)
		init_tfrm_unit_postnorm: bool,              # Initialise the post-transformer norm to work with unit norm (True) as opposed to unit standard deviation (False)
		init_tfrm_proj_layers: bool,                # Whether to adjust the initial residual projection weight values for the number of layers (if not init mode default, adjustment is intended for layer norm first case)
		init_zero_norm: bool,                       # Whether to initialise all transformer layer norm weights to zero (if layer norm is first then these are on the residual paths, if layer norm is not first this is unlikely to do anything useful if True, overrides init_tfrm_unit_norm if True)
		init_rezero_mode: str,                      # Whether to apply ReZero to initially zero out the residual paths (none = Do not use ReZero, perskip = One scale parameter per skip connection, perlayer = One scale parameter per layer)
	):
		super().__init__()
		self.embedder = embedder
		self.target_config = self.embedder.target_config
		self.target_vocab = self.embedder.target_vocab
		self.data_config = data_config
		self.vocab_quant = vocab_quant
		self.num_end_loss = num_end_loss
		assert num_end_loss >= 1
		self.label_smoothing = label_smoothing
		self.embed_dtype = self.embedder.embed_dtype
		self.embed_dim = self.embedder.embed_dim
		self.hidden_dim = hidden_dim
		self.feedfwd_scale = fractions.Fraction(feedfwd_scale)
		feedfwd_dim_frac = self.hidden_dim * self.feedfwd_scale
		if feedfwd_dim_frac.denominator != 1:
			raise ValueError(f"Feedforward dimension scaler ({self.feedfwd_scale}) must result in an integral feedforward dimension when applied to hidden dimension ({self.hidden_dim})")
		self.feedfwd_dim = feedfwd_dim_frac.numerator
		self.mlp_seq_len = mlp_seq_len
		assert self.mlp_seq_len >= 1
		self.mlp_hidden_layer = mlp_hidden_layer
		self.mlp_hidden_bias = mlp_hidden_bias
		self.mlp_hidden_norm = mlp_hidden_norm
		self.mlp_hidden_activation = mlp_hidden_activation
		self.input_dropout = input_dropout
		self.num_layers = num_layers
		self.num_heads = num_heads
		self.layer_dropout = layer_dropout
		self.layer_activation = layer_activation
		self.layer_norm_first = layer_norm_first
		self.layer_bias = layer_bias
		self.logits_bias = logits_bias
		self.init_bias_zero = init_bias_zero
		self.init_mlp_mode = init_mlp_mode
		self.init_mlp_unit_norm = init_mlp_unit_norm
		self.init_tfrm_mode = init_tfrm_mode
		self.init_tfrm_unit_norm = init_tfrm_unit_norm
		self.init_tfrm_unit_postnorm = init_tfrm_unit_postnorm
		self.init_tfrm_proj_layers = init_tfrm_proj_layers
		self.init_zero_norm = init_zero_norm
		self.init_rezero_mode = init_rezero_mode

	def get_num_params(self) -> tuple[ParamCount, dict[str, ParamCount]]:
		# Return the total parameter count details for the model, as well as a dict of the parameter count details of each named part of the model (see self.count_num_params())
		raise NotImplementedError

	def forward(self, embed: torch.Tensor, target: Optional[torch.Tensor], target_padding: Optional[torch.Tensor], target_weight: Optional[torch.Tensor], calc_loss: bool, calc_correct: bool, only_pred: bool, guide_targets: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
		# embed = BxF batch of embedding vectors to decode (floating point unit vectors)
		# target = If calculating loss/correct or required for generation, BxC batch of token IDs formatted as per the target configuration (except for potential deviations in masked padding values, see target_padding)
		# target_padding = If target is given, an optional BxC batch of padding token masks indicating which token IDs should be considered to be padding and consequently ignored (boolean where True = Ignore)
		# target_weight = If target is given, an optional B batch of target weights indicating how much each target should be weighted for the loss calculation (floating point tensor of same dtype as embed, all sequence locations of targets with zero weight are considered to be padded in any case and therefore ignored)
		# calc_loss = Whether to calculate and return the model loss (if False then the third/fourth return values are None, returned loss is affected by only_pred)
		# calc_correct = Whether to calculate which tokens were correctly predicted (if False then the fifth return value is None, correct is affected by only_pred)
		# only_pred = Whether to return only the logits/padding/losses/correct for the target token IDs that are actually newly predicted by this forward pass (e.g. for autoregressive models this will only be one token at a time per sample, no tensor dimensions are squeezed)
		# guide_targets = WxCmax tensor of tokenized guide targets to restrict the correctness evaluation to (only relevant if calc_correct, incompatible with only_pred, Cmax = self.target_config.token_length, W = Number of permitted possible output target nouns)
		# Returns:
		#  - Output BxTxV logits tensor
		#  - BxT target padding tensor (None if both the input target_padding and target_weight are None, True if the token at that location is padding/zero-weighted, do NOT write to this tensor as it could be an expanded tensor)
		#  - Scalar loss sum (returned if calc_loss, loss sum across unmasked/non-padding/non-zero-weighted tokens only, affected by only_pred)
		#  - Scalar loss basis (returned if calc_loss, basis sum across unmasked/non-padding/non-zero-weighted tokens only, affected by only_pred)
		#  - BxT correct tokens tensor (returned if calc_correct, True if the token at that location is non-padding/non-zero-weighted and was correctly predicted, affected by only_pred)
		# Note: The loss sum and basis should have the property that mean batch loss = sum / basis, and that calculating the mean loss for two batches separately and combining them, i.e. (sum1 + sum2) / (basis1 + basis2), is equivalent to concatenating the batches and calculating the mean loss in one go
		# Note: The output target padding and correct tokens tensors by definition are disjoint, i.e. can never both be True for the same element
		# Note: If a target_weight is provided that has zero weighting for some samples, then meaningless output data may be generated for those samples
		# Note: B = Batch size, T = (<=C if only_pred else C), V = self.target_config.vocab_size
		# Note: If multiple targets is configured in the data configuration, then any input target/target padding/target weight and any output logits/target padding/correct tokens must replace B with BxM or MxB (depending on multi_first) in the stated required dimensions
		raise NotImplementedError

	def generate(self, embed: torch.Tensor, collect_logits: bool, calc_loss: bool, temperature: float, length_alpha: float, sample_weight: Optional[torch.Tensor], guide_targets: Optional[torch.Tensor], guide_renorm: bool) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
		# embed = BxF batch of embedding vectors to decode (floating point unit vectors)
		# collect_logits = Whether to collate and return the logits used during generation
		# calc_loss = Whether to calculate and return the model loss/score
		# temperature = If calc_loss, temperature scaler to divide the raw logits by prior to softmax (must be positive non-zero, affects scores but not loss and not which output tokens are chosen)
		# length_alpha = If calc_loss, length normalisation factor alpha (0 = Score is sum of log-probs = No length normalisation, 1 = Score is mean log-prob = Full length normalisation, <0 = Possible, affects scores but not loss and not which output tokens are chosen)
		# sample_weight = If calc_loss, an optional B batch of per-sample target weights indicating how much each sample should be weighted for the loss calculation (floating point tensor of same dtype as embed, does not affect which output tokens are chosen, does not affect score calculation as scores are per-sample)
		# guide_targets = WxCmax tensor of tokenized guide targets to restrict the output generation to (Cmax = self.target_config.token_length, W = Number of permitted possible output target nouns)
		# guide_renorm = If guide_targets is provided, whether to renormalise probabilities after the effect of guiding (does not affect the returned logits or loss, just any returned scores)
		# Returns:
		#  - Predicted token IDs tensor BxC in a format suitable for target configuration decoding
		#  - Predicted token padding tensor BxC (True = padded sequence location)
		#  - The corresponding BxCxV logits (returned if collect_logits, or if logits were required anyway for loss calculation)
		#  - Scalar loss sum (returned if calc_loss, loss sum across unmasked/non-padding tokens)
		#  - Scalar loss basis (returned if calc_loss, basis sum across unmasked/non-padding tokens)
		#  - Target sequence scores tensor B (returned if calc_loss, per-sample scores across unmasked/non-padding tokens)
		# Note: Refer to forward() for more discussion of loss sum/basis
		# Note: B = Batch size, C = However many tokens needed to be generated up to at most Cmax = self.target_config.token_length, V = self.target_config.vocab_size
		raise NotImplementedError

	def generate_beam(self, embed: torch.Tensor, topk: int, temperature: float, length_alpha: float, vocab_targets: Optional[torch.Tensor], vocab_per_token: bool, vocab_scaler: float, guide_targets: Optional[torch.Tensor], guide_renorm: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		# embed = BxF batch of embedding vectors to decode (floating point unit vectors)
		# topk = Number H of top target sequence candidates to maintain at each stage of the beam search
		# temperature = Temperature scaler to divide the raw logits by prior to softmax (must be positive non-zero)
		# length_alpha = Length normalisation factor alpha (0 = Score is sum of log-probs = No length normalisation, 1 = Score is mean log-prob = Full length normalisation, <0 = Possible)
		# vocab_targets = Optional ZxCmax tensor of tokenized vocabulary targets to use for calculating and correcting the scores for the expected prior probability of each token ID at each stage (can be same as guide_targets)
		# vocab_per_token = Whether to calculate the expected prior token probabilities on a per unique vocab target (False) or per unique vocab token (True) basis (the latter is a locally uniform distribution)
		# vocab_scaler = Dimensionless scaler that premultiplies the score adjustments made due to the expected prior probabilities (0 = Disable, 1 = Full correction by prior probability)
		# guide_targets = WxCmax tensor of tokenized guide targets to restrict the output generation to (Cmax = self.target_config.token_length, W = Number of permitted possible output target nouns)
		# guide_renorm = If guide_targets is provided, whether to renormalise probabilities after the effect of guiding
		# Returns:
		#  - Predicted token IDs tensor BxHxC in a format suitable for target configuration decoding
		#  - Predicted token padding tensor BxHxC (True = padded sequence location)
		#  - Target sequence scores tensor BxH (targets are always returned in descending score order per sample)
		raise NotImplementedError

	def precompute_generate_all(self, length_alpha: float, vocab_targets: Optional[torch.Tensor], vocab_per_token: bool, vocab_scaler: float, guide_targets: torch.Tensor, guide_renorm: bool) -> Any:
		# See generate_all() for a description of the arguments
		# Returns any entity that can be passed as the precompute argument of generate_all() to avoid needing to calculate certain fixed things in every call
		# If using the precompute argument, all other common arguments to the call to this method must match exactly
		raise NotImplementedError

	def generate_all(self, embed: torch.Tensor, topk: int, temperature: float, length_alpha: float, vocab_targets: Optional[torch.Tensor], vocab_per_token: bool, vocab_scaler: float, guide_targets: torch.Tensor, guide_renorm: bool, precompute: Optional[Any] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		# embed = BxF batch of embedding vectors to decode (floating point unit vectors)
		# topk = Number K of top decoded target nouns to return
		# temperature = Temperature scaler to divide the raw logits by prior to softmax (must be positive non-zero)
		# length_alpha = Length normalisation factor alpha (0 = Score is sum of log-probs = No length normalisation, 1 = Score is mean log-prob = Full length normalisation, <0 = Possible)
		# vocab_targets = Optional ZxCmax tensor of tokenized vocabulary targets to use for calculating and correcting the scores for the expected prior probability of each token ID at each stage (can be same as guide_targets)
		# vocab_per_token = Whether to calculate the expected prior token probabilities on a per unique vocab target (False) or per unique vocab token (True) basis (the latter is a locally uniform distribution)
		# vocab_scaler = Dimensionless scaler that premultiplies the score adjustments made due to the expected prior probabilities (0 = Disable, 1 = Full correction by prior probability)
		# guide_targets = WxCmax tensor of tokenized guide targets to restrict the output generation to (Cmax = self.target_config.token_length, W = Number of permitted possible output target nouns)
		# guide_renorm = Whether to renormalise probabilities after the effect of guiding
		# precompute = Precomputed values to use instead of calculating them again (all arguments must match exactly to what was used in the call to precompute_generate_all())
		# Returns:
		#  - Predicted token IDs tensor BxKxC in the same format as given in guide_targets (other than systematically trailing padding being reduced)
		#  - Predicted token padding tensor BxKxC (True = padded sequence location)
		#  - Target sequence scores tensor BxK (targets are always returned in descending score order per sample)
		# Note: The difference to generate_beam() is that the scores for ALL guide targets are explicitly calculated and filtered down to the best topk per sample
		raise NotImplementedError

	def create_mlp(self, output_bias: bool, dropout_prob: Optional[float]):

		if self.init_mlp_mode == 'default':
			init_mlp_balanced = None
		elif self.init_mlp_mode == 'balanced':
			init_mlp_balanced = 1 if output_bias else 1 / math.sqrt(2)  # The sum of two random vectors of approximately equal length is a vector of norm approximately sqrt(2) times that, so we account for the summing of MLP token and positional embeddings (it is assumed there are no positional embeddings coming if output bias is present)
		else:
			raise ValueError(f"Unrecognised value for MLP initialisation mode: {self.init_mlp_mode}")

		self.embed_mlp = EmbeddingVectorMLP(
			embed_dim=self.embed_dim,
			output_dim=self.hidden_dim,
			output_seq_len=self.mlp_seq_len,
			output_bias=output_bias,
			hidden_layer=self.mlp_hidden_layer,
			hidden_bias=self.mlp_hidden_bias,
			hidden_norm=self.mlp_hidden_norm,
			hidden_activation=self.mlp_hidden_activation,
			dropout_prob=dropout_prob,
			init_unit_norm=self.init_mlp_unit_norm,
			init_balanced=init_mlp_balanced,
			init_bias_zero=self.init_bias_zero,
			init_output_bias_zero=False,
		)

	def create_embedding(self, max_seq_len: int, pos_embed: bool, token_inputs: bool, weight_tying: bool):
		# max_seq_len = Maximum sequence length that is ever provided to the transformer
		# pos_embed = Whether a positional embedding is required
		# token_inputs = Whether tokens are provided as inputs to the transformer, and hence whether an input token embedding is required
		# weight_tying = If token_inputs, whether to tie the input token embeddings to the linear logits output weights

		self.max_seq_len = max_seq_len
		self.vocab_size_quant = math.ceil(self.target_config.vocab_size / 64) * 64 if self.vocab_quant else self.target_config.vocab_size  # Q = Quantized vocab size, V = True (potentially smaller) vocab size
		init_embed_std = 1 / math.sqrt(2 * self.hidden_dim) if self.init_mlp_unit_norm else 1 / math.sqrt(2)  # Standard deviation and 2-norm are directly linked under the assumption of zero mean

		logits_linear_class = utils.LinearEmbed if token_inputs and weight_tying else nn.Linear
		self.logits_linear = logits_linear_class(in_features=self.hidden_dim, out_features=self.vocab_size_quant, bias=self.logits_bias)
		nn.init.normal_(self.logits_linear.weight, mean=0.0, std=init_embed_std)  # Note: This cannot have a special factor of 1/sqrt(2) due to weight tying linking the weights to the token embedding
		if self.logits_linear.bias is None or self.init_bias_zero:
			if self.logits_linear.bias is not None:
				nn.init.zeros_(self.logits_linear.bias)
		else:
			nn.init.normal_(self.logits_linear.bias, mean=0.0, std=init_embed_std if self.init_tfrm_unit_postnorm else init_embed_std * math.sqrt(self.hidden_dim))

		if token_inputs:
			if weight_tying:
				self.token_embedding = None
				self.embed_tokens = self.logits_linear.embed
			else:
				self.token_embedding = nn.Embedding(num_embeddings=self.vocab_size_quant, embedding_dim=self.hidden_dim)
				self.embed_tokens = self.token_embedding
				nn.init.normal_(self.token_embedding.weight, mean=0.0, std=init_embed_std)
		else:
			self.token_embedding = None
			self.embed_tokens = None

		if pos_embed:
			self.pos_embedding = LearnedPosEmbedding(max_seq_len=self.max_seq_len, pos_embed_dim=self.hidden_dim, dropout_prob=self.input_dropout)
			nn.init.normal_(self.pos_embedding.embedding.weight, mean=0.0, std=init_embed_std)
		else:
			self.pos_embedding = None

		self.unused_map = {}
		if self.vocab_size_quant > self.target_config.vocab_size:
			params_with_unused = [self.logits_linear.weight]
			if self.logits_linear.bias is not None:
				params_with_unused.append(self.logits_linear.bias)
			if self.token_embedding is not None:
				params_with_unused.append(self.token_embedding.weight)
			for param_with_unused in params_with_unused:
				unused_params = param_with_unused[self.target_config.vocab_size:]
				nn.init.zeros_(unused_params)
				self.unused_map[param_with_unused] = unused_params.numel()
			self.verify_unused()
			self.register_state_dict_pre_hook(self.__class__.verify_unused)
			self.register_load_state_dict_post_hook(self.__class__.verify_unused)

	def create_transformer(self, num_layers: int, self_attn_dim: int, cross_attn_dim: int = 0, init_unit_postnorm: Optional[bool] = None, transformer_layer_kwargs: Optional[dict[str, Any]] = None, transformer_kwargs: Optional[dict[str, Any]] = None) -> nn.Module:
		# num_layers = Number of transformer layers
		# self_attn_dim = Nominal assumed dimension of self-attention (minimum number of contributing attention locations for any sequence location that predicts a token, e.g. 1 if the very first sequence location causally predicts a token)
		# cross_attn_dim = Nominal assumed dimension of cross-attention (minimum number of contributing memory attention locations for any sequence location that predicts a token, e.g. memory sequence length if full cross-attention, or 0 if no cross-attention at all)
		# init_unit_postnorm = Initialise the post-transformer norm to work with unit norm (True) as opposed to unit standard deviation (False), or use the default value (None)
		# transformer_layer_kwargs = Additional keyword arguments to provide to the transformer layer constructor
		# transformer_kwargs = Additional keyword arguments to provide to the transformer constructor
		# Return the created and initialised transformer

		if init_unit_postnorm is None:
			init_unit_postnorm = self.init_tfrm_unit_postnorm
		if transformer_layer_kwargs is None:
			transformer_layer_kwargs = {}
		if transformer_kwargs is None:
			transformer_kwargs = {}

		use_custom_impl = False
		if self.init_rezero_mode != 'none':
			use_custom_impl = True
			transformer_layer_kwargs.update(rezero=self.init_rezero_mode)

		if use_decoder := (cross_attn_dim > 0):
			transformer_layer_class = TransformerDecoderLayer if use_custom_impl else nn.TransformerDecoderLayer
			transformer_class = nn.TransformerDecoder
		else:
			transformer_layer_class = TransformerEncoderLayer if use_custom_impl else nn.TransformerEncoderLayer
			transformer_class = nn.TransformerEncoder

		activation, activation_gain = utils.get_activation_gain(name=self.layer_activation, functional=True, unit_std=not (self.init_tfrm_unit_norm or self.init_zero_norm))
		transformer_layer = transformer_layer_class(
			d_model=self.hidden_dim,
			nhead=self.num_heads,
			dim_feedforward=self.feedfwd_dim,
			dropout=self.layer_dropout,
			activation=activation,
			batch_first=True,
			norm_first=self.layer_norm_first,
			bias=self.layer_bias,
			**transformer_layer_kwargs,
		)
		if isinstance(transformer_layer, nn.TransformerEncoderLayer) and not self.layer_bias:
			transformer_layer.activation_relu_or_gelu = 0  # Note: Very dirty hack that SHOULD have the ONLY effect of disabling the sparsity fast path, to try to deal with (until it is fixed): https://github.com/pytorch/pytorch/issues/116546
		transformer = transformer_class(
			transformer_layer,
			num_layers=num_layers,
			norm=nn.LayerNorm(normalized_shape=self.hidden_dim, bias=self.layer_bias) if self.layer_norm_first else None,
			**transformer_kwargs,
		)

		factor = 1 / math.sqrt(self.hidden_dim)
		num_layers_factor = 1 / math.sqrt((3 if use_decoder else 2) * transformer.num_layers)
		nominal_std = factor if self.init_tfrm_unit_norm else 1
		nominal_residual_std = nominal_std * num_layers_factor if self.init_tfrm_proj_layers else nominal_std
		init_norm_scale = 0 if self.init_zero_norm else nominal_std
		init_postnorm_scale = factor if init_unit_postnorm else 1

		if self.init_tfrm_mode == 'default':
			init_std_sa_in_proj = None
			init_std_sa_out_proj = None
			init_std_mha_in_proj = None
			init_std_mha_out_proj = None
			init_std_ff1 = None
			init_std_ff2 = None
		else:
			if self.init_tfrm_mode == 'open':
				init_std_sa_in_proj = factor
				init_std_sa_out_proj = factor
				init_std_mha_in_proj = factor  # OpenCLIP doesn't have cross-attention so we take the same values as self-attention
				init_std_mha_out_proj = factor
				init_std_ff1 = factor / math.sqrt(2)
				init_std_ff2 = factor
			elif self.init_tfrm_mode == 'balanced':
				attn_scale = lambda attn_dim: math.sqrt((1 + (nominal_std ** 4) * (attn_dim - 1) / attn_dim) / attn_dim)  # noqa: The attention scale is an estimate of the scale factor that happens due to multihead attention if there are attn_dim active attention locations at a sequence location
				init_std_sa_in_proj = factor
				init_std_sa_out_proj = factor / attn_scale(max(self_attn_dim, 1))
				init_std_mha_in_proj = factor
				init_std_mha_out_proj = factor / attn_scale(max(cross_attn_dim, 1))
				init_std_ff1 = factor
				init_std_ff2 = 1 / (math.sqrt(self.feedfwd_dim) * activation_gain)
			else:
				raise ValueError(f"Unrecognised value for transformer initialisation mode: {self.init_tfrm_mode}")
			if self.init_tfrm_proj_layers:
				init_std_sa_out_proj *= num_layers_factor
				init_std_mha_out_proj *= num_layers_factor
				init_std_ff2 *= num_layers_factor

		for name, module in transformer.named_modules():

			if isinstance(module, nn.LayerNorm):
				nn.init.constant_(module.weight, val=init_norm_scale)
				if module.bias is not None:
					nn.init.zeros_(module.bias)
			else:

				if name.endswith('.self_attn'):
					weight, bias, weight_std, output_std = module.in_proj_weight, module.in_proj_bias, init_std_sa_in_proj, nominal_std
				elif name.endswith('.self_attn.out_proj'):
					weight, bias, weight_std, output_std = module.weight, module.bias, init_std_sa_out_proj, nominal_residual_std
				elif name.endswith('.multihead_attn'):
					weight, bias, weight_std, output_std = module.in_proj_weight, module.in_proj_bias, init_std_mha_in_proj, nominal_std
				elif name.endswith('.multihead_attn.out_proj'):
					weight, bias, weight_std, output_std = module.weight, module.bias, init_std_mha_out_proj, nominal_residual_std
				elif name.endswith('.linear1'):
					weight, bias, weight_std, output_std = module.weight, module.bias, init_std_ff1, nominal_std
				elif name.endswith('.linear2'):
					weight, bias, weight_std, output_std = module.weight, module.bias, init_std_ff2, nominal_residual_std
				else:
					params = dict(module.named_parameters(recurse=False))
					if isinstance(module, (TransformerEncoderLayer, TransformerDecoderLayer)):
						params = {n: p for n, p in params.items() if not (n[:-1] == 'scale' and n[-1] in ('1', '2', '3'))}
					if params:
						raise ValueError(f"Module with unexpected parameters: {module}")
					continue

				if weight_std is None or output_std is None:
					if bias is not None and self.init_bias_zero:
						nn.init.zeros_(bias)
				else:
					if bias is None or self.init_bias_zero:
						nn.init.normal_(weight, mean=0.0, std=weight_std)
						if bias is not None:
							nn.init.zeros_(bias)
					else:
						nn.init.normal_(weight, mean=0.0, std=weight_std / math.sqrt(2))
						nn.init.normal_(bias, mean=0.0, std=output_std / math.sqrt(2))

		postnorm_weight = transformer.layers[-1].norm2.weight if transformer.norm is None else transformer.norm.weight
		nn.init.constant_(postnorm_weight, val=init_postnorm_scale)  # Note: Bias has already been zeroed above

		return transformer

	def count_num_params(self, extra_groups: Optional[dict[str, set[Optional[nn.Module]]]] = None, ignore_param: Optional[set[nn.Parameter]] = None) -> tuple[ParamCount, dict[str, ParamCount]]:
		# extra_groups = Extra parameter groups that will get merged/appended to 'param_groups' (see its definition below)
		# ignore_param = Parameters to ignore from the count
		# Return the total parameter count details for the model, as well as a dict of the parameter count details of each named part of the model (the required create_*() methods need to have been called prior to this)

		param_groups: dict[str, set[Optional[nn.Module]]] = {
			'Input MLP': {self.embed_mlp},
			'Token embed/logits': {self.token_embedding, self.logits_linear},  # Note: If token embedding module is None it just gets ignored in the parameter counts
		}

		if self.pos_embedding:
			param_groups['Positional embed'] = {self.pos_embedding}

		if extra_groups:
			for group, extra_modules in extra_groups.items():
				if (modules := param_groups.get(group, None)) is None:
					param_groups[group] = extra_modules
				else:
					modules.update(extra_modules)

		count_kwargs = dict(ignore_param=ignore_param, unused_map=self.unused_map)
		param_counts = {group: ParamCount.from_modules(*modules, **count_kwargs) for group, modules in param_groups.items()}
		total_param_count = ParamCount.from_modules(self, **count_kwargs)
		assert total_param_count == ParamCount.from_counts(*param_counts.values())
		return total_param_count, param_counts

	# noinspection PyUnusedLocal
	def verify_unused(self, *args, **kwargs):
		for param_with_unused in self.unused_map:
			if torch.any(param_with_unused[self.target_config.vocab_size:] != 0):
				raise ValueError("Unexpected values in the unused portion of a parameter tensor")

	def print_params(self):
		for name, param in self.named_parameters(prefix='decoder'):
			if param in self.unused_map:
				param = param[:self.target_config.vocab_size]
			utils.show(param, prefix=name)

#
# Dud decoder
#

# Dud decoder model that just cheats its way to success where possible (useful for evaluation of metric upper bounds)
class DudDecoder(EmbeddingDecoder):

	@classmethod
	def get_target_config_kwargs(cls, **target_kwargs) -> dict[str, Any]:
		return target_kwargs

	@classmethod
	def get_data_config_kwargs(cls, **data_kwargs) -> dict[str, Any]:
		return data_kwargs

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.dud_target, self.dud_target_padding = self.embedder.tokenize_target('unknown')
		if torch.any(self.dud_target < 0):
			self.dud_target, self.dud_target_padding = self.embedder.tokenize_target('')
		assert self.dud_target.shape == self.dud_target_padding.shape and self.dud_target.shape[0] == 1 and self.dud_target.shape[1] >= 1 and not self.dud_target_padding.any()

	def get_num_params(self) -> tuple[ParamCount, dict[str, ParamCount]]:
		zero_param_count = ParamCount(total=0, used=0, unused=0, trained=0, frozen=0)
		return zero_param_count, {'Dud': zero_param_count}

	def forward(self, embed: torch.Tensor, target: Optional[torch.Tensor], target_padding: Optional[torch.Tensor], target_weight: Optional[torch.Tensor], calc_loss: bool, calc_correct: bool, only_pred: bool, guide_targets: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
		# Note: embed is BxF, target is ZxC, target_padding is Optional[ZxC], target_weight is Optional[Z], where Z is B or BxM or MxB
		# Note: Any supplied guide target nouns are ignored

		if target is None:
			raise ValueError(f"{self.__class__.__name__} can only cheat, so it requires targets that it can cheat from")
		*Z, C = target.shape
		V = self.target_config.vocab_size

		if target_weight is not None:
			if target_padding is None:
				target_padding = target_weight.logical_not().unsqueeze(dim=-1).expand_as(target)  # Z --> Zx1 --> ZxC
			else:
				target_padding = target_padding.logical_or(target_weight.logical_not().unsqueeze(dim=-1))  # noqa / Z --> Z --> Zx1 / (ZxC, Zx1) --> ZxC

		if target_padding is not None:
			if self.num_end_loss > 1:  # N > 1 (alternative is N = 1 as we know N >= 1)
				padding_keep = C - self.num_end_loss + 1  # C-N+1 <= C-1
				if padding_keep <= 1:
					target_padding = target_padding[..., 0:1].expand_as(target)  # Zx1 --> ZxC
				else:
					target_padding = torch.concat(tensors=(target_padding[..., 0:1].expand(*Z, self.num_end_loss - 1), target_padding[..., :padding_keep]), dim=-1)  # (Zx(N-1), Zx(C-N+1)) --> ZxC

		target_pred = target.clone()  # ZxC
		if len(Z) > 1:  # If multi-target, i.e. Z is BxM or MxB
			if self.data_config.multi_first:
				M, B = Z
				for m in range(M - 1):
					R = M - m  # R = M - m >= 2
					target_slice = target[m:, :, :]  # MxBxC --> RxBxC
					target_pred_slice = target_pred[m:, :, :]  # MxBxC --> RxBxC
					target_equiv = torch.concat((target.new_ones(size=(R, B, 1), dtype=torch.bool), torch.eq(target_slice[:1, :, :-1], target_slice[:, :, :-1]).cummin(dim=-1)[0]), dim=-1)  # (RxBx1, RxBx(C-1)) --> RxBxC
					if target_padding is not None:
						target_equiv.logical_and_(target_padding[m:, :, :].logical_not())  # noqa / (RxBxC, RxBxC) --> RxBxC
					if target_weight is None:
						target_priority = target_pred.new_zeros(size=(V + 1, B, C), dtype=target.dtype).scatter_(dim=0, index=target_pred_slice.masked_fill(mask=target_equiv.logical_not(), value=V), value=1, reduce='add')  # (RxBxC, RxBxC) --> RxBxC / ((V+1)xBxC, RxBxC) --> (V+1)xBxC
					else:
						target_priority = target_pred.new_zeros(size=(V + 1, B, C), dtype=target_weight.dtype).scatter_add_(dim=0, index=target_pred_slice.masked_fill(mask=target_equiv.logical_not(), value=V), src=target_weight[m:, :, None].expand(-1, -1, C))  # (RxBxC, RxBxC) --> RxBxC / MxB --> RxBxC / ((V+1)xBxC, RxBxC, RxBxC) --> (V+1)xBxC
					target_pred_slice[target_equiv] = target_priority[:-1, :, :].argmax(dim=0, keepdim=True).expand(R, -1, -1)[target_equiv]  # (V+1)xBxC --> VxBxC --> 1xBxC --> RxBxC / (RxBxC, RxBxC) --> RxBxC
			else:
				B, M = Z
				for m in range(M - 1):
					R = M - m  # R = M - m >= 2
					target_slice = target[:, m:, :]  # BxMxC --> BxRxC
					target_pred_slice = target_pred[:, m:, :]  # BxMxC --> BxRxC
					target_equiv = torch.concat((target.new_ones(size=(B, R, 1), dtype=torch.bool), torch.eq(target_slice[:, :1, :-1], target_slice[:, :, :-1]).cummin(dim=-1)[0]), dim=-1)  # (BxRx1, BxRx(C-1)) --> BxRxC
					if target_padding is not None:
						target_equiv.logical_and_(target_padding[:, m:, :].logical_not())  # noqa / (BxRxC, BxRxC) --> BxRxC
					if target_weight is None:
						target_priority = target_pred.new_zeros(size=(B, V + 1, C), dtype=target.dtype).scatter_(dim=1, index=target_pred_slice.masked_fill(mask=target_equiv.logical_not(), value=V), value=1, reduce='add')  # (BxRxC, BxRxC) --> BxRxC / (Bx(V+1)xC, BxRxC) --> Bx(V+1)xC
					else:
						target_priority = target_pred.new_zeros(size=(B, V + 1, C), dtype=target_weight.dtype).scatter_add_(dim=1, index=target_pred_slice.masked_fill(mask=target_equiv.logical_not(), value=V), src=target_weight[:, m:, None].expand(-1, -1, C))  # (BxRxC, BxRxC) --> BxRxC / BxM --> BxRxC / (Bx(V+1)xC, BxRxC, BxRxC) --> Bx(V+1)xC
					target_pred_slice[target_equiv] = target_priority[:, :-1, :].argmax(dim=1, keepdim=True).expand(-1, R, -1)[target_equiv]  # Bx(V+1)xC --> BxVxC --> Bx1xC --> BxRxC / (BxRxC, BxRxC) --> BxRxC

		x = embed.new_zeros(size=(*target_pred.shape, V)).scatter_(dim=-1, index=target_pred.unsqueeze(dim=-1), value=1.0)  # noqa / ZxC --> ZxCx1 --> ZxCxV

		if only_pred:  # T = 1 (else T = C)
			target_pred = target_pred[..., -1:]  # ZxC --> ZxT
			x = x[..., -1:, :]  # ZxCxV --> ZxTxV
			target = target[..., -1:]  # ZxC --> ZxT
			if target_padding is not None:
				target_padding = target_padding[..., -1:]  # ZxC --> ZxT

		loss_sum = torch.tensor(1.0, dtype=embed.dtype, device=embed.device) if calc_loss else None
		loss_basis = torch.tensor(1.0, dtype=embed.dtype, device=embed.device) if calc_loss else None

		if calc_correct:
			correct = torch.eq(target_pred, target)  # noqa / (ZxT, ZxT) --> ZxT
			if target_padding is not None:
				correct.logical_and_(target_padding.logical_not())  # noqa / (ZxT, ZxT) --> ZxT
		else:
			correct = None

		return x, target_padding, loss_sum, loss_basis, correct

	def generate(self, embed: torch.Tensor, collect_logits: bool, calc_loss: bool, temperature: float, length_alpha: float, sample_weight: Optional[torch.Tensor], guide_targets: Optional[torch.Tensor], guide_renorm: bool) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
		# Note: Any supplied guide target nouns are ignored

		B = embed.shape[0]
		C = self.dud_target.shape[1]  # C >= 1

		target = self.dud_target.to(device=embed.device).expand(B, -1).contiguous()
		target_padding = self.dud_target_padding.to(device=embed.device).expand(B, -1).contiguous()
		seq_logits = embed.new_zeros(size=(B, C, self.target_config.vocab_size)).scatter_(dim=-1, index=target.unsqueeze(dim=-1), value=1.0) if collect_logits or calc_loss else None  # noqa

		if calc_loss:

			score_logits = seq_logits.div(temperature)
			target_score = torch.nn.functional.log_softmax(score_logits, dim=2)
			target_score = target_score.gather(dim=2, index=target.unsqueeze(dim=2)).squeeze(dim=2)
			target_score = target_score.sum(dim=1)
			if length_alpha != 0:
				target_score.mul_(math.pow(C, -length_alpha))

			loss_sum = nn.functional.cross_entropy(input=seq_logits.view(-1, seq_logits.shape[-1]), target=target.view(-1), ignore_index=-1, reduction='sum', label_smoothing=self.label_smoothing)
			loss_basis = torch.tensor(target.numel(), dtype=embed.dtype, device=embed.device)

		else:
			loss_sum = loss_basis = target_score = None

		return target, target_padding, seq_logits, loss_sum, loss_basis, target_score

	def generate_beam(self, embed: torch.Tensor, topk: int, temperature: float, length_alpha: float, vocab_targets: Optional[torch.Tensor], vocab_per_token: bool, vocab_scaler: float, guide_targets: Optional[torch.Tensor], guide_renorm: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		# Note: Any supplied guide target nouns are ignored, and the output beam contains only one valid dud result

		B = embed.shape[0]
		H = topk
		C = self.dud_target.shape[1]

		target = torch.zeros(size=(B, H, C), dtype=self.dud_target.dtype, device=embed.device)
		target_padding = torch.ones(size=(B, H, C), dtype=self.dud_target_padding.dtype, device=embed.device)
		target_score = embed.new_full(size=(B, H), fill_value=-torch.inf)
		target[:, 0, :] = self.dud_target.to(device=embed.device)
		target_padding[:, 0, :] = self.dud_target_padding.to(device=embed.device)
		target_score[:, 0] = -1.0

		return target, target_padding, target_score

	def precompute_generate_all(self, length_alpha: float, vocab_targets: Optional[torch.Tensor], vocab_per_token: bool, vocab_scaler: float, guide_targets: torch.Tensor, guide_renorm: bool) -> Any:
		return None

	def generate_all(self, embed: torch.Tensor, topk: int, temperature: float, length_alpha: float, vocab_targets: Optional[torch.Tensor], vocab_per_token: bool, vocab_scaler: float, guide_targets: torch.Tensor, guide_renorm: bool, precompute: Optional[Any] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		# Note: The supplied guide target nouns are ignored, and the output contains only one valid dud result

		B = embed.shape[0]
		H = topk
		Cw = guide_targets.shape[1]

		target = torch.zeros(size=(B, H, Cw), dtype=self.dud_target.dtype, device=embed.device)
		target_padding = torch.ones(size=(B, H, Cw), dtype=self.dud_target_padding.dtype, device=embed.device)
		target_score = embed.new_full(size=(B, H), fill_value=-torch.inf)
		target[:, 0, :self.dud_target.shape[1]] = self.dud_target.to(device=embed.device)
		target_padding[:, 0, :self.dud_target_padding.shape[1]] = self.dud_target_padding.to(device=embed.device)
		target_score[:, 0] = -1.0

		return target, target_padding, target_score

#
# Prefixed iterative decoder
#

# Prefixed iterative embedding decoder model
class PrefixedIterDecoder(EmbeddingDecoder):

	@classmethod
	def get_target_config_kwargs(cls, **target_kwargs) -> dict[str, Any]:
		# Note: Based on this setup we know an end token is present, end token = pad token = 0, and from token 1 onwards is other (non-start) actually-used tokens
		target_kwargs.update(
			with_start_token=False,
			with_end_token=True,
			compact_ids=True,
		)
		return target_kwargs

	@classmethod
	def get_data_config_kwargs(cls, **data_kwargs) -> dict[str, Any]:
		return data_kwargs

	def __init__(
		self,
		mlp_seq_len: int,       # SPECIAL: Number of sequence elements P the embedding vector is transformed to using an MLP in order to input it to the transformer as a prefix (must be >=1)
		weight_tying: bool,     # Whether to apply weight tying between the token embedding and logits linear layer
		strictly_causal: bool,  # Whether to make the transformer strictly causal, meaning that even the prefix tokens are processed causally
		enable_nested: bool,    # Whether to enable the use of nested tensors for the forward pass of the transformer (related to BetterTransformer, fast sparsity path, transformer padding mask)
		**kwargs,               # Keyword arguments to pass to base class constructor
	):

		super().__init__(mlp_seq_len=mlp_seq_len, **kwargs)
		self.weight_tying = weight_tying
		self.strictly_causal = strictly_causal
		self.enable_nested = enable_nested

		self.create_mlp(output_bias=False, dropout_prob=None)
		self.create_embedding(max_seq_len=self.mlp_seq_len + self.target_config.token_length - 1, pos_embed=True, token_inputs=True, weight_tying=self.weight_tying)  # Max sequence length = P + Cmax - 1 (minus 1 due to presence of end token, as we do not need to predict the next token after an end token)
		self.transformer = self.create_transformer(num_layers=self.num_layers, self_attn_dim=self.mlp_seq_len, transformer_kwargs=dict(enable_nested_tensor=self.enable_nested))

		causality_mask = nn.Transformer.generate_square_subsequent_mask(sz=self.max_seq_len, dtype=self.embed_dtype)
		if not self.strictly_causal:
			causality_mask[:self.mlp_seq_len, :self.mlp_seq_len].zero_()
		self.register_buffer('causality_mask', causality_mask)

	def get_num_params(self) -> tuple[ParamCount, dict[str, ParamCount]]:
		return self.count_num_params(extra_groups={'Transformer': {self.transformer}})

	def forward(self, embed: torch.Tensor, target: Optional[torch.Tensor], target_padding: Optional[torch.Tensor], target_weight: Optional[torch.Tensor], calc_loss: bool, calc_correct: bool, only_pred: bool, guide_targets: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

		assert embed.ndim == 2 and embed.dtype == self.embed_dtype  # Embeddings tensor must be of the expected dtype and a suitable shape
		x = self.embed_mlp(embed)  # BxF --> BxPxE

		B = M = None
		if target is not None and target.ndim == 3:  # A = BM = MB (otherwise A = B so x is AxPxE, target is AxC, target_padding is AxC, target_weight is A)
			multi_first = self.data_config.multi_first if self.data_config.multi_target else False  # Note: If passing multi-target data despite multi-target not being configured, then it is treated as B-before-M data
			if multi_first:  # A = MB
				M, B = target.shape[:2]
				if M > 1:
					x = x.repeat(M, 1, 1)  # BxPxE --> AxPxE
			else:  # A = BM
				B, M = target.shape[:2]
				if M > 1:
					x = x.repeat_interleave(repeats=M, dim=0)  # BxPxE --> AxPxE
			target = target.view(-1, target.shape[-1])  # BxMxC --> BMxC = AxC / MxBxC --> MBxC = AxC
			if target_padding is not None:
				target_padding = target_padding.view(-1, target_padding.shape[-1])  # BxMxC --> BMxC = AxC / MxBxC --> MBxC = AxC
			if target_weight is not None:
				target_weight = target_weight.view(-1)  # BxM --> BM = A / MxB --> MB = A

		if target is not None and target_weight is not None:
			if target_padding is None:
				target_padding = target_weight.logical_not().unsqueeze(dim=1).expand_as(target)  # A --> A --> Ax1 --> AxC
			else:
				target_padding = target_padding.logical_or(target_weight.logical_not().unsqueeze(dim=1))  # noqa / A --> A --> Ax1, (AxC, Ax1) --> AxC

		assert target is None or (target.dtype == self.target_config.token_dtype and target.ndim == 2 and target.shape[0] == x.shape[0] and target.shape[1] >= 1)  # Must be of expected dtype and contain at least a single end token (even if inferencing due to padding and that location being required as the output target)
		assert target_padding is None or (target is not None and target_padding.dtype == self.target_config.mask_dtype and target_padding.shape == target.shape)  # Target padding should only be given if target is given, and if so, both must be the same shape
		assert target_weight is None or (target is not None and target_weight.dtype == self.embed_dtype and target_weight.ndim == 1 and target_weight.shape[0] == target.shape[0])  # Target weight should only be given if target is given, and if so, must be 1D of the appropriate batch size

		if target is not None and target.shape[1] > 1:
			x = torch.concat(tensors=(x, self.embed_tokens(target[:, :-1])), dim=1)  # AxC --> Ax(C-1) --> Ax(C-1)xE / (AxPxE, Ax(C-1)xE) --> Ax(P+C-1)xE = AxSxE for S = P+C-1
		x = self.pos_embedding(x)  # AxSxE --> AxSxE
		S = x.shape[1]  # Sequence length S

		if target_padding is None:
			seq_padding_mask = None
		else:
			A, C = target.shape
			padding_expand = self.mlp_seq_len + self.num_end_loss - 2  # P+N-2
			padding_keep = C - self.num_end_loss + 1  # C-N+1 (S = P+C-1 = padding_expand + padding_keep)
			if padding_expand < 1:  # Equivalent to padding_keep >= S / P >= 1 and N >= 1 so P+N-2 < 1 implies P = N = 1
				seq_padding_mask_bool = target_padding  # AxC = Ax(P+C-1) = AxS as P = 1
			else:  # Equivalent to padding_expand >= 1 and padding_keep <= S-1
				if padding_keep <= 1:  # Equivalent to padding_expand >= S-1
					seq_padding_mask_bool = target_padding[:, 0:1].expand(-1, S)  # AxS
				else:
					seq_padding_mask_bool = torch.concat(tensors=(target_padding[:, 0:1].expand(-1, padding_expand), target_padding[:, :padding_keep]), dim=1)  # (Ax(P+N-2), Ax(C-N+1)) --> Ax(P+C-1) = AxS
				target_padding = seq_padding_mask_bool[:, -C:]  # AxS --> AxC
			seq_padding_mask = torch.zeros_like(seq_padding_mask_bool, dtype=self.embed_dtype)  # AxS
			if S > 1:
				seq_padding_mask[:, 1:].masked_fill_(mask=seq_padding_mask_bool[:, 1:], value=-torch.inf)  # bool(False, True) --> float(0, -inf) while ensuring the first sequence location is never masked with -inf values (otherwise transformer produces NaNs, note that target padding is unaffected for loss calculation)

		x = self.transformer(src=x, mask=self.causality_mask[:S, :S], src_key_padding_mask=seq_padding_mask)  # AxSxE --> AxSxE / Note: Neither mask nor src_key_padding_mask may completely mask a sequence location from receiving attention as otherwise NaNs occur and later operations cannot deal with that in the backward pass (e.g. cross_entropy amongst many other things)

		if only_pred:  # T = 1
			x = x[:, -1:, :]  # AxSxE --> AxTxE
			if target is not None:
				target = target[:, -1:]  # AxC --> AxT
				if target_padding is not None:
					target_padding = target_padding[:, -1:]  # AxC --> AxT
		else:  # T = C
			x = x[:, (self.mlp_seq_len - 1):, :]  # AxSxE --> AxTxE (T = C)

		x = self.logits_linear(x)  # AxTxE --> AxTxQ
		if self.vocab_quant:
			x = x[:, :, :self.target_config.vocab_size]  # AxTxQ --> AxTxV

		loss_sum = loss_basis = correct = None
		if calc_loss or calc_correct:  # Note: This if-scope assumes target is not None and will error at some point if not

			if target_padding is not None:
				assert target.shape == target_padding.shape
				target = target.masked_fill(mask=target_padding, value=-1)  # (AxT, AxT) --> AxT

			if calc_loss:
				if target_weight is None:
					loss_sum = nn.functional.cross_entropy(input=x.view(-1, x.shape[-1]), target=target.view(-1), ignore_index=-1, reduction='sum', label_smoothing=self.label_smoothing)  # (AxTxV, AxT) --> 1
					if target_padding is None:
						loss_basis = torch.tensor(target.numel(), device=target.device)  # 1
					else:
						loss_basis = target_padding.numel() - target_padding.sum()  # AxT --> 1
				else:
					loss_sum = nn.functional.cross_entropy(input=x.view(-1, x.shape[-1]), target=target.view(-1), ignore_index=-1, reduction='none', label_smoothing=self.label_smoothing).view(target.shape)  # (AxTxV, AxT) --> AT --> AxT
					loss_sum = target_weight.dot(loss_sum.sum(dim=1))  # AxT --> A / (A, A) --> 1
					if target_padding is None:
						loss_basis = target.shape[1] * target_weight.sum()  # AxT --> 1
					else:
						loss_basis = target_weight.dot((target_padding.shape[1] - target_padding.sum(dim=1)).to(dtype=target_weight.dtype))  # AxT --> A / (A, A) --> 1

			if calc_correct:
				if guide_targets is None:
					pred_tokens = x.argmax(dim=2)  # AxTxV --> AxT
				else:
					assert not only_pred  # Guide targets is only supported for T = C
					A, C, V = x.shape
					W = guide_targets.shape[0]
					guide_targets_T = guide_targets.T  # Cmax x W
					guide_mask = torch.concat(tensors=(torch.zeros(size=(A, 1, W), dtype=torch.bool, device=target.device), torch.ne(target[:, :(C - 1), None], guide_targets_T[None, :(C - 1), :]).cummax(dim=1)[0]), dim=1)  # (Ax(C-1)x1, 1x(C-1)xW) --> Ax(C-1)xW --> Ax(C-1)xW / (Ax1xW, Ax(C-1)xW) --> AxCxW
					pred_tokens = x.new_full(size=(A, C, V + 1), fill_value=-torch.inf).scatter_(dim=2, index=guide_targets_T[None, :C, :].expand(A, -1, -1).masked_fill(mask=guide_mask, value=V), value=0)[:, :, :-1].add_(x).argmax(dim=2)  # 1xCxW --> AxCxW / (AxCxW, AxCxW) --> AxCxW / (AxCx(V+1), AxCxW) --> AxCx(V+1) --> AxCxV / (AxCxV, AxCxV) --> AxCxV --> AxC = AxT
				correct = torch.eq(pred_tokens, target)  # (AxT, AxT) --> AxT / Note: For all masked sequence locations target is -1 (see above) and so correct is False (argmax always returns non-negative indices)

		if M is not None:
			if multi_first:  # noqa
				x = x.view(M, B, x.shape[1], x.shape[2])  # AxTxV --> MxBxTxV
				if target_padding is not None:
					target_padding = target_padding.view(M, B, target_padding.shape[1])  # AxT --> MxBxT
				if correct is not None:
					correct = correct.view(M, B, correct.shape[1])  # AxT --> MxBxT
			else:
				x = x.view(B, M, x.shape[1], x.shape[2])  # AxTxV --> BxMxTxV
				if target_padding is not None:
					target_padding = target_padding.view(B, M, target_padding.shape[1])  # AxT --> BxMxT
				if correct is not None:
					correct = correct.view(B, M, correct.shape[1])  # AxT --> BxMxT

		return x, target_padding, loss_sum, loss_basis, correct

	def generate(self, embed: torch.Tensor, collect_logits: bool, calc_loss: bool, temperature: float, length_alpha: float, sample_weight: Optional[torch.Tensor], guide_targets: Optional[torch.Tensor], guide_renorm: bool) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

		B = embed.shape[0]  # B = Batch size
		G = self.target_config.token_length - 1  # G = Cmax-1 / Token length includes a trailing end token which we do not need to explicitly generate (hence minus 1)
		V = self.target_config.vocab_size  # V = Number of unique available token IDs

		target = torch.zeros(size=(B, G), dtype=self.target_config.token_dtype, device=embed.device)  # Full BxG batch of end tokens = 0
		target_padding = torch.zeros(size=(B, G), dtype=self.target_config.mask_dtype, device=embed.device)  # Full BxG batch of token paddings, initially all False
		sample_mask = torch.zeros(size=(B, 1), dtype=self.target_config.mask_dtype, device=embed.device)  # Bx1 boolean sample mask of initially all False values (0 in bool dtype) indicating whether each sample has finished generating
		guide_mask = torch.zeros(size=(B, 1, guide_targets.shape[0]), dtype=torch.bool, device=embed.device) if guide_targets is not None else None  # Bx1xW (guide_targets is WxCmax)

		seq_logits = [] if collect_logits or calc_loss else None
		guide_scores = [] if calc_loss and guide_targets is not None and guide_renorm else None
		for C in range(1, G + 1):
			Cm = C - 1

			target_slice = target[:, :C]  # BxC batch of what has been generated so far (initially nothing) plus a column of trailing end tokens (C = 1 initially)
			if C > 1:  # If C = 1 then we would be overwriting a column of False with a column of False (see initialisation)
				target_padding[:, Cm:C] = sample_mask  # Update Bx1 slice of BxG tensor
			pred_logits = self(embed=embed, target=target_slice, target_padding=sample_mask.expand(-1, C), target_weight=None, calc_loss=False, calc_correct=False, only_pred=True, guide_targets=None)[0]  # The Bx1xV output logits predict which token IDs should replace the last column of pure end tokens that is present in target_slice (logits are meaningless data for masked sequence locations, providing target_weight=sample_weight here would be pointless as calc_loss=False and would actually unintentionally affect which tokens are generated due to padding considerations)
			if seq_logits is not None:
				seq_logits.append(pred_logits)  # list[Bx1xV]

			if guide_targets is None:
				if C <= 1:
					pred_token_ids = pred_logits[:, :, 1:].argmax(dim=2) + 1  # Bx1 / Disallows the very first token from being an end token
				else:
					pred_token_ids = pred_logits.argmax(dim=2)  # Bx1 / Arbitrary meaningless token IDs are predicted for masked samples and these should be ignored later
			else:
				guide_targets_slice = guide_targets[:, Cm]  # WxCmax --> W
				guide_score = pred_logits.new_full(size=(B, 1, V + 1), fill_value=-torch.inf).scatter_(dim=2, index=guide_targets_slice.expand(B, 1, -1).masked_fill(mask=guide_mask, value=V), value=0)[:, :, :-1]  # W --> Bx1xW / (Bx1x(V+1), Bx1xW) --> Bx1x(V+1) --> Bx1xV
				if guide_scores is not None:
					guide_scores.append(guide_score)  # list[Bx1xV]
				pred_token_ids = guide_score.add(pred_logits).argmax(dim=2)  # (Bx1xV, Bx1xV) --> Bx1xV --> Bx1
				guide_mask.logical_or_(torch.ne(pred_token_ids, guide_targets_slice.unsqueeze(dim=0)).unsqueeze(dim=1))  # noqa / W --> 1xW / (Bx1, 1xW) --> BxW --> Bx1xW / (Bx1xW, Bx1xW) --> Bx1xW

			target[:, Cm:C] = pred_token_ids  # Replace the Bx1 column of trailing end tokens with the Bx1 predicted token IDs in the BxG target tensor
			sample_mask.logical_or_(pred_token_ids.logical_not_())  # noqa / Bx1 --> Bx1 / End tokens are ID 0, so the logical not is equivalent to 'predicted token is end token', which we update the sample mask with using OR
			if sample_mask.all():  # GPU-CPU synchronization point / If all samples are masked (have predicted an end token at some point) then none have anything further to generate and we are done
				target = target_slice  # BxT where T is the current value of C / Note that this does not necessarily contain a trailing end token for the longest sample generations, but this is okay for detokenization
				target_padding = target_padding[:, :C]  # BxT where T is the current value of C
				break  # If the loop exits naturally then T = G

		if seq_logits is not None:
			seq_logits = torch.concat(seq_logits, dim=1)  # list[Bx1xV] --> BxTxV
		target.masked_fill_(mask=target_padding, value=0)  # (BxT, BxT) --> BxT

		if calc_loss:

			score_logits = seq_logits.div(temperature)  # BxTxV --> BxTxV
			if guide_scores is not None:
				score_logits.add_(torch.concat(guide_scores, dim=1))  # list[Bx1xV] --> BxTxV / (BxTxV, BxTxV) --> BxTxV
			target_score = torch.nn.functional.log_softmax(score_logits, dim=2)  # BxTxV --> BxTxV
			target_score = target_score.gather(dim=2, index=target.unsqueeze(dim=2)).squeeze(dim=2)  # BxT --> BxTx1 / (BxTxV, BxTx1) --> BxTx1 --> BxT
			target_score.masked_fill_(mask=target_padding, value=0)  # (BxT, BxT) --> BxT
			target_score = target_score.sum(dim=1)  # BxT --> B
			if length_alpha != 0:
				target_score.mul_((target_padding.shape[1] - target_padding.sum(dim=1, dtype=target_score.dtype)).clamp_(min=1).pow_(exponent=-length_alpha))  # BxT --> B --> B --> B / (B, B) --> B

			loss_target = target.masked_fill(mask=target_padding, value=-1)  # (BxT, BxT) --> BxT
			if sample_weight is None:
				loss_sum = nn.functional.cross_entropy(input=seq_logits.view(-1, seq_logits.shape[-1]), target=loss_target.view(-1), ignore_index=-1, reduction='sum', label_smoothing=self.label_smoothing)  # (BxTxV, BxT) --> 1
				loss_basis = target_padding.numel() - target_padding.sum()  # BxT --> 1
			else:
				loss_sum = nn.functional.cross_entropy(input=seq_logits.view(-1, seq_logits.shape[-1]), target=loss_target.view(-1), ignore_index=-1, reduction='none', label_smoothing=self.label_smoothing).view(loss_target.shape)  # (BxTxV, BxT) --> BT --> BxT
				loss_sum = sample_weight.dot(loss_sum.sum(dim=1))  # BxT --> B / (B, B) --> 1
				loss_basis = sample_weight.dot((target_padding.shape[1] - target_padding.sum(dim=1)).to(dtype=sample_weight.dtype))  # BxT --> B / (B, B) --> 1

		else:
			loss_sum = loss_basis = target_score = None

		return target, target_padding, seq_logits, loss_sum, loss_basis, target_score

	def generate_beam(self, embed: torch.Tensor, topk: int, temperature: float, length_alpha: float, vocab_targets: Optional[torch.Tensor], vocab_per_token: bool, vocab_scaler: float, guide_targets: Optional[torch.Tensor], guide_renorm: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		# Note: This implementation is incompatible with setting both multi_target=True and multi_first=True (can train like that, but eval with multi_first=False)

		B = embed.shape[0]  # B = Batch size
		H = topk            # H = Beam width
		G = self.target_config.token_length - 1  # G = Cmax-1 / Token length includes a trailing end token which we do not need to explicitly generate (hence minus 1)
		V = self.target_config.vocab_size        # V = Number of unique available token IDs

		target = torch.zeros(size=(B, H, G), dtype=self.target_config.token_dtype, device=embed.device)  # BxHxG batch of initially all end tokens = 0
		target_padding = torch.ones(size=(B, H, G), dtype=self.target_config.mask_dtype, device=embed.device)  # BxHxG batch of token paddings (0 = False = Not padding, 1 = True = Padding)
		target_padding[:, 0, 0].zero_()  # Every sample starts off with a beam that contains only a single non-padded empty candidate along with the required trailing end token
		target_score = embed.new_full(size=(B, H), fill_value=-torch.inf)  # BxH batch of current candidate scores
		target_score[:, 0].zero_()  # Non-padded empty start candidate starts with a score of 0 (all other totally padded candidates have a score of -inf to avoid being chosen)
		target_score_3d = target_score.unsqueeze(dim=2)  # BxH --> BxHx1

		top_indices = target.new_empty(size=(B, H))  # Preallocated BxH tensor to hold the top indices in each iteration
		top_candidates = target.new_empty(size=(B, H))  # Preallocated BxH tensor to hold the top candidate indices in each iteration
		top_candidates_3d = top_candidates.unsqueeze(dim=2)  # BxH --> BxHx1
		top_token_ids = target.new_empty(size=(B, H))  # Preallocated BxH tensor to hold the top token IDs in each iteration
		top_token_ids_3d = top_token_ids.unsqueeze(dim=2)  # BxH --> BxHx1

		if guide_targets is None:
			W = guide_mask = guide_mask_other = None
		else:  # guide_targets is WxCmax
			W = guide_targets.shape[0]
			guide_mask = torch.ones(size=(B, H, W), dtype=torch.bool, device=embed.device)  # BxHxW
			guide_mask[:, 0, :].zero_()  # Non-padded empty start candidate starts with all guide target nouns still being possible
			guide_mask_other = torch.empty_like(guide_mask)  # BxHxW

		if vocab_targets is None or vocab_scaler == 0:
			Z = vocab_is_guide = vocab_mask = vocab_mask_other = None
		else:  # vocab_targets is ZxCmax
			Z = vocab_targets.shape[0]
			vocab_is_guide = guide_targets is not None and (vocab_targets is guide_targets or torch.equal(vocab_targets, guide_targets))
			if vocab_is_guide:  # Potential GPU-CPU synchronization point
				vocab_mask = vocab_mask_other = None
			else:
				vocab_mask = torch.ones(size=(B, H, Z), dtype=torch.bool, device=embed.device)  # BxHxZ
				vocab_mask[:, 0, :].zero_()  # Non-padded empty start candidate starts with all vocab target nouns still being possible
				vocab_mask_other = torch.empty_like(vocab_mask)  # BxHxZ

		if length_alpha == 0:
			target_score_normed = target_seq_len = target_score_scale = target_score_scale_3d = scores_normed = None
			topk = (target_score, top_indices)  # (BxH, BxH)
		else:
			target_score_normed = target_score.clone()  # BxH --> BxH
			target_seq_len = embed.new_zeros(size=(B, H))  # BxH batch of current candidate sequence lengths
			target_seq_len[:, 0].fill_(1)  # Non-padded empty start candidate will have length 1 as soon as the end token is replaced by a generated token
			target_score_scale = embed.new_empty(size=(B, H))  # Preallocated BxH tensor to hold the normalising scaler for each top candidate
			target_score_scale_3d = target_score_scale.unsqueeze(dim=2)  # BxH --> BxHx1
			scores_normed = embed.new_empty(size=(B, H, V))  # Preallocated BxHxV tensor to hold the normalised scores
			topk = (target_score_normed, top_indices)  # (BxH, BxH)

		for C in range(1, G + 1):
			Cm = C - 1

			target_slice = target[:, :, :C]  # BxHxC
			target_padding_slice = target_padding[:, :, :C]  # BxHxC

			logits, logits_padding, _, _, _ = self(embed=embed, target=target_slice, target_padding=target_padding_slice, target_weight=None, calc_loss=False, calc_correct=False, only_pred=True, guide_targets=None)  # BxHx1xV logits (and associated BxHx1 boolean padding tensor) that predict which token IDs should replace the current trailing end tokens in the candidates (logits are meaningless data for padded sequence locations)
			logits = logits.div_(temperature).squeeze(dim=2)  # BxHx1xV --> BxHx1xV --> BxHxV
			logits[:, :, 1:].masked_fill_(mask=logits_padding, value=-torch.inf)  # (BxHxV, BxHx1) --> BxHxV / Forces every finished (i.e. padded) candidate sequence to predict an end token with a score of 0

			if guide_targets is not None:
				guide_indices = guide_targets[:, Cm].expand(B, H, -1).masked_fill(mask=guide_mask, value=V)  # WxCmax --> W --> BxHxW / (BxHxW, BxHxW) --> BxHxW
				guide_score = logits.new_full(size=(B, H, V + 1), fill_value=-torch.inf).scatter_(dim=2, index=guide_indices, value=0)[:, :, :-1]  # (BxHx(V+1), BxHxW) --> BxHx(V+1) --> BxHxV
				guide_score[:, :, :1].masked_fill_(mask=logits_padding, value=0)  # (BxHx1, BxHx1) --> BxHx1
				if guide_renorm:
					logits.add_(guide_score)  # (BxHxV, BxHxV) --> BxHxV

			scores = nn.functional.log_softmax(logits, dim=2)  # BxHxV --> BxHxV

			if vocab_is_guide is not None:
				vocab_indices = guide_indices if vocab_is_guide else vocab_targets[:, Cm].expand(B, H, -1).masked_fill(mask=vocab_mask, value=V)  # noqa / ZxCmax --> Z --> BxHxZ / (BxHxZ, BxHxZ) --> BxHxZ
				if vocab_per_token:
					vocab_probs = scores.new_zeros(size=(B, H, V + 1)).scatter_(dim=2, index=vocab_indices, value=1)[:, :, :-1]  # (BxHx(V+1), BxHxZ) --> BxHx(V+1) --> BxHxV
					vocab_probs.div_(vocab_probs.sum(dim=2, keepdim=True))  # BxHxV --> BxHx1 / (BxHxV, BxHx1) --> BxHxV
				else:
					vocab_probs = scores.new_zeros(size=(B, H, V + 1)).scatter_(dim=2, index=vocab_indices, value=1, reduce='add')  # (BxHx(V+1), BxHxZ) --> BxHx(V+1)
					vocab_probs_count = vocab_probs[:, :, -1:]  # BxHx(V+1) --> BxHx1
					vocab_probs = vocab_probs[:, :, :-1].div_(torch.sub(Z, vocab_probs_count, out=vocab_probs_count))  # BxHx1 --> BxHx1 / (BxHxV, BxHx1) --> BxHxV
				vocab_probs.log_()  # BxHxV --> BxHxV
				vocab_probs.nan_to_num_(nan=torch.inf, neginf=torch.inf, posinf=torch.inf)  # BxHxV --> BxHxV
				vocab_probs[:, :, :1].masked_fill_(mask=logits_padding, value=0)  # (BxHx1, BxHx1) --> BxHx1
				scores.sub_(vocab_probs, alpha=vocab_scaler)  # (BxHxV, BxHxV) --> BxHxV

			scores.add_(target_score_3d)  # (BxHxV, BxHx1) --> BxHxV / Calculates the total score each candidate (of H) would have if a certain token ID (of V) was chosen as the next token
			if C <= 1:
				scores[:, 0, 0] = -torch.inf  # Disallow the first generated token from being an end token (without affecting the probabilities)

			if guide_targets is not None and not guide_renorm:
				scores.add_(guide_score)  # noqa / (BxHxV, BxHxV) --> BxHxV

			if length_alpha == 0:
				torch.topk(scores.view(B, -1), k=H, dim=1, largest=True, sorted=True, out=topk)  # BxHxV --> BxHV --> (BxH, BxH) / Retrieves the value and indices of the H best possible new total scores
			else:
				torch.pow(target_seq_len.clamp(min=1), exponent=-length_alpha, out=target_score_scale)  # BxH --> BxH
				torch.mul(scores, target_score_scale_3d, out=scores_normed)  # (BxHxV, BxHx1) --> BxHxV
				torch.topk(scores_normed.view(B, -1), k=H, dim=1, largest=True, sorted=True, out=topk)  # BxHxV --> BxHV --> (BxH, BxH)
				torch.gather(input=scores.view(B, -1), dim=1, index=top_indices, out=target_score)  # BxHxV --> BxHV / (BxHV, BxH) --> BxH

			torch.floor_divide(top_indices, V, out=top_candidates)  # BxH --> BxH
			torch.remainder(top_indices, V, out=top_token_ids)  # BxH --> BxH

			if C > 1:
				target_slice[:, :, :-1] = torch.gather(input=target_slice[:, :, :-1], dim=1, index=top_candidates_3d.expand(-1, -1, Cm))  # BxHx1 --> BxHx(C-1) / (BxHx(C-1), BxHx(C-1)) --> BxHx(C-1)
			target_slice[:, :, -1] = top_token_ids  # BxH
			target_padding[:, :, :C] = torch.gather(input=target_padding_slice, dim=1, index=top_candidates_3d.expand(-1, -1, C))  # BxHx1 --> BxHxC / (BxHxC, BxHxC) --> BxHxC

			if C < G:

				torch.logical_not(top_token_ids, out=(sample_padding := target_padding[:, :, C])).logical_or_(target_padding[:, :, Cm])  # noqa / BxH --> BxH / (BxH, BxH) --> BxH
				if sample_padding.all():  # GPU-CPU synchronization point
					target = target_slice  # BxHxC
					target_padding = target_padding_slice  # BxHxC
					break

				if guide_targets is not None:
					torch.gather(input=guide_mask, dim=1, index=top_candidates_3d.expand(-1, -1, W), out=guide_mask_other)  # BxHx1 --> BxHxW / (BxHxW, BxHxW) --> BxHxW
					torch.ne(top_token_ids_3d, guide_targets[None, None, :, Cm], out=guide_mask).logical_or_(guide_mask_other)  # noqa / WxCmax --> 1x1xW / (BxHx1, 1x1xW) --> BxHxW / (BxHxW, BxHxW) --> BxHxW

				if vocab_mask is not None:
					torch.gather(input=vocab_mask, dim=1, index=top_candidates_3d.expand(-1, -1, Z), out=vocab_mask_other)  # BxHx1 --> BxHxZ / (BxHxZ, BxHxZ) --> BxHxZ
					torch.ne(top_token_ids_3d, vocab_targets[None, None, :, Cm], out=vocab_mask).logical_or_(vocab_mask_other)  # noqa / ZxCmax --> 1x1xZ / (BxHx1, 1x1xZ) --> BxHxZ / (BxHxZ, BxHxZ) --> BxHxZ

				if length_alpha != 0:
					target_seq_len = torch.gather(input=target_seq_len, dim=1, index=top_candidates).add_(sample_padding.logical_not())  # (BxH, BxH) --> BxH / BxH --> BxH / (BxH, BxH) --> BxH

		target.masked_fill_(mask=target_padding, value=0)  # BxHxC --> BxHxC
		if length_alpha != 0:
			target_score = target_score_normed  # BxH

		return target, target_padding, target_score

	def precompute_generate_all(self, length_alpha: float, vocab_targets: Optional[torch.Tensor], vocab_per_token: bool, vocab_scaler: float, guide_targets: torch.Tensor, guide_renorm: bool) -> Any:

		W, Cmax = guide_targets.shape  # W = Number of guide targets, Cmax = self.target_config.token_length
		V = self.target_config.vocab_size  # V = Number of unique available token IDs

		guide_paddings = torch.zeros(size=(W, Cmax), dtype=self.target_config.mask_dtype, device=guide_targets.device)  # WxCmax
		guide_padding_slice = guide_paddings[:, 1:]  # WxCmax --> Wx(Cmax-1)
		torch.logical_not(guide_targets[:, :-1], out=guide_padding_slice)  # Wx(Cmax-1) --> Wx(Cmax-1)
		torch.cummax(guide_padding_slice, dim=1, out=(guide_padding_slice, guide_targets.new_empty(size=(W, Cmax - 1))))  # Wx(Cmax-1) --> (Wx(Cmax-1), Wx(Cmax-1))

		C = Cmax - guide_paddings.all(dim=0).sum().item()  # GPU-CPU synchronization point / WxCmax --> Cmax --> 1
		guide_paddings = guide_paddings[:, :C]  # WxCmax --> WxC
		guide_targets = guide_targets[:, :C].masked_fill(mask=guide_paddings, value=0)  # WxCmax --> WxC / (WxC, WxC) --> WxC

		if guide_renorm:
			guide_targets_T = guide_targets.T  # CxW
			guide_mask = torch.zeros(size=(W, C, W), dtype=torch.bool, device=guide_targets.device)  # WxCxW
			guide_mask_slice = guide_mask[:, 1:, :]  # WxCxW --> Wx(C-1)xW
			torch.ne(guide_targets[:, :-1, None], guide_targets_T[None, :-1, :], out=guide_mask_slice)  # WxC --> Wx(C-1)x1 / CxW --> 1x(C-1)xW / (Wx(C-1)x1, 1x(C-1)xW) --> Wx(C-1)xW
			torch.cummax(guide_mask_slice, dim=1, out=(guide_mask_slice, guide_targets.new_empty(size=(W, C - 1, W))))  # Wx(C-1)xW --> (Wx(C-1)xW, Wx(C-1)xW)
			guide_indices = guide_targets_T.expand(W, -1, -1).masked_fill(mask=guide_mask, value=V)  # CxW --> WxCxW / (WxCxW, WxCxW) --> WxCxW
			guide_scores = torch.full(size=(W, C, V + 1), fill_value=-torch.inf, dtype=self.embedder.embed_dtype, device=guide_targets.device).scatter_(dim=2, index=guide_indices, value=0)[None, :, :, :-1]  # noqa / (WxCx(V+1), WxCxW) --> WxCx(V+1) --> 1xWxCxV
		else:
			guide_scores = None

		if vocab_targets is None or vocab_scaler == 0:
			vocab_scores = None
		else:
			Z = vocab_targets.shape[0]
			vocab_targets = vocab_targets[:, :C]  # ZxCmax --> ZxC
			vocab_targets_T = vocab_targets.T  # CxZ
			vocab_mask = torch.zeros(size=(W, C, Z), dtype=torch.bool, device=vocab_targets.device)  # WxCxZ
			vocab_mask_slice = vocab_mask[:, 1:, :]  # WxCxZ --> Wx(C-1)xZ
			torch.ne(guide_targets[:, :-1, None], vocab_targets_T[None, :-1, :], out=vocab_mask_slice)  # WxC --> Wx(C-1)x1 / CxZ --> 1x(C-1)xZ / (Wx(C-1)x1, 1x(C-1)xZ) --> Wx(C-1)xZ
			torch.cummax(vocab_mask_slice, dim=1, out=(vocab_mask_slice, vocab_targets.new_empty(size=(W, C - 1, Z))))  # Wx(C-1)xZ --> (Wx(C-1)xZ, Wx(C-1)xZ)
			vocab_indices = vocab_targets_T.expand(W, -1, -1).masked_fill(mask=vocab_mask, value=V)  # CxZ --> WxCxZ / (WxCxZ, WxCxZ) --> WxCxZ
			if vocab_per_token:
				vocab_scores = torch.zeros(size=(W, C, V + 1), dtype=self.embedder.embed_dtype, device=vocab_targets.device).scatter_(dim=2, index=vocab_indices, value=1)[:, :, :-1]  # noqa / (WxCx(V+1), WxCxZ) --> WxCx(V+1) --> WxCxV
				vocab_scores.div_(vocab_scores.sum(dim=2, keepdim=True))  # WxCxV --> WxCx1 / (WxCxV, WxCx1) --> WxCxV
			else:
				vocab_scores = torch.zeros(size=(W, C, V + 1), dtype=self.embedder.embed_dtype, device=vocab_targets.device).scatter_(dim=2, index=vocab_indices, value=1, reduce='add')  # noqa / (WxCx(V+1), WxCxZ) --> WxCx(V+1)
				vocab_scores_count = vocab_scores[:, :, -1:]  # WxCx(V+1) --> WxCx1
				vocab_scores = vocab_scores[:, :, :-1].div_(torch.sub(Z, vocab_scores_count, out=vocab_scores_count))  # WxCx1 --> WxCx1 / (WxCxV, WxCx1) --> WxCxV
			vocab_scores = vocab_scores.gather(dim=2, index=guide_targets.unsqueeze(dim=2)).squeeze(dim=2)  # WxC --> WxCx1 / (WxCxV, WxCx1) --> WxCx1 --> WxC
			vocab_scores = vocab_scores.log_().nan_to_num_(nan=torch.inf, neginf=torch.inf, posinf=torch.inf)  # WxC --> WxC --> WxC
			vocab_scores.masked_fill_(mask=guide_paddings, value=0)  # (WxC, WxC) --> WxC
			vocab_scores = vocab_scores.sum(dim=1).mul_(vocab_scaler).unsqueeze(dim=0)  # WxC --> W --> W --> 1xW

		if length_alpha == 0:
			alpha_scale = None
		else:
			alpha_scale = guide_paddings.sum(dim=1)  # WxC --> W
			torch.sub(C, alpha_scale, out=alpha_scale)  # W --> W
			alpha_scale = torch.pow(alpha_scale.clamp_(min=1), exponent=-length_alpha, out=torch.empty(size=(W,), dtype=self.embedder.embed_dtype, device=alpha_scale.device)).unsqueeze(dim=0)  # W --> W --> W --> 1xW

		return guide_targets, guide_paddings, guide_scores, vocab_scores, alpha_scale  # (WxC, WxC, Optional[1xWxCxV], Optional[1xW], Optional[1xW])

	def generate_all(self, embed: torch.Tensor, topk: int, temperature: float, length_alpha: float, vocab_targets: Optional[torch.Tensor], vocab_per_token: bool, vocab_scaler: float, guide_targets: torch.Tensor, guide_renorm: bool, precompute: Optional[Any] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		# Note: This implementation is incompatible with setting both multi_target=True and multi_first=True (can train like that, but eval with multi_first=False)

		if precompute is None:
			guide_targets, guide_paddings, guide_scores, vocab_scores, alpha_scale = self.precompute_generate_all(length_alpha=length_alpha, vocab_targets=vocab_targets, vocab_per_token=vocab_per_token, vocab_scaler=vocab_scaler, guide_targets=guide_targets, guide_renorm=guide_renorm)
		else:
			guide_targets, guide_paddings, guide_scores, vocab_scores, alpha_scale = precompute  # (WxC, WxC, Optional[1xWxCxV], Optional[1xW], Optional[1xW])

		B = embed.shape[0]  # B = Batch size
		K = topk  # K = Top-k = Nominal guide targets chunk size
		W, C = guide_targets.shape  # W = Number of guide targets, C = Sequence length of guide targets

		scores = embed.new_empty(size=(B, W))
		for i in range(0, W, K):
			j = i + K
			guide_target = guide_targets[i:j, :].expand(B, -1, -1).contiguous()  # WxC --> HxC --> BxHxC where H <= K
			guide_padding = guide_paddings[i:j, :]  # WxC --> HxC
			logits = self(embed=embed, target=guide_target, target_padding=guide_padding.expand(B, -1, -1).contiguous(), target_weight=None, calc_loss=False, calc_correct=False, only_pred=False, guide_targets=None)[0]  # (BxF, BxHxC, BxHxC) --> BxHxCxV
			logits.div_(temperature)  # BxHxCxV --> BxHxCxV
			if guide_scores is not None:
				logits.add_(guide_scores[:, i:j, :, :])  # 1xWxCxV --> 1xHxCxV / (BxHxCxV, 1xHxCxV) --> BxHxCxV
			score = nn.functional.log_softmax(logits, dim=3)  # BxHxCxV --> BxHxCxV
			score = torch.gather(score, dim=3, index=guide_target.unsqueeze(dim=3)).squeeze(dim=3)  # BxHxC --> BxHxCx1 / (BxHxCxV, BxHxCx1) --> BxHxCx1 --> BxHxC
			score.masked_fill_(mask=guide_padding.unsqueeze(dim=0), value=0)  # (BxHxC, 1xHxC) --> BxHxC
			torch.sum(score, dim=2, out=scores[:, i:j])  # BxHxC --> BxH

		if vocab_scores is not None:
			scores.sub_(vocab_scores)  # (BxW, 1xW) --> BxW
		if alpha_scale is not None:
			scores.mul_(alpha_scale)  # (BxW, 1xW) --> BxW

		topk_scores, topk_indices = torch.topk(scores, k=K, dim=1, largest=True, sorted=True)  # BxW --> (BxK, BxK)
		topk_indices_3d = topk_indices.unsqueeze(dim=2).expand(-1, -1, C)  # BxK --> BxKx1 --> BxKxC
		topk_targets = guide_targets.expand(B, -1, -1).gather(dim=1, index=topk_indices_3d)  # WxC --> BxWxC / (BxWxC, BxKxC) --> BxKxC
		topk_paddings = guide_paddings.expand(B, -1, -1).gather(dim=1, index=topk_indices_3d)  # WxC --> BxWxC / (BxWxC, BxKxC) --> BxKxC

		return topk_targets, topk_paddings, topk_scores

#
# Helper modules
#

# Custom transformer encoder layer
class TransformerEncoderLayer(nn.TransformerEncoderLayer):

	def __init__(self, *args, device=None, dtype=None, rezero='none', **kwargs):

		super().__init__(*args, device=device, dtype=dtype, **kwargs)
		factory_kwargs = {'device': device, 'dtype': dtype}

		self.activation_relu_or_gelu = 0  # Note: Very dirty hack that SHOULD have the ONLY effect of disabling the sparsity fast path, as otherwise the custom method overrides in this class aren't even executed

		if rezero == 'none':
			self.scale1 = None
			self.scale2 = None
		elif rezero == 'perskip':
			self.scale1 = nn.Parameter(torch.zeros(size=(), **factory_kwargs))
			self.scale2 = nn.Parameter(torch.zeros(size=(), **factory_kwargs))
		elif rezero == 'perlayer':
			self.scale1 = nn.Parameter(torch.zeros(size=(), **factory_kwargs))
			self.scale2 = self.scale1
		else:
			raise ValueError(f"Invalid ReZero specification: {rezero}")

	def _sa_block(self, *args, **kwargs) -> torch.Tensor:
		x = super()._sa_block(*args, **kwargs)
		if self.scale1 is not None:
			x *= self.scale1
		return x

	def _ff_block(self, *args, **kwargs) -> torch.Tensor:
		x = super()._ff_block(*args, **kwargs)
		if self.scale2 is not None:
			x *= self.scale2
		return x

# Custom transformer decoder layer
class TransformerDecoderLayer(nn.TransformerDecoderLayer):

	def __init__(self, *args, device=None, dtype=None, rezero='none', **kwargs):

		super().__init__(*args, device=device, dtype=dtype, **kwargs)
		factory_kwargs = {'device': device, 'dtype': dtype}

		if rezero == 'none':
			self.scale1 = None
			self.scale2 = None
			self.scale3 = None
		elif rezero == 'perskip':
			self.scale1 = nn.Parameter(torch.zeros(size=(), **factory_kwargs))
			self.scale2 = nn.Parameter(torch.zeros(size=(), **factory_kwargs))
			self.scale3 = nn.Parameter(torch.zeros(size=(), **factory_kwargs))
		elif rezero == 'perlayer':
			self.scale1 = nn.Parameter(torch.zeros(size=(), **factory_kwargs))
			self.scale2 = self.scale1
			self.scale3 = self.scale1
		else:
			raise ValueError(f"Invalid ReZero specification: {rezero}")

	def _sa_block(self, *args, **kwargs) -> torch.Tensor:
		x = super()._sa_block(*args, **kwargs)
		if self.scale1 is not None:
			x *= self.scale1
		return x

	def _mha_block(self, *args, **kwargs) -> torch.Tensor:
		x = super()._mha_block(*args, **kwargs)
		if self.scale2 is not None:
			x *= self.scale2
		return x

	def _ff_block(self, *args, **kwargs) -> torch.Tensor:
		x = super()._ff_block(*args, **kwargs)
		if self.scale3 is not None:
			x *= self.scale3
		return x

# Embedding vector MLP
class EmbeddingVectorMLP(nn.Module):

	def __init__(
		self,
		embed_dim: int,                  # Dimension F of the embedding vectors to be transformed by this MLP
		output_dim: int,                 # Dimension E of the output vectors to be produced by this MLP
		output_seq_len: int,             # Number of output vectors to generate for each input embedding vector
		output_bias: bool,               # Whether the MLP should finish with a bias or not (e.g. if using positional embeddings directly on top of this MLP then an output bias is redundant)
		hidden_layer: str,               # String specifying how to compute the feature size of the hidden layer, e.g. 'none', 'min', 'max', 'amean' (arithmetic mean), 'gmean' (geometric mean)
		hidden_bias: bool,               # Whether to incorporate biases in case of a hidden layer (the last linear layer has a bias if output_bias is True, independent of this argument)
		hidden_norm: bool,               # Whether in case of a hidden layer a normalization layer should be used prior to the activation function
		hidden_activation: str,          # Activation module type to use after the first linear layer in the embedding vector MLP in case of a hidden layer (e.g. 'relu', 'gelu' or 'tanh', see utils.get_activation_gain)
		dropout_prob: Optional[float],   # Dropout probability (applied after MLP, None = No dropout layer)
		init_unit_norm: bool,            # Whether to initialise the MLP to work with unit norm (True) as opposed to unit standard deviation (False)
		init_balanced: Optional[float],  # The scale to perform a balanced init to (positive float, nominally 1, None = Do not perform balanced init). If provided, initialise the internal weights so that for unit embedding vector inputs, the expected output vector norms/stds (see init_unit_norm) are equal to this value (otherwise retain PyTorch-default weight initialisations).
		init_bias_zero: bool,            # Whether to initialise all non-output bias parameters to zero
		init_output_bias_zero: bool,     # Whether to initialise the output bias (if output_bias) to zero
	):

		super().__init__()
		self.embed_dim = embed_dim
		self.output_dim = output_dim
		self.output_seq_len = output_seq_len
		self.output_bias = output_bias
		self.hidden_layer = hidden_layer
		self.hidden_bias = hidden_bias
		self.hidden_norm = hidden_norm
		self.hidden_activation = hidden_activation
		self.dropout_prob = dropout_prob
		self.init_unit_norm = init_unit_norm
		self.init_balanced = init_balanced
		self.init_bias_zero = init_bias_zero
		self.init_output_bias_zero = init_output_bias_zero

		self.output_size = self.output_seq_len * self.output_dim
		if self.hidden_layer == 'none':
			self.hidden_size = None
		elif self.hidden_layer == 'min':
			self.hidden_size = min(self.embed_dim, self.output_size)
		elif self.hidden_layer == 'max':
			self.hidden_size = max(self.embed_dim, self.output_size)
		elif self.hidden_layer == 'amean':
			self.hidden_size = (self.embed_dim + self.output_size) // 2
			self.hidden_size = round(self.hidden_size / 64) * 64
		elif self.hidden_layer == 'gmean':
			self.hidden_size = math.sqrt(self.embed_dim * self.output_size)
			self.hidden_size = round(self.hidden_size / 64) * 64
		else:
			raise ValueError(f"Unsupported hidden layer argument: {self.hidden_layer}")
		if self.embed_dim <= 0 or self.output_size <= 0 or (self.hidden_size is not None and self.hidden_size <= 0):
			raise ValueError("Embedding vector MLP sizes cannot be non-positive")

		if self.init_balanced is None:
			self.init_output_norm = None
			self.init_output_std = None
		else:
			assert self.init_balanced > 0
			if self.init_unit_norm:
				self.init_output_norm = self.init_balanced
				self.init_output_std = self.init_balanced / math.sqrt(self.output_dim)
			else:
				self.init_output_norm = self.init_balanced * math.sqrt(self.output_dim)
				self.init_output_std = self.init_balanced

		def init_linear(linear, *, weight_std, output_std, bias_zero):
			if self.init_balanced is None:
				if linear.bias is not None and bias_zero:
					nn.init.zeros_(linear.bias)
			else:
				if linear.bias is None or bias_zero:
					nn.init.normal_(linear.weight, mean=0.0, std=weight_std)
					if linear.bias is not None:
						nn.init.zeros_(linear.bias)
				else:
					nn.init.normal_(linear.weight, mean=0.0, std=weight_std / math.sqrt(2))
					nn.init.normal_(linear.bias, mean=0.0, std=output_std / math.sqrt(2))

		if self.hidden_size is None:

			layers = [linear1 := nn.Linear(in_features=self.embed_dim, out_features=self.output_size, bias=self.output_bias)]
			init_linear(linear1, weight_std=self.init_output_std, output_std=self.init_output_std, bias_zero=self.init_output_bias_zero)

		else:

			activation, activation_gain = utils.get_activation_gain(name=self.hidden_activation, functional=False, unit_std=not self.init_unit_norm)

			layers = [linear1 := nn.Linear(in_features=self.embed_dim, out_features=self.hidden_size, bias=self.hidden_bias)]
			if self.hidden_norm:
				layers.append(norm := nn.LayerNorm(normalized_shape=self.hidden_size, bias=self.hidden_bias))
			else:
				norm = None
			layers.append(activation)
			layers.append(linear2 := nn.Linear(in_features=self.hidden_size, out_features=self.output_size, bias=self.output_bias))

			if self.init_balanced is not None:
				hidden_std = (self.init_output_norm / activation_gain) * math.sqrt(self.output_seq_len / self.hidden_size)
			elif self.init_unit_norm:
				hidden_std = math.sqrt(self.output_seq_len / self.hidden_size)  # Note: If unit norm is desired and not custom inited, the layer norm causes a large increase in activation standard deviation that would drown out the token and position embedding if we don't do this
			else:
				hidden_std = 1

			init_linear(linear1, weight_std=hidden_std, output_std=hidden_std, bias_zero=self.init_bias_zero)
			if norm is not None:
				nn.init.constant_(norm.weight, val=hidden_std)
				if norm.bias is not None:
					nn.init.zeros_(norm.bias)
			init_linear(linear2, weight_std=1 / math.sqrt(self.output_size), output_std=self.init_output_std, bias_zero=self.init_output_bias_zero)

		if self.dropout_prob is not None:
			layers.append(nn.Dropout(p=self.dropout_prob, inplace=False))
		self.mlp = nn.Sequential(*layers)

	def forward(self, embed: torch.Tensor) -> torch.Tensor:
		# embed = BxF batch of embedding vectors (vectors are normalized to unit norm internally just to be safe)
		# Returns a BxPxE batch of P-length sequences of E-dim vectors
		return self.mlp(torch.nn.functional.normalize(embed, dim=-1)).view(embed.shape[0], self.output_seq_len, self.output_dim)

# Learned positional embedding class
class LearnedPosEmbedding(nn.Module):

	def __init__(self, max_seq_len: int, pos_embed_dim: int, dropout_prob: float):
		# max_seq_len = Maximum sequence length S (i.e. Smax) to support (only this many positional embeddings exist)
		# pos_embed_dim = Dimension E of the positional embeddings
		# dropout_prob = Dropout probability (applied after adding positional embeddings)
		super().__init__()
		self.max_seq_len = max_seq_len
		self.pos_embed_dim = pos_embed_dim
		self.dropout_prob = dropout_prob
		self.embedding = nn.Embedding(num_embeddings=self.max_seq_len, embedding_dim=self.pos_embed_dim)
		self.dropout = nn.Dropout(p=self.dropout_prob, inplace=False)

	def forward(self, seq: torch.Tensor) -> torch.Tensor:
		# seq = BxSxE batch of vector sequences for S <= Smax
		S = seq.shape[1]
		if S > self.max_seq_len:
			raise ValueError(f"Sequence is longer than number of available positional embeddings: {S} > {self.max_seq_len}")
		return self.dropout(seq + self.embedding.weight[:S, :])

#
# Miscellaneous
#

# Parameter count class
@dataclasses.dataclass(frozen=True)
class ParamCount:

	total: int    # Total number of weights that exist
	used: int     # Number of weights that exist and are used
	unused: int   # Number of weights that exist but are not used
	trained: int  # Number of used weights that are trained (gradients enabled)
	frozen: int   # Number of used weights that are frozen (gradients disabled)

	def to_str(self):
		return f"{self.used} params{f' + {self.unused} unused' if self.unused != 0 else ''}{f' where used is {self.trained} trained + {self.frozen} frozen' if self.frozen != 0 else ''}"

	@staticmethod
	def from_modules(*modules: Optional[nn.Module], ignore_param: Optional[set[nn.Parameter]] = None, unused_map: Optional[dict[nn.Parameter, int]] = None):
		used_trained = used_frozen = unused = 0
		for module in modules:
			if module is None:
				continue
			for param in module.parameters():
				if ignore_param and param in ignore_param:
					continue
				count = param.numel()
				count_unused = unused_map.get(param, 0) if unused_map else 0
				count_used = count - count_unused
				if count_used < 0:
					raise ValueError("Number of unused parameters of a parameter tensor cannot exceed the tensor size")
				unused += count_unused
				if param.requires_grad:
					used_trained += count_used
				else:
					used_frozen += count_used
		used = used_trained + used_frozen
		total = used + unused
		return ParamCount(total=total, used=used, unused=unused, trained=used_trained, frozen=used_frozen)

	@staticmethod
	def from_counts(*counts: ParamCount):
		return ParamCount(
			total=sum(count.total for count in counts),
			used=sum(count.used for count in counts),
			unused=sum(count.unused for count in counts),
			trained=sum(count.trained for count in counts),
			frozen=sum(count.frozen for count in counts),
		)
# EOF
