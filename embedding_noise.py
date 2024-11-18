# Embedding training noise classes

# Imports
from __future__ import annotations
import math
from typing import Optional
import torch
from logger import log

#
# Embedding noise
#

# Embedding noise class
class EmbeddingNoise(torch.nn.Module):

	@staticmethod
	def create(
		scheme: str,       # Embedding noise scheme: GaussElem, GaussVec, GaussAngle, UniformAngle
		embed_dim: int,    # Dimension of the embedding vectors
		vec_norm: float,   # Vector norm of the noise
		angle_min: float,  # Minimum noise angle in degrees
		angle_max: float,  # Maximum noise angle in degrees
		angle_std: float,  # Standard deviation of noise angle in degrees
		mix_ratio: float,  # Ratio to mix in a secondary noise scheme by
	) -> Optional[EmbeddingNoise]:
		if not scheme:
			return None
		else:
			scheme_lower = scheme.lower()
			if scheme_lower == 'gausselem':
				return GaussElemNoise(embed_dim=embed_dim, vec_norm=vec_norm)
			elif scheme_lower == 'gaussvec':
				return GaussVecNoise(embed_dim=embed_dim, vec_norm=vec_norm)
			elif scheme_lower == 'gaussangle':
				return GaussAngleNoise(embed_dim=embed_dim, angle_std=angle_std, angle_max=angle_max)
			elif scheme_lower == 'uniformangle':
				return UniformAngleNoise(embed_dim=embed_dim, angle_min=angle_min, angle_max=angle_max)
			elif scheme_lower == 'gausselemuniformangle':
				return GaussElemUniformAngleNoise(embed_dim=embed_dim, vec_norm=vec_norm, angle_min=angle_min, angle_max=angle_max, mix_ratio=mix_ratio)
			else:
				raise ValueError(f"Unsupported embedding noise type: {scheme}")

	def __init__(self, scheme: str, embed_dim: int):
		super().__init__()
		self.scheme = scheme
		self.embed_dim = embed_dim

	def forward(self, embed: torch.Tensor) -> torch.Tensor:
		# embed = Input BxF embedding vectors (MUST be unit vectors, will potentially be modified in-place)
		# Return the corresponding embedding vectors with the effect of noise (could return embed itself if it was modified totally in-place)
		raise NotImplementedError

#
# Embedding noise implementations
#

# Element-level Gaussian embedding noise
class GaussElemNoise(EmbeddingNoise):

	def __init__(self, embed_dim: int, vec_norm: float):
		super().__init__(scheme='GaussElem', embed_dim=embed_dim)
		self.vec_norm = vec_norm
		self.elem_std = self.vec_norm / math.sqrt(self.embed_dim)
		if self.elem_std <= 0:
			raise ValueError(f"Element noise standard deviation must be positive: {self.elem_std:.3g}")
		log.info(f"Applying {self.scheme} noise of mean norm {self.vec_norm:.3g} to embedding vectors")

	def extra_repr(self) -> str:
		return f"embed_dim={self.embed_dim}, vec_norm={self.vec_norm:.3g}, elem_std={self.elem_std:.3g}"

	def forward(self, embed: torch.Tensor) -> torch.Tensor:
		embed.add_(torch.randn_like(embed), alpha=self.elem_std)
		torch.nn.functional.normalize(embed, dim=-1, out=embed)
		return embed

# Vector-level Gaussian embedding noise
class GaussVecNoise(EmbeddingNoise):

	def __init__(self, embed_dim: int, vec_norm: float):
		super().__init__(scheme='GaussVec', embed_dim=embed_dim)
		self.vec_norm = vec_norm
		if self.vec_norm <= 0:
			raise ValueError(f"Vector noise norm must be positive: {self.vec_norm:.3g}")
		log.info(f"Applying {self.scheme} noise of norm std {self.vec_norm:.3g} to embedding vectors")

	def extra_repr(self) -> str:
		return f"embed_dim={self.embed_dim}, vec_norm={self.vec_norm:.3g}"

	def forward(self, embed: torch.Tensor) -> torch.Tensor:
		noise = torch.randn_like(embed)
		noise = torch.nn.functional.normalize(noise, dim=-1, out=noise)
		embed.addcmul_(noise, torch.randn(size=(embed.shape[0], 1), dtype=embed.dtype, device=embed.device), value=self.vec_norm)
		torch.nn.functional.normalize(embed, dim=-1, out=embed)
		return embed

# Angle noise base class
class AngleNoise(EmbeddingNoise):

	def rand_angle(self, embed: torch.Tensor) -> torch.Tensor:
		# embed = BxF embedding vectors
		# Return a random Bx1 angle tensor of the same dtype and device as embed
		raise NotImplementedError

	def forward(self, embed: torch.Tensor) -> torch.Tensor:
		noise_dirn = torch.randn_like(embed)
		noise_dirn.addcmul_(embed, torch.linalg.vecdot(embed, noise_dirn, dim=1).unsqueeze(dim=1), value=-1)
		noise_dirn = torch.nn.functional.normalize(noise_dirn, dim=-1, out=noise_dirn)
		angle = self.rand_angle(embed=embed)
		embed.mul_(angle.cos()).addcmul_(noise_dirn, angle.sin())
		torch.nn.functional.normalize(embed, dim=-1, out=embed)  # Just to be sure...
		return embed

# Gaussian angle noise
class GaussAngleNoise(AngleNoise):

	def __init__(self, embed_dim: int, angle_std: float, angle_max: float):
		# Note: angle_std and angle_max are in degrees
		super().__init__(scheme='GaussAngle', embed_dim=embed_dim)
		self.angle_std = angle_std
		self.angle_max = angle_max
		self.angle_std_rad = math.radians(self.angle_std)
		self.angle_max_rad = math.radians(self.angle_max)
		if self.angle_std_rad <= 0 or self.angle_max_rad <= 0:
			raise ValueError(f"Angular noise standard deviation and maximum value must both be positive: std {self.angle_std_rad:.3g} radians, max {self.angle_max_rad:.3g} radians")
		log.info(f"Applying {self.scheme} noise of angle std {self.angle_std:.3g}\xB0 and angle max {self.angle_max:.3g}\xB0 to embedding vectors")

	def extra_repr(self) -> str:
		return f"embed_dim={self.embed_dim}, angle_std={self.angle_std:.3g}\xB0, angle_max={self.angle_max:.3g}\xB0"

	def rand_angle(self, embed: torch.Tensor) -> torch.Tensor:
		return torch.randn(size=(embed.shape[0], 1), dtype=embed.dtype, device=embed.device).mul_(self.angle_std_rad).clamp_(min=-self.angle_max_rad, max=self.angle_max_rad)

# Uniform angle noise
class UniformAngleNoise(AngleNoise):

	def __init__(self, embed_dim: int, angle_min: float, angle_max: float):
		# Note: angle_min and angle_max are in degrees
		super().__init__(scheme='UniformAngle', embed_dim=embed_dim)
		self.angle_min = angle_min
		self.angle_max = angle_max
		self.angle_min_rad = math.radians(self.angle_min)
		self.angle_max_rad = math.radians(self.angle_max)
		if self.angle_min_rad > self.angle_max_rad:
			raise ValueError(f"Minimum angular noise must be smaller than maximum angular noise: min {self.angle_min_rad:.3g} radians, max {self.angle_max_rad:.3g} radians")
		log.info(f"Applying {self.scheme} noise of angle range {self.angle_min:.3g}\xB0 to {self.angle_max:.3g}\xB0 to embedding vectors")

	def extra_repr(self) -> str:
		return f"embed_dim={self.embed_dim}, angle_min={self.angle_min:.3g}\xB0, angle_max={self.angle_max:.3g}\xB0"

	def rand_angle(self, embed: torch.Tensor) -> torch.Tensor:
		return embed.new_empty(size=(embed.shape[0], 1)).uniform_(self.angle_min_rad, self.angle_max_rad)

# Element-level Gaussian embedding noise with uniform angle noise mixed in
class GaussElemUniformAngleNoise(EmbeddingNoise):

	def __init__(self, embed_dim: int, vec_norm: float, angle_min: float, angle_max: float, mix_ratio: float):
		super().__init__(scheme='GaussElemUniformAngle', embed_dim=embed_dim)
		self.gauss_elem_noise = GaussElemNoise(embed_dim=embed_dim, vec_norm=vec_norm)
		self.uniform_angle_noise = UniformAngleNoise(embed_dim=embed_dim, angle_min=angle_min, angle_max=angle_max)
		self.mix_ratio = mix_ratio
		if self.mix_ratio < 0 or self.mix_ratio > 1:
			raise ValueError(f"Mix ratio must be in the range [0, 1]: {self.mix_ratio:.3g}")
		log.info(f"Applying {self.scheme} noise with mix ratio {self.mix_ratio:.3g} of {self.uniform_angle_noise.scheme}")

	def extra_repr(self) -> str:
		return f"mix_ratio={self.mix_ratio:.3g}"

	def forward(self, embed: torch.Tensor) -> torch.Tensor:
		embed_uniform_angle = self.uniform_angle_noise(embed=embed.clone())
		embed = self.gauss_elem_noise(embed=embed)
		return torch.where(torch.rand(size=(embed.shape[0], 1), dtype=embed.dtype, device=embed.device) < self.mix_ratio, embed_uniform_angle, embed)
# EOF
