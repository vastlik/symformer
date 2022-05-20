from dataclasses import dataclass
from typing import List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .input_regularizers import Regularizer


DecoderMode = Literal["sum", "concat", "no_reg"]


@dataclass
class EncoderConfig:
    row_wise_ff_num_layers: int
    num_heads: int
    num_layers: int
    d_model: int
    m_inducing_points: int
    k_seed_vectors: int = 32
    dropout_rate: Optional[float] = None


@dataclass
class DecoderConfig:
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    max_len: int
    dropout_rate: float = 0.1


@dataclass
class DecoderWithRegressionConfig(DecoderConfig):
    mode: DecoderMode = "sum"


@dataclass
class DatasetConfig:
    path: str
    valid_path: str
    num_of_not_seen: int
    batch_size: int
    sample_points: bool
    test_size: int
    dataset_size: int
    variables: List[str]
    num_points: int
    extended_representation: bool = True


@dataclass
class GreedySearchConfig:
    max_num: int
    evaluate_each: int


@dataclass
class BeamSearchConfig(GreedySearchConfig):
    beam_width: int


@dataclass
class CallbackConfig:
    greedy_search: GreedySearchConfig
    beam_search: BeamSearchConfig
    regression_delay: int


@dataclass
class Config:
    encoder_config: EncoderConfig
    decoder_config: Union[DecoderConfig, DecoderWithRegressionConfig]
    loss: Literal["mse", "mae", "huber"]
    reg_head_num_layers: int
    reg_head_dim: int
    dataset_config: DatasetConfig
    max_epoch: int
    input_regularizer: Optional[Regularizer]
    callback_config: CallbackConfig
    regression_lambda: float = 0.01
    label_smoothing: float = 0.0
