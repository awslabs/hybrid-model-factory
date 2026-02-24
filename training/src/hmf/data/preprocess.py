# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import TYPE_CHECKING, Callable, Literal

from .processor.sequence_parallel import pad_sequence, sp_split


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ..hparams import DataArguments, ModelArguments


def get_sequence_parallel_preprocess(
    data_args: "DataArguments",
    model_args: "ModelArguments",
    stage: Literal["pad", "split"],
    tokenizer: "PreTrainedTokenizer",
) -> Callable:
    if stage == "pad":
        preprocess_func = partial(pad_sequence, data_args=data_args, tokenizer=tokenizer)
    elif stage == "split":
        preprocess_func = partial(sp_split, model_args=model_args)
    else:
        raise NotImplementedError(f"Unexpected stage in sequence_parallel_preprocess: {stage}")

    return preprocess_func
