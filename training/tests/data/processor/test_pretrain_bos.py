"""
Tests that pretraining's packed blocks start with the BOS token of a tokenizer that adds it through the post-processor
of its tokenizer.json alone, as Gemma 4's does, while its add_bos_token attribute, which comes from
tokenizer_config.json only, says False. The tests build tiny word-level tokenizers, and run on the CPU.
"""

from types import SimpleNamespace

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from hmf.data.decision_prompt import adds_bos_token
from hmf.data.processor.pretrain import PretrainDatasetProcessor

VOCAB = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "a": 3, "b": 4, "c": 5, "[UNK]": 6}


def _tokenizer(post_processor_adds_bos: bool, **kwargs) -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    if post_processor_adds_bos:
        backend.post_processor = processors.TemplateProcessing(
            single="<bos> $A", pair="<bos> $A $B", special_tokens=[("<bos>", 2)]
        )
    return PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>",
                                   **kwargs)


def test_bos_added_by_the_post_processor_alone_is_detected():
    tokenizer = _tokenizer(post_processor_adds_bos=True)
    assert not tokenizer.add_bos_token and tokenizer("a").input_ids[0] == 2
    assert adds_bos_token(tokenizer)


def test_tokenizer_that_adds_no_bos_is_detected():
    assert not adds_bos_token(_tokenizer(post_processor_adds_bos=False))


def test_add_bos_token_attribute_is_followed():
    assert adds_bos_token(_tokenizer(post_processor_adds_bos=False, add_bos_token=True))


def test_packed_blocks_start_with_the_bos_token():
    data_args = SimpleNamespace(template="empty", add_eos_token=True, packing=True, cutoff_len=4)
    processor = PretrainDatasetProcessor(template=None, tokenizer=_tokenizer(post_processor_adds_bos=True),
                                         processor=None, data_args=data_args)
    examples = {"_prompt": [[{"content": "a b"}], [{"content": "c a b"}]]}
    # "a b <eos> c a b <eos>" packs into one block of four tokens, whose first token becomes the BOS token
    assert processor.preprocess_dataset(examples)["input_ids"] == [[2, 4, 1, 5]]
