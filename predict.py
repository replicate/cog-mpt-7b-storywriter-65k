import time
from typing import Optional
import subprocess

import torch
import os

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
from collections import OrderedDict
from cog import BasePredictor, ConcatenateIterator, Input, Path


# from config import DEFAULT_MODEL_NAME, DEFAULT_CONFIG_PATH, load_tokenizer, load_tensorizer
from subclass import YieldingMPT

TENSORIZER_WEIGHTS_PATH = "model/mpt-7b-storywriter-65.tensors"  # path from which we pull weights when there's no COG_WEIGHTS environment variable
# TENSORIZER_WEIGHTS_PATH = None 

DEFAULT_CONFIG_PATH = "model/"
TOKENIZER_PATH = "model/"

def maybe_download(path):
    if path.startswith("gs://"):
        st = time.time()
        output_path = "/tmp/weights.tensors"
        subprocess.check_call(["gcloud", "storage", "cp", path, output_path])
        print(f"weights downloaded in {time.time() - st}")
        return output_path
    return path


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # set TOKENIZERS_PARALLELISM to false to avoid a warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if weights is not None and weights.name == "weights":
            # bugfix
            weights = None
        if weights is None and TENSORIZER_WEIGHTS_PATH:
            self.model = self.load_tensorizer(
                weights=maybe_download(TENSORIZER_WEIGHTS_PATH), plaid_mode=True, cls=YieldingMPT, config_path=DEFAULT_CONFIG_PATH,
            )
        
        elif hasattr(weights, "filename") and "tensors" in weights.filename:
            self.model = self.load_tensorizer(
                weights=weights, plaid_mode=True, cls=YieldingMPT, config_path=DEFAULT_CONFIG_PATH,
            )
        elif hasattr(weights, "suffix") and "tensors" in weights.suffix:
            self.model = self.load_tensorizer(
                weights=weights, plaid_mode=True, cls=YieldingMPT
            )
        # elif "tensors" in weights:
        #     self.model = self.load_tensorizer(
        #         weights=weights, plaid_mode=True, cls=YieldingMPT
        #     )
        else:
            weights = "./model/"

            self.model = self.load_huggingface_model(weights=weights)

        self.tokenizer = self.load_tokenizer(TOKENIZER_PATH)
    
    def load_tokenizer(self, path):
        tokenizer = AutoTokenizer.from_pretrained(path)

        return tokenizer

    def load_huggingface_model(self, weights=None):

        config = AutoConfig.from_pretrained(
            weights,
            trust_remote_code=True
        )

        config.attn_config['attn_impl'] = 'triton'

        st = time.time()
        print(f"loading weights from {weights} w/o tensorizer")
        model = YieldingMPT.from_pretrained(
            weights, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        model.to(self.device)
        print(f"weights loaded in {time.time() - st}")
        return model
    
    def load_tensorizer(self, weights, plaid_mode, cls, config_path):
        st = time.time()
        print(f"deserializing weights from {weights}")

        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        config.attn_config['attn_impl'] = 'triton'


        model = no_init_or_tensor(
            lambda: cls.from_pretrained(
                None, config=config, state_dict=OrderedDict(), trust_remote_code=True,
            )
        )


        des = TensorDeserializer(weights, plaid_mode=True)
        des.load_into_module(model)
        model = model.to(dtype=torch.bfloat16)

        print(f"weights loaded in {time.time() - st}")
        return model

    def predict(
        self,
        prompt: str = Input(description=f"Prompt to send to MPT-StoryWriter."),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=500,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1,
        ),
        length_penalty: float = Input(
            description="Increasing the length_penalty parameter above 1.0 will cause the model to favor longer sequences, while decreasing it below 1.0 will cause the model to favor shorter sequences.",
            ge=0.01,
            le=5,
            default=1,
        ),
        no_repeat_ngram_size: int = Input(
            description="If set to int > 0, all ngrams of size no_repeat_ngram_size can only occur once.",
            ge=0,
            default=0,
        ),
        stop_sequence: str = Input(
            description="Generation will hault if this token is produced. Currently, only single token stop sequences are support and it is recommended to use `###` as the stop sequence if you want to control generation termination.",
            default=None,
        ),
        seed: int = Input(
            description="Set seed for reproducible outputs. Set to -1 for random seed.",
            ge=-1,
            default=-1,
        ),
        debug: bool = Input(
            description="provide debugging output in logs", default=False
        ),
    ) -> ConcatenateIterator[str]:
        input = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # set torch seed
        if seed == -1:
            torch.seed()

        else:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        with torch.inference_mode():
            first_token_yielded = False
            prev_ids = []
            for output in self.model.generate(
                input,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            ):
                cur_id = output.item()

                # in order to properly handle spaces, we need to do our own tokenizing. Fun!
                # we're building up a buffer of sub-word / punctuation tokens until we hit a space, and then yielding whole words + punctuation.
                cur_token = self.tokenizer.convert_ids_to_tokens(cur_id)

                # skip initial newline, which this almost always yields. hack - newline id = 13.
                if not first_token_yielded and not prev_ids and cur_id == 187:
                    continue

                # underscore means a space, means we yield previous tokens
                if cur_token.startswith("Ġ"):  # this is not a standard underscore.
                    # first token
                    if not prev_ids:
                        prev_ids = [cur_id]
                        continue

                    # there are tokens to yield
                    else:
                        token = self.tokenizer.decode(prev_ids)
                        prev_ids = [cur_id]

                        if not first_token_yielded:
                            # no leading space for first token
                            token = token.strip()
                            first_token_yielded = True
                        yield token
                                # End token
                elif cur_token == "<|endoftext|>":
                    break
                
                elif stop_sequence and cur_token == stop_sequence:
                    break

                else:
                    prev_ids.append(cur_id)
                    continue

            # remove any special tokens such as </s>
            token = self.tokenizer.decode(prev_ids, skip_special_tokens=True)
            if not first_token_yielded:
                # no leading space for first token
                token = token.strip()
                first_token_yielded = True
            yield token

        if debug:
            print(f"cur memory: {torch.cuda.memory_allocated()}")
            print(f"max allocated: {torch.cuda.max_memory_allocated()}")
            print(f"peak memory: {torch.cuda.max_memory_reserved()}")

