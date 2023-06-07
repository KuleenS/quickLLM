import argparse

import csv

import os

import tomli 

from typing import Dict, Any, Iterable, List

import pandas as pd

import torch

from transformers import (
    AutoTokenizer,
    AutoModel,
    GenerationConfig
)

from tqdm import tqdm

class HuggingFaceOffline:
    def __init__(self, model_name: str, generation_args: Dict[str, Any], batch_size: int = 8):
        """HF offline model initializer

        :param model_name: name of model
        :type model_name: str
        :param temperature: temperature of model when generating, defaults to 1
        :type temperature: float, optional
        :param max_tokens: maximum number of tokens generated, defaults to 5
        :type max_tokens: int, optional
        """
        self.model_name = model_name
        self.batch_size = batch_size

        self.model = None

        self.generation_args = None

        n_gpus = torch.cuda.device_count()

        if n_gpus == 1:

            self.model = AutoModel.from_pretrained(self.model_name).to(0)
        
        else:
            self.model = AutoModel.from_pretrained(self.model_name, device_map = "balanced_low_0", torch_dtype=torch.float16, trust_remote_code=True)
            self.model = self.model.half()


        if self.model_name == ["tiiuae/falcon-40b-instruct", "facebook/opt-30b", "facebook/opt-66b"]:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, padding_side='left'
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )

        if self.model_name in ["huggyllama/llama-13b", "huggyllama/llama-65b"]:
            self.tokenizer.pad_token = '<unk>'
            self.tokenizer.pad_token_id = 0
        
        elif self.model_name in ["chavinlo/alpaca-13b", "chavinlo/alpaca-native"]:
            self.tokenizer.pad_token = '[PAD]'
            self.tokenizer.pad_token_id = 0
        
        elif self.model_name == "tiiuae/falcon-40b-instruct":
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.eval()

        if generation_args is not None:

            self.generation_args = GenerationConfig(**generation_args)
        
        else:
            self.generation_args = self.model.generation_config

        

    def get_response(self, prompts: Iterable[str]) -> Dict[str, Any]:
        """ "Get response from HF model with prompt batch

        :param prompt: prompt to send to model
        :type prompt: Iterable[str]
        :return: response of API endpoint
        :rtype: Dict[str, Any]
        """
        tokenized_input = self.tokenizer(
            prompts, padding=True, return_tensors="pt"
        )

        if self.model_name in [
            "chavinlo/alpaca-13b",
            "huggyllama/llama-13b",
            "huggyllama/llama-65b",
            "chavinlo/alpaca-native",
            "tiiuae/falcon-40b-instruct"
        ]:
            del tokenized_input["token_type_ids"]

        outputs = self.model.generate(
            **tokenized_input.to(0),
            generation_config=self.generation_config
        )

        del tokenized_input

        output = self.tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)

        return output

    def format_response(self, response: str) -> str:
        """Clean up response from Offline HF model and return generated string

        :param response: response from Offline HF model
        :type response: Dict[str, Any]
        :return: generated string
        :rtype: str
        """
        text = response.replace("\n", " ").strip()
        return text

    def generate_from_prompts(self, examples: Iterable[str]) -> List[str]:
        """Send all examples to offline HF model and get its responses

        :param examples: list of prompts
        :type examples: Iterable[str]
        :return: list of cleaned responses
        :rtype: List[str]
        """
        responses = []

        with torch.inference_mode():
            for i in tqdm(range(0, len(examples), self.batch_size), ncols=0):
                prompt_batch = examples[i : min(i + self.batch_size, len(examples))]

                response = self.get_response(prompt_batch)

                response = [self.format_response(x) for x in response]

                responses.extend(response)

        del self.model
        torch.cuda.empty_cache()

        return responses


def main(args):
    configpath = args.config

    # open config
    with open(configpath, "rb") as f:
        data = tomli.load(f)
    
    model_name = data['model']['name']
    batch_size = data['model']['batch_size']
    generation_args = data['model']['generation_args']

    prompts_file = data['prompts_file']

    prompts_output = data['output']

    prompts_df = pd.read_csv(prompts_file)

    prompts_df = prompts_df[~prompts_df.prompts.isna()]
    
    prompts = prompts_df['prompts'].tolist()

    model = HuggingFaceOffline(model_name, generation_args, batch_size)

    responses = model.generate_from_prompts(prompts)

    if not os.path.exists(os.path.join(prompts_output)):
        os.makedirs(os.path.join(prompts_output), exist_ok=True)

    model_name_clean = model_name.split("/")[1]

    with open(
        os.path.join(
            prompts_output, f"results_{model_name_clean}.csv"
        ),
        "w",
        encoding="utf-8",
    ) as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(["prompt", "response"])

        for prepared_result in zip(prompts, responses):
            csvwriter.writerow(prepared_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    main(args)


