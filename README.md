# QuickLLM

Script to run prompts quickly through local huggingface models

# Setup 

Tested using Python 3.10.11, conda 4.8.3, git 2.28.0

Set up conda environment called `quickllm` with
```bash
conda env create -f environment.yml
```

## Config

Create `.toml` file similar to one provided

### General
`prompts_file` - path to csv file with a column named `prompts`

`output` - path to folder to place csv file

### Model
`model_name` - huggingface string for model name

`batch_size` - batch size for generation

`model.generation_args` - generation args found [here](https://huggingface.co/docs/transformers/main_classes/text_generation) 
`model.generation_args: max_new_tokens`: number of new tokens to generate (required)

## Running

### Locally
```bash
python quickllm.py --config /path/to/config.toml
```

### Rockfish
To run on rockfish

You must first replace all <> items in `run_model`
I recommend 2 cpus per gpu and enough RAM and VRAM to load the model weights

```bash
sbatch run_model.job
```