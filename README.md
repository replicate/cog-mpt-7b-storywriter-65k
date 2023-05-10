# cog-mpt-7b-storywriter-65k
[![Replicate](https://replicate.com/replicate/mpt-7b-storywriter/badge)](https://replicate.com/replicate/mpt-7b-storywriter) 

A cog implementation of MosaicML's MPT-7B-StoryWriter-65k+ Large Language Model

This is a guide to running MPT-7B-StoryWriter-65k+ in the cloud using Replicate. You'll use the [Cog](https://github.com/replicate/cog) command-line tool to package the model and push it to Replicate as a web interface and API.

MPT-7B-StoryWriter-65k+ is a language model that specializes in generating fictional stories with lengthy context lengths. The model was created by finetuning MPT-7B with a context length of 65k tokens on a filtered fiction subset of the books3 dataset. Thanks to ALiBi, the model can extrapolate beyond 65k tokens at inference time, allowing for longer story generations. The MosaicML team demonstrated the ability to generate stories as long as 84k tokens on a single node of 8 A100-80GB GPUs in their blog [post]([url](https://www.mosaicml.com/blog/mpt-7b)).

## Prerequisites

- **GPU machine**. You'll need a Linux machine with an NVIDIA GPU attached and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed. If you don't already have access to a machine with a GPU, check out our [guide to getting a 
GPU machine](https://replicate.com/docs/guides/get-a-gpu-machine).

- **Docker**. You'll be using the [Cog](https://github.com/replicate/cog) command-line tool to build and push a model. Cog uses Docker to create containers for models.

## Step 0: Install Cog

First, [install Cog](https://github.com/replicate/cog#install):

```
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

## Step 1: Set up weights

You can use the following script to pull the model weights from the Hugging Face Hub. We also recommend using `tensorizer` to tensorize your weights, which will dramatically reduce the time it takes to load your model. 


```
chmod +x scripts/download_and_prepare_model.py
cog run python scripts/download_and_prepare_model.py --model_name mosaicml/mpt-7b-storywriter --model_path model --tensorize --tensorizer_path model/mpt-7b-storywriter-65.tensors
```

## Step 2: Run the model

You can run the model locally to test it:

```
cog predict -i prompt="On a dark and stormy night "
```

## Step 3: Push your model weights to cloud storage

If you want to deploy your own cog version of this model, we recommend pushing the tensorized weights to a public bucket. You can then configure the `setup` method in `predict.py` to pull the tensorized weights. 

Currently, we provide boiler-plate code for pulling weights from GCP. To use the current configuration, simply set `TENSORIZER_WEIGHTS_PATH` to the public Google Cloud Storage Bucket path of your tensorized model weights. At setup time, they'll be downloaded and loaded into memory. 

Alternatively, you can implement your own solution using your cloud storage provider of choice. 

To see if the remote weights configuration works, you can run the model locally.

## Step 4: Create a model on Replicate

Go to [replicate.com/create](https://replicate.com/create) to create a Replicate model.

Make sure to specify "private" to keep the model private.

## Step 5: Configure the model to run on A100 GPUs

Replicate supports running models on a variety of GPUs. The default GPU type is a T4, but for best performance you'll want to configure your model to run on an A100.

Click on the "Settings" tab on your model page, scroll down to "GPU hardware", and select "A100". Then click "Save".

## Step 6: Push the model to Replicate

Log in to Replicate:

```
cog login
```

Push the contents of your current directory to Replicate, using the model name you specified in step 3:

```
cog push r8.im/username/modelname
```

[Learn more about pushing models to Replicate.](https://replicate.com/docs/guides/push-a-model)
