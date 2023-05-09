# cog-mpt-7b-storywriter-65k
A cog implementation of MosaicML's MPT-7B-StoryWriter-65k+ Large Language Model

This is a guide to running MPT-7B-StoryWriter-65k+ in the cloud using Replicate. You'll use the [Cog](https://github.com/replicate/cog) command-line tool to package the model and push it to Replicate as a web interface and API.

MPT-7B-StoryWriter-65k+ is a language model that specializes in generating fictional stories with lengthy context lengths. The model was created by finetuning MPT-7B with a context length of 65k tokens on a filtered fiction subset of the books3 dataset. Thanks to ALiBi, the model can extrapolate beyond 65k tokens at inference time, allowing for longer story generations. The MosaicML team demonstrated the ability to generate stories as long as 84k tokens on a single node of 8 A100-80GB GPUs in their blog post.

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


## Step 2: Run the model

You can run the model locally to test it:

```
cog predict -i prompt="On a dark and stormy night "
```

## Step 3: Create a model on Replicate

Go to [replicate.com/create](https://replicate.com/create) to create a Replicate model.

Make sure to specify "private" to keep the model private.

## Step 4: Configure the model to run on A100 GPUs

Replicate supports running models on a variety of GPUs. The default GPU type is a T4, but for best performance you'll want to configure your model to run on an A100.

Click on the "Settings" tab on your model page, scroll down to "GPU hardware", and select "A100". Then click "Save".

## Step 5: Push the model to Replicate

Log in to Replicate:

```
cog login
```

Push the contents of your current directory to Replicate, using the model name you specified in step 3:

```
cog push r8.im/username/modelname
```

[Learn more about pushing models to Replicate.](https://replicate.com/docs/guides/push-a-model)
