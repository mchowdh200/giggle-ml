# @title Batch example
"""
Let's say you want to do inference on a dataset to grab a lot of embeddings,
you can just loop thru a dataloader like this.

"""

import json
import os
import subprocess

import torch
from transformers import PreTrainedModel

from gpu_embeds.standalone_hyenadna import HyenaDNAModel


# helper 1


def inject_substring(orig_str):
    """Hack to handle matching keys between models trained with and without
    gradient checkpointing.
    Written by hyenadna authors."""

    # modify for mixer keys
    pattern = r"\.mixer"
    injection = ".mixer.layer"

    modified_string = re.sub(pattern, injection, orig_str)

    # modify for mlp keys
    pattern = r"\.mlp"
    injection = ".mlp.layer"

    modified_string = re.sub(pattern, injection, modified_string)

    return modified_string

# helper 2


def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """Loads pretrained (backbone only) weights into the scratch state dict.
    Written by hyenadna authors."""

    # loop thru state dict of scratch
    # find the corresponding weights in the loaded model, and set it

    # need to do some state dict "surgery"
    for key, value in scratch_dict.items():
        if 'backbone' in key:
            # the state dicts differ by one prefix, '.model', so we add that
            key_loaded = 'model.' + key
            # breakpoint()
            # need to add an extra ".layer" in key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except:
                raise Exception('key mismatch in the state dicts!')

    # scratch_dict has been updated
    return scratch_dict


class HyenaDNAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.

    Written by hyenadna authors.
    """
    base_model_prefix = "hyenadna"

    def __init__(self, config):
        pass

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    @classmethod
    def from_pretrained(cls,
                        path,
                        model_name,
                        config=None,
                        device='cpu',
                        use_head=False,
                        n_classes=2,
                        ):
        # first check if it is a local path
        pretrained_model_name_or_path = os.path.join(path, model_name)
        if os.path.isdir(pretrained_model_name_or_path):
            print("Using cached model checkpoint",
                  pretrained_model_name_or_path)
            if config is None:
                config = json.load(
                    open(os.path.join(pretrained_model_name_or_path, 'config.json')))
        else:
            print("Downloading model from Hugging Face")
            hf_url = f'https://huggingface.co/LongSafari/{model_name}'

            subprocess.run(
                f'rm -rf {pretrained_model_name_or_path}', shell=True)
            command = f'mkdir -p {path} && cd {path} && git lfs install && git clone {hf_url}'
            subprocess.run(command, shell=True)

            if config is None:
                config = json.load(
                    open(os.path.join(pretrained_model_name_or_path, 'config.json')))

        scratch_model = HyenaDNAModel(
            **config, use_head=use_head, n_classes=n_classes)  # the new model format
        loaded_ckpt = torch.load(
            os.path.join(pretrained_model_name_or_path, 'weights.ckpt'),
            map_location=torch.device(device)
        )

        # need to load weights slightly different if using gradient checkpointing
        if config.get("checkpoint_mixer", False):
            checkpointing = config["checkpoint_mixer"] == True or config["checkpoint_mixer"] == True
        else:
            checkpointing = False

        # grab state dict from both and load weights
        state_dict = load_weights(scratch_model.state_dict(
        ), loaded_ckpt['state_dict'], checkpointing=checkpointing)

        # scratch model has now been updated
        scratch_model.load_state_dict(state_dict)
        print("Loaded pretrained weights ok!")
        return scratch_model


def prepare_model(rank, device):
    '''
    this selects which backbone to use, and grabs weights/ config from HF
    4 options:
      'hyenadna-tiny-1k-seqlen'   # fine-tune on colab ok
      'hyenadna-small-32k-seqlen'
      'hyenadna-medium-160k-seqlen'  # inference only on colab
      'hyenadna-medium-450k-seqlen'  # inference only on colab
      'hyenadna-large-1m-seqlen'  # inference only on colab
    '''

    # TODO: this should pull via AutoModel.from_pretrained

    # select model
    # use None if training from scratch
    pretrained_model_name = 'hyenadna-tiny-1k-seqlen'

    max_lengths = {
        'hyenadna-tiny-1k-seqlen': 1024,
        'hyenadna-small-32k-seqlen': 32768,
        'hyenadna-medium-160k-seqlen': 160000,
        'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
        'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
    }

    # we need these for the decoder head, if using
    use_head = False
    n_classes = 2  # not used for embeddings only

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one in None
    backbone_cfg = None

    # instantiate the model (pretrained here)
    if pretrained_model_name not in ['hyenadna-tiny-1k-seqlen',
                                     'hyenadna-small-32k-seqlen',
                                     'hyenadna-medium-160k-seqlen',
                                     'hyenadna-medium-450k-seqlen',
                                     'hyenadna-large-1m-seqlen']:
        raise ValueError(
            f"Invalid pretrained model name: {pretrained_model_name}")

    model = HyenaDNAPreTrainedModel.from_pretrained(
        './checkpoints',
        pretrained_model_name,
        config=backbone_cfg,
        device=device,
        use_head=use_head,
        n_classes=n_classes)

    return model
