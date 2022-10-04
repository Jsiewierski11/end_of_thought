from dataclasses import dataclass, fields, field, asdict

import json
import argparse
import yaml


@dataclass
class ClassificationConfig:
    data_dir: str
    cache_dir: str
    model_name_or_path: str
    tokenizer_name: str
    config_name: str
    output_dir: str
    do_train: bool
    do_predict: bool
    weight_decay: float
    adam_epsilon: float
    warmup_steps: int
    adafactor: bool
    learning_rate: float
    lr_scheduler: str
    gradient_accumulation_steps: int
    fp16: bool
    max_epochs: int
    max_grad_norm: float
    max_seq_length: int
    train_batch_size: int
    eval_batch_size: int
    overwrite_cache: bool
    accumulate_grad_batches: int
    gpus: int
    n_tpu_cores: int
    num_train_epochs: int
    seed: int

    @classmethod
    def from_json(cls, json_file):
        config_json = json.load(open(json_file))
        return cls(**config_json)

    @classmethod
    def from_yaml(cls, yaml_file):
        config_yaml = yaml.load(open(yaml_file), yaml.FullLoader)
        return cls(**config_yaml)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def to_argparse(self, **kwargs):
        parser = argparse.ArgumentParser(description=self.__doc__)
        for attr, f in zip(self.__dict__, fields(self)):
            value = self.__dict__[attr]
            value_or_class = f.type
            if f.metadata:
                help = f.metadata['help']
                parser.add_argument("--" + str(attr.replace('_', '-')), default=value, type=value_or_class, help=help)
            else:
                parser.add_argument("--" + str(attr.replace('_', '-')), default=value, type=value_or_class)

        arg = parser.parse_args(**kwargs)
        return arg