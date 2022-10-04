import os
import glob
import torch
import logging
import argparse
import transformers
import numpy as np
from torch import nn
from pathlib import Path
from typing import Any, Dict
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer
from argparse import ArgumentParser, Namespace
from pytorch_lightning import (
    LightningModule,
    Trainer,
    Callback,
    utilities,
    seed_everything,
    callbacks
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import glue_convert_examples_to_features as convert_examples_to_features

from utils.config import ClassificationConfig
from dataset import EndOfThoughtDataProcessor
from utils.utils import compute_metrics

logger = logging.getLogger(__name__)


class RoBertaFinetuner(LightningModule):
    def __init__(self, hparams: argparse.Namespace, **config_kwargs):
        super().__init__()
        self.save_hyperparameters(hparams)

        # Additional hparams
        self.hparams.glue_output_mode = "classification"
        num_labels = 2
        cache_dir = self.hparams.cache_dir
        self.step_count = 0

        # self.model = BertModel.from_pretrained("bert-base-cased", output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", output_attentions=True)

        self.config = AutoConfig.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                **({"num_labels": num_labels} if num_labels is not None else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )

        self.model = AutoModelForSequenceClassification.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=self.config,
                cache_dir=cache_dir,
            )
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #         self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
        #         cache_dir=cache_dir,
        #     )
        
        # self.W = nn.Linear(768, 2)
        # self.num_classes = 2

    # def forward(self, input_ids, attention_mask, token_type_ids):
    #     h, _, attn = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    #     h_cls = h[:, 0]
    #     logits = self.W(h_cls)
    #     return logits, attn

    def forward(self, **inputs):
        return self.model(**inputs)

    def prepare_data(self):
        "Called to initialize data. Use the call to construct features"
        args = self.hparams
        processor = EndOfThoughtDataProcessor()
        self.labels = processor.get_labels()

        for mode in ["train", "val"]:
            cached_features_file = self._feature_file(mode)
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s", cached_features_file)
            else:
                logger.info("Creating features from dataset file at %s", args.data_dir)
                examples = (processor.get_train_examples(args.data_dir))

                features = convert_examples_to_features(
                    examples,
                    self.tokenizer,
                    max_length=args.max_seq_length,
                    label_list=self.labels,
                    output_mode=args.glue_output_mode,
                )
                print("Saving features into cached file")
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.get_dataloader("val", self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def train_dataloader(self):
        return self.train_loader

    def get_lr_scheduler(self):
        arg_to_scheduler = {
            "linear": get_linear_schedule_with_warmup,
            "cosine": get_cosine_schedule_with_warmup,
            "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
            "polynomial": get_polynomial_decay_schedule_with_warmup,
            # '': get_constant_schedule,             # not supported for now
            # '': get_constant_schedule_with_warmup, # not supported for now
        }
        arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
        arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"
        
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps()
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
            )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def test_epoch_end(self, outputs) -> dict:
        ret, predictions, targets = self._eval_end(outputs)
        logs = ret["log"]
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def _eval_end(self, outputs) -> tuple:
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu().item()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)

        if self.hparams.glue_output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.hparams.glue_output_mode == "regression":
            preds = np.squeeze(preds)

        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        results = {**{"val_loss": val_loss_mean}, **compute_metrics(preds, out_label_ids)}

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        print(f"ret, preds_list, out_label_list: {ret}, {preds_list}, {out_label_list}")
        return ret, preds_list, out_label_list

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        
        # Calculate dataset_size
        self.train_loader = self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)
        self.dataset_size = len(self.train_dataloader().dataset)

        total_steps = (self.dataset_size / effective_batch_size) * self.hparams.max_epochs
        print(f"@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"Total steps: {total_steps}")
        return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs

    def training_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        # if self.config.model_type != "distilbert":
        #     inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None
        inputs["token_type_ids"] = None

        outputs = self(**inputs)
        loss = outputs[0]

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None
        inputs["token_type_ids"] = None

        # print(f"Validation step: {inputs}")
        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def get_dataloader(self, mode: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        "Load datasets. Called after prepare data."

        # We test on dev set to compare to benchmarks without having to submit to GLUE server
        mode = "val" if mode == "test" else mode

        cached_features_file = self._feature_file(mode)
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        # all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if self.hparams.glue_output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif self.hparams.glue_output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        return DataLoader(
            TensorDataset(all_input_ids, all_attention_mask, all_labels),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=5
        )

    @utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = os.path.join(self.hparams.output_dir, "best_tfmr")
        self.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_model_specifi_args(parser, root_dir):
        parser.add_argument(
            "--output_dir",
            default="./results",
            type=str,
            required=True
        )
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=3, type=int)

class LoggingCallback(Callback):
    def on_batch_end(self, trainer, pl_module):
        lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
        lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_lr())}
        pl_module.logger.log_metrics(lrs)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule):
        rank_zero_info("***** Test results *****")
        metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))



if __name__ == "__main__":
    args = ClassificationConfig.from_json('./configs/classification_config.json').to_argparse()
    print(f"Args: {args.data_dir}")

    model = RoBertaFinetuner(args)
    
    model.prepare_data()

    # add custom checkpoints
    logging_callback = LoggingCallback()
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=args.output_dir, filename="checkpoint", mode="min", save_top_k=1
    )

    trainer = Trainer(
        accelerator='gpu',
        devices='1',
        max_epochs=10,
        callbacks=[checkpoint_callback]
        )

    # Prepare model for training
    seed_everything(args.seed)

    # init model
    odir = Path( args.output_dir)
    odir.mkdir(exist_ok=True)

    if args.do_train:
        trainer.fit(model)

    if args.do_predict:
        checkpoints = list(
            sorted(glob.glob(os.path.join(args.output_dir, "checkpoint=*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1])
        results = trainer.test(model)