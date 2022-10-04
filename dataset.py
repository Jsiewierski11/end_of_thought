import os
import csv
from transformers import DataProcessor, InputExample

# from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

# class EndOfThoughtDataModule(LightningDataModule):
#     def __init__(self, data_dir="./data", batch_size=32):
#         super(EndOfThoughtDataModule).__init__()

#         self.data_dir = data_dir
#         self.batch_size = 32

#     def prepare_data(self):
#         """
#         Empty prepare_data method left in intentionally. 
#         https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
#         """
#         pass


class EndOfThoughtDataProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["text"].decode("utf-8"),  # tokenized text
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        print("Get train examples for DementiaTalkBankProcessor\n")
        train_path = os.path.join(data_dir, "train.tsv")
        print(f"train_path: {train_path}")
        return self._create_examples(self._read_tsv(train_path), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        print("Get train examples for DementiaTalkBankProcessor\n")
        test_path = os.path.join(data_dir, "test.tsv")
        print(f"train_path: {test_path}")
        return self._create_examples(self._read_tsv(test_path), "train")

    def get_val_examples(self, data_dir):
        """See base class."""
        print("Get train examples for DementiaTalkBankProcessor\n")
        val_path = os.path.join(data_dir, "val.tsv")
        print(f"train_path: {val_path}")
        return self._create_examples(self._read_tsv(val_path), "val")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        guid = 0
        for (i, line) in enumerate(lines):
            text_a = line[0]
            label = None if set_type.startswith("test") else line[1]
            # label = self.format(label)
            if label == "":
                label = "0"
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
            guid += 1
        return examples

    def format(self, x):
        return "_".join(x.lower().split())

    def get_labels(self):
        """See base class."""
        labels = ["1", "0"]
        return labels
        # return [self.format(x) for x in labels]

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a comma separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter=",", quotechar=quotechar))

if __name__ == '__main__':
    processor = EndOfThoughtDataProcessor()
    examples = processor.get_train_examples('./data')

    print(examples)