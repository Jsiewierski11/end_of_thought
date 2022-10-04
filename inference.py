import os
import glob
import torch
from torch.nn import Softmax

from train_pipeline import RoBertaFinetuner
from utils.config import ClassificationConfig

if __name__ == "__main__":
    args = ClassificationConfig.from_json('./configs/classification_config.json').to_argparse()
    model = RoBertaFinetuner(args)

    model.prepare_data()
    print("Prepared data")

    checkpoints = list(
        sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-v1.ckpt"),
               recursive=True))
    )
    print(f"Checkpoints: {checkpoints}")
    model = model.load_from_checkpoint(checkpoints[-1]).eval()

    INPUT = "Hmm that's a tough question. Let me think."
    # input_tensor = torch.tensor([INPUT])
    inputs = model.tokenizer([INPUT])

    inputs['input_ids'] = torch.tensor(inputs['input_ids'])
    inputs['attention_mask'] = torch.tensor(inputs['attention_mask'])
    inputs['token_type_ids'] = torch.tensor(inputs['token_type_ids'])
    print(f"Inputs: {type(inputs)}, {inputs}")
    outputs = model(**inputs)
    print(f"Model outputs: {outputs}")
    sm = Softmax()
    preds = sm(outputs[0])
    print(preds)