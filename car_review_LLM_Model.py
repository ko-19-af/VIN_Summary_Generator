import pandas as pd
import json
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding

# pip install sentencepiece
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# pip install hf_xet
# pip install 'accelerate>=0.26.0'

model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)

def load_initial_dataset():
    ds = pd.read_csv('model_data.csv') # read in the csv with the data
    return ds


def preprocess_dataset(ds):
    # Loading the dataset
    dataset = Dataset.from_pandas(ds)

    # encoding the 'risk_rating' column
    dataset = dataset.class_encode_column("risk_rating")

    # Splitting the dataset into training and testing sets
    # stratify_by_column = avoid too many reviews of the same type
    dataset = dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column="risk_rating")

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    return train_dataset, test_dataset


def data_remap_for_training(examples):
    # remap plain-text data to token ids

    # since T5 requires an instruction (such as "summarize/translate"), the "review" instruction is used here
    #(this means appending "review:" to the beginning of the prompt)
    examples['prompt'] = [f"review: {prompt}" for prompt in zip(examples['prompt'])]
    examples['response'] = [f"{response}" for response in examples['response']]

    # turn strings into input ids
    inputs = tokenizer(examples['prompt'], padding='max_length', truncation=True, max_length=128)
    targets = tokenizer(examples['response'], padding='max_length', truncation=True, max_length=128)

    # set padding to -100, so that the paddings are ignored when calculating loss
    target_input_ids = []
    for ids in targets['input_ids']:
        target_input_ids.append([id if id != tokenizer.pad_token_id else -100 for id in ids])

    inputs.update({'labels': target_input_ids})
    return inputs


def main():
    # Prepare data for training
    ds = load_initial_dataset()
    train_dataset, test_dataset = preprocess_dataset(ds)
    train_dataset = train_dataset.map(data_remap_for_training, batched=True)
    test_dataset = test_dataset.map(data_remap_for_training, batched=True)

    # Load the T5 tokenizer and model, data collator
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    for i in range(5):
        random_products = test_dataset.shuffle().select(range(10))['risk_rating']
        print("-" * 32)
        print(random_products)
        print("-" * 32)

    # Configure training arguments
    model_output_folder = "./models/t5_fine_tuned_reviews"
    training_args = TrainingArguments(
        output_dir=model_output_folder,
        num_train_epochs=3,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        save_strategy='epoch',
    )

    # Start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(model_output_folder)

if __name__ == "__main__":
    main()