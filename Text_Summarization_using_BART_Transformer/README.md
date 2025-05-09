#   Text Summarization using BART Transformer

This project demonstrates text summarization using the BART Transformer model. It covers both implementations: using the pre-trained model directly and fine-tuning it on the dialogsum dataset for improved performance.

##   Overview

The project focuses on summarizing text using the BART (Bidirectional and Auto-Regressive Transformer) model. It includes implementations both with and without fine-tuning the model. The dataset used is the dialog sum dataset from Hugging Face.

##   Key Features

* **Text Summarization:** Summarizes text using the BART Transformer model.
* **Fine-tuning:** Demonstrates how to fine-tune the BART model for improved summarization performance.
* **Dataset:** Utilizes the dialog sum dataset[cite: 1].
* **Implementation:** Provides code for loading the dataset, preprocessing the data, and training the model.

##   Setup and Installation

1.  **Install Dependencies:**

    ```bash
    pip install datasets transformers
    ```
2.  **Load the Dataset:**

    ```python
    from datasets import load_dataset

    ds = load_dataset("knkarthick/dialogsum")
    ```
3.  **Preprocess the Data:**

    ```python
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    def preprocessor_function(batch):
        source = batch['dialogue']
        target = batch['summary']
        #   ... (rest of the preprocessing code)
    df_source = ds.map(preprocessor_function, batched=True)
    ```

## Usage

### Without Fine-tuning

```python
from transformers import pipeline

pipe = pipeline("summarization", model="facebook/bart-large-cnn")

article_1 = ds['train'][1]['dialogue']
summary = pipe(article_1, max_length=20, min_length=18, do_sample=False)
print(summary)
```

### With Fine-tuning
1. **Load Model and Tokenizer:**
  
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
```

2. **Train the Model:**
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="/content",
    per_device_train_batch_size=8,
    num_train_epochs=2,
    #   ... other arguments
)
    
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=df_source['train'],
    eval_dataset=df_source['test']
    )
    
trainer.train()
```

## Example

```python
def summarize(blog_post):
    #   ... (summarization code)

blog_post = """
#   The Future of Remote Work: Balancing Flexibility and Collaboration
... (rest of the blog post)
"""

summary = summarize(blog_post)
print(f"Summary: {summary}")
```


## Credits
  
[InsightsByRish's video](https://youtu.be/6pkqIVfr_VE?si=UbcFYaPKThocy0ZU) for the original tutorial.
