# Article Dynamo

Article Dynamo is a text generator based on a decoder-only Transformer architecture, trained on a collection of my own articles published on Medium. The model is designed to generate contextually relevant text in the style and content of the provided articles.


## Text Generation

To generate text using the pretrained model, you can use the generate_text.py script:

```bash
python generate_text.py --input "Your initial text here"
```

This will output the generated text based on the initial input provided.

## Model Architecture

The model is a decoder-only Transformer with the following key components:

- Tokenization: Converts input text into tokens using OpenAI's (Tiktoken)[https://github.com/openai/tiktoken].
- Embedding: Maps tokens to dense vectors of dimension d_model.
- Decoder Layers: Stacked layers of self-attention and feed-forward networks.
- Output Layer: Projects decoder outputs to vocabulary logits, followed by a softmax layer to obtain probabilities.

### Key Hyperparameters
- block_size: Length of one sequence.
- d_model (embedding_dim): Dimension of the embedding and hidden states.
- num_heads (heads): Number of attention heads.
- d_k (head_size): Dimension of one attention head.
- d_ff: Dimension of the feed-forward network, hardcoded to 4 times the embedding dimension.


```bash
  | Name                       | Type               | Params | Mode 
--------------------------------------------------------------------------
0 | embedding_table            | Embedding          | 5.0 M  | train
1 | positional_encodings_table | Embedding          | 5.0 M  | train
2 | attention_block            | MultiHeadAttention | 40.4 K | train
3 | feed_forward               | FeedForwardNetwork | 80.5 K | train
4 | linear_head                | Linear             | 5.1 M  | train
5 | loss_func                  | CrossEntropyLoss   | 0      | train
--------------------------------------------------------------------------

```

Total: 15.2 M Trainable params

### Training results

Results using this configuration:

```bash
data:
  train: 
    path: data/train.txt
    max_samples: 100
  val: 
    path: data/val.txt
    max_samples: 100

model:
  embedding_dim: 100
  heads: 2
  head_size: 50
  block_size: 8

dataloader:
  batch_size: 2
  shuffle: True
  num_workers: 0
  #prefetch_factor: 2 # Should be >0 in case multiprocessing is enabled, i.e. num_workers > 0
  drop_last: False

trainer:
  max_epochs: 100
  num_sanity_val_steps: 1
  log_every_n_steps: 25
```


!(Training vs validation loss)[assets/loss.jpg]