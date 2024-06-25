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

Tokenization: Converts input text into tokens using OpenAI's (Tiktoken)[https://github.com/openai/tiktoken].
Embedding: Maps tokens to dense vectors of dimension d_model.
Decoder Layers: Stacked layers of self-attention and feed-forward networks.
Output Layer: Projects decoder outputs to vocabulary logits, followed by a softmax layer to obtain probabilities.

### Key Hyperparameters
d_model: Dimension of the embedding and hidden states.
num_heads: Number of attention heads.
d_ff: Dimension of the feed-forward network, hardcoded to 4 times the embedding dimension.

Total: 15.2 M Trainable params


  | Name                       | Type               | Params | Mode 
--------------------------------------------------------------------------
0 | embedding_table            | Embedding          | 5.0 M  | train
1 | positional_encodings_table | Embedding          | 5.0 M  | train
2 | attention_block            | MultiHeadAttention | 40.4 K | train
3 | feed_forward               | FeedForwardNetwork | 80.5 K | train
4 | linear_head                | Linear             | 5.1 M  | train
5 | loss_func                  | CrossEntropyLoss   | 0      | train
--------------------------------------------------------------------------