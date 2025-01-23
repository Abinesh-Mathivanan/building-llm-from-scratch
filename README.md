# building-llm-from-scratch

Implementation of the book, ["Building a Large Language Model (From Scratch")](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka.


To understand this project, read the files and folders in the following order:

### Chapters

1. **Chapter 2 - Encoders**: This directory contains the code related to encoders.
    - **simpleTokenizer.py**: Implements a basic tokenizer with a predefined vocabulary. It includes encoding and decoding functionalities.
    - **bytePair.py**: Implements Byte Pair Encoding (BPE) using the `tiktoken` library for tokenizing text data into subword units. It also demonstrates how to create input-output pairs for language modeling.
    - **tokenEmbedding.py**: Focuses on embedding tokens, representing them as vectors in a high-dimensional space. It demonstrates how to use PyTorch's `nn.Embedding` layer for token and positional embeddings.
    - **pyTorchTokenizer.py**: Builds a PyTorch DataLoader for processing text data. It uses the tokenizer from `bytePair.py` to create input and output token IDs for training.
    
  

2. **Chapter 3 - Self Attention**: This directory contains the code for implementing different self-attention mechanisms.
    - **simpleAttention.py**: Implements a basic attention mechanism by calculating attention scores and context vectors.
    - **trainableAttention.py**: Introduces the concept of weight matrices (Query, Key, Value) to efficiently compute context vectors.
    - **pyTorchAttention.py**: Implements self-attention using PyTorch's linear layers and a class-based approach.
    - **casualAttention.py**: Implements casual attention, a technique used in autoregressive models to prevent the model from attending to future tokens.
    - **multiHeadAttention.py**: Implements multi-head attention, where the input is processed through multiple attention heads in parallel.
  
3. **Chapter 4 - LLM Architecture**: This directory contains the implementation of core architecture.
   - **beensGPT.py**: Main architecture file 
   - **feedForward.py**: FFNN implementation 
   - **layerNorm.py**: Layer Normalization implementation 
   - **multiHeadAttention.py**: Implemented Multi head attention mechanism for beensGPT
   - **shortcutConnection.py**: Implemented shortcut connection mechanism of the GPT 
   - **SimpleText.py**: practice implementation of small GPT 
  
    
  

