# Transformer Sequence-to-Sequence Translation System

A PyTorch implementation of transformer-based sequence-to-sequence models for translation tasks, featuring both autoregressive and non-autoregressive variants with distributed training support.

## Features

- **Autoregressive Transformer**: Standard transformer decoder with teacher forcing
- **Non-Autoregressive Transformer**: Parallel decoding with length prediction
- **Distributed Training**: Multi-GPU support with PyTorch DDP
- **Custom Group Attention**: Enhanced attention mechanism with grouping
- **Comprehensive Evaluation**: Accuracy metrics and visualization tools

## Project Structure

```
├── main.py                           # Main training script (autoregressive)
├── non_training.py                   # Non-autoregressive training
├── main_gpu.py                       # Distributed GPU training
├── transformer.py                    # Autoregressive transformer implementation
├── non_autoregressive_transformer.py # Non-autoregressive transformer
├── train_greedy_with_bica.py        # Alternative training with BICA attention
├── data.py                          # Data preprocessing utilities
├── util.py                          # Utility functions and data management
├── tree.py                          # Tree structure for hierarchical outputs
├── sample.py                        # Model sampling and inference
└── data/                            # Data directory
    ├── train.txt                    # Training data
    ├── test.txt                     # Test data
    ├── vocab.q.txt                  # Source vocabulary
    └── vocab.f.txt                  # Target vocabulary
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/transformer-seq2seq.git
cd transformer-seq2seq
```

2. Install required dependencies:
```bash
pip install torch torchvision matplotlib numpy
```

3. Create data directory and add your datasets:
```bash
mkdir -p data
# Add your train.txt, test.txt, vocab.q.txt, vocab.f.txt files
```

## Data Format

### Training Data (train.txt, test.txt)
Each line should contain source and target sequences separated by a tab:
```
source sentence 1	target sentence 1
source sentence 2	target sentence 2
```

### Vocabulary Files (vocab.q.txt, vocab.f.txt)
Each line contains a token followed by its frequency:
```
token1	frequency1
token2	frequency2
```

## Usage

### 1. Autoregressive Training

Basic training with default settings:
```bash
python main.py
```

### 2. Non-Autoregressive Training

Train the non-autoregressive model:
```bash
python non_training.py
```

### 3. Distributed Training (Multi-GPU)

For distributed training across multiple GPUs:
```bash
python main_gpu.py
```

### 4. Data Preprocessing

Process your data for tree-based training:
```bash
python data.py -data_dir ./data/ -train train -test test
```

### 5. Model Sampling

Generate translations using a trained model:
```bash
python sample.py -model checkpoint_dir/model_seq2seq -data_dir ./data/
```

## Configuration

### Model Parameters

The main configuration is defined in the `Config` class:

```python
class Config:
    BATCH_SIZE = 1
    EMBED_SIZE = 1024
    NUM_ENC_LAYERS = 5
    NUM_DEC_LAYERS = 5
    NUM_HEADS = 8
    DROPOUT = 0.4
    LEARNING_RATE = 0.000009
    NUM_EPOCHS = 700
```

### Key Hyperparameters

- **EMBED_SIZE**: Embedding dimension (default: 1024)
- **NUM_ENC_LAYERS**: Number of encoder layers (default: 5)
- **NUM_DEC_LAYERS**: Number of decoder layers (default: 5)
- **NUM_HEADS**: Number of attention heads (default: 8)
- **LEARNING_RATE**: Learning rate (default: 9e-6)
- **DROPOUT**: Dropout rate (default: 0.4)

## Model Architecture

### Autoregressive Transformer
- Standard encoder-decoder architecture
- Multi-head self-attention and cross-attention
- Position-wise feed-forward networks
- Custom group attention mechanism

### Non-Autoregressive Transformer
- Parallel decoding for faster inference
- Length prediction module
- Position-based queries for decoding
- Eliminates sequential dependency in generation

### Special Features

1. **Group Attention**: Custom attention mechanism that models local dependencies
2. **Teacher Forcing**: During training, uses ground truth tokens as input
3. **Gradient Clipping**: Prevents gradient explosion
4. **Learning Rate Scheduling**: Adaptive learning rate adjustment

## Training Process

1. **Data Loading**: Sequences are padded and converted to tensors
2. **Teacher Forcing**: Ground truth tokens guide the training process
3. **Loss Calculation**: Negative log-likelihood loss with padding mask
4. **Validation**: Accuracy calculated on test set after each epoch
5. **Model Saving**: Best model saved based on validation accuracy

## Evaluation Metrics

- **Accuracy**: Exact match accuracy between predicted and target sequences
- **Training Loss**: Cross-entropy loss during training
- **Validation Accuracy**: Performance on held-out test set

## Output

The training process generates:
- `best_model.pt`: Best performing model checkpoint
- `train_losses.png`: Training loss visualization
- `accuracies.png`: Validation accuracy plot
- Console logs with epoch-wise progress

## Advanced Features

### Distributed Training

The `main_gpu.py` script supports multi-GPU training:
- Automatic GPU detection
- Data parallel training with DDP
- Gradient synchronization across GPUs
- Enhanced performance monitoring

### Tree-based Processing

The system supports hierarchical output structures:
- Tree representation of sequences
- Recursive processing for nested structures
- Normalization and accuracy computation for trees

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model dimensions
2. **Convergence Issues**: Adjust learning rate or add regularization
3. **Data Format**: Ensure proper tab-separated format for training data

### Performance Tips

1. **Use GPU**: Enable CUDA for faster training
2. **Batch Size**: Increase batch size if memory allows
3. **Gradient Clipping**: Helps with training stability
4. **Learning Rate**: Start with 1e-4 and adjust based on convergence

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{transformer-seq2seq,
  title={Transformer Sequence-to-Sequence Translation System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/transformer-seq2seq}
}
```

## Acknowledgments

- PyTorch team for the deep learning framework
- Attention mechanism inspired by "Attention Is All You Need"
- Non-autoregressive generation techniques from recent literature
