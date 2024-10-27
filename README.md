# Confidence-Weighted PMI (CW-PMI)

A novel approach to calculating Pointwise Mutual Information (PMI) that combines information theory, Bayesian statistics, and error propagation analysis. This implementation addresses traditional PMI calculation challenges through theoretically grounded smoothing techniques that properly account for uncertainty in both observed and expected probabilities.

## Key Features

- Entropy-based degrees of freedom calculation
- Proper error propagation in probability products
- Units-consistent smoothing formulation 
- Confidence-weighted probability adjustment
- Robust handling of both rare and common n-grams

## Installation

```bash
# Clone the repository
git clone https://github.com/s243a/CW_PMI.git
cd CW_PMI

# Install required packages
pip install numpy scipy nltk tqdm google-colab
```

## Usage

Basic usage example:

```python
from Confidence_Weighted_PMI import EnhancedEfficientTfidfVectorizer

# Initialize the vectorizer
vectorizer = EnhancedEfficientTfidfVectorizer(
    max_features=10000, 
    ngram_range=(1, 3),
    pmi_threshold=0, 
    store_results=True
)

# Fit on your text data
vectorizer.fit(documents)

# Get PMI scores for specific n-grams
pmi_score = vectorizer._calculate_pmi_with_t_score("your phrase here")
```

## Documentation

The theoretical foundation and implementation details are described in the accompanying paper `document.tex`. Key topics covered include:

- Information theory connections
- Statistical significance testing
- Error propagation methodology
- Entropy-based calculations
- Confidence interval derivations

## Citation

If you use this work in your research, please cite:

```bibtex
@article{cwpmi2024,
  title={Confidence-Weighted PMI: An Information-Theoretic Approach with Proper Error Propagation},
  author={Creighton, John},
  year={2024}
}
```

## License

To be determined.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

https://twitter.com/s243a_PT