# ROBIN-Protocol

**Research-Oriented Biomedical Intelligence Network Protocol**

A comprehensive framework for biomedical data analysis, featuring advanced research models, evidence collection, and batch processing capabilities.

## Overview

ROBIN-Protocol is a multi-system architecture designed for rigorous biomedical research and data analysis:

- **System 2**: Core research and analysis components
- **System 3**: Data organization and batch processing infrastructure

## Features

### System 2 Components

- **Orchestrator CLI**: Command-line interface for managing System 2 operations
- **Omni Audit**: Comprehensive data quality and model validation auditing
- **Research Models**: Research-grade machine learning models (Decision Trees, Random Forest, Gradient Boosting)
- **Results Engine**: Flexible results processing and export (JSON, CSV)
- **Evidence Pro**: Professional evidence collection and validation system

### System 3 Components

- **Chronos Batch**: Time-series batch processing for large-scale data
- **Master Index**: Efficient data organization and retrieval system

## Installation

### Using Conda

```bash
conda env create -f environment.yml
conda activate robin-protocol
```

### Manual Installation

```bash
pip install numpy pandas scikit-learn matplotlib seaborn pyyaml pytest
```

## Quick Start

### Basic Usage

```python
# Initialize the orchestrator
from src.system2_orchestrator_cli import main as orchestrator_main

# Run the orchestrator CLI
orchestrator_main()
```

### Using Research Models

```python
from src.system2_research_models import ResearchModels
import numpy as np

# Initialize research models
research = ResearchModels()

# Train a model
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, 100)
model = research.train_model('random_forest', X_train, y_train)
```

### Evidence Collection

```python
from src.system2_evidence_pro import EvidencePro

# Initialize Evidence Pro
evidence = EvidencePro()

# Collect evidence
evidence.collect_evidence('study_1', 'clinical_trial', {'outcome': 'positive'})

# Analyze evidence
analysis = evidence.analyze_evidence()
print(analysis)
```

## Configuration

Edit `config.yaml` to customize system behavior:

```yaml
system2:
  orchestrator:
    enabled: true
    log_level: INFO
  research_models:
    model_types:
      - decision_tree
      - random_forest
      - gradient_boosting
```

## Examples

Two example datasets are included:

1. **Diabetes UCI** (`examples/diabetes_uci/`): Pima Indians Diabetes dataset examples
2. **BRFSS Heart** (`examples/brfss_heart/`): Behavioral Risk Factor Surveillance System heart disease examples

## Project Structure

```
ROBIN-Protocol/
├── README.md
├── LICENSE
├── environment.yml
├── config.yaml
├── src/
│   ├── system2_orchestrator_cli.py
│   ├── system2_omni_audit.py
│   ├── system2_research_models.py
│   ├── system2_results_engine.py
│   ├── system2_evidence_pro.py
│   ├── system3_chronos_batch.py
│   └── system3_master_index.py
└── examples/
    ├── diabetes_uci/
    └── brfss_heart/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and support, please open an issue on the GitHub repository.