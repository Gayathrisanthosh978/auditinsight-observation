# Audit Insights

This repository provides tools for generating audit insights and observations from scenario-based data using advanced language models and data analysis techniques.

## Contents
- `audit_insights.py`: Main module for generating audit insights.
- `observation.py`: Module for generating audit observations.
- `utils.py`: Utility functions for LLM-based sentence generation.
- `config.py`: Configuration for models and database connections.
- `logger.py`: Logging setup.
- `main.py`: Entry point to run audit or observation workflows.

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Setup
1. Clone this repository:
   ```sh
   git clone <repo-url>
   cd audit-insights
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up environment variables as needed (see `config.py`).

## Usage
Run either audit insights or observation generation:

```sh
python main.py audit
python main.py observation
```

You can also import the modules and use their functions in your own scripts.

## License
MIT
