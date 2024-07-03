# Creating an LLM-based conversation assistant for hotel accommodation booking

This repository contains the research paper and code for a hotel booking assistant developed using fine-tuned Large Language Models (LLMs).

## Repository Structure

- `src/`: 
  - `dataset/`: Scripts for generating, saving, loading and preprocessing the synthetic dataset
  - `model/`: Code for LLM loading, fine-tuning and inference
  - `evaluation/`: Scripts for automatic evaluation
  - `assistant/`: Hotel booking assistant itself
- `data/`: Data and templates used for dataset creation
- `notebooks/`: Notebooks with usage examples
- `main.py`: Script providing console interface for the assistant
- `paper.pdf`: The PDF of the research paper

## Usage
Clone the repository and install requirements:
```bash
git clone https://github.com/anna-marshalova/hotel-booking-assistant
cd hotel-booking-assistant
pip install -r requirements.txt
```

Run main.py
```bash
python main.py
```