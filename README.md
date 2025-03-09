# Suicide Intervention Project

Overview: identify underexplored suicide intervention opportunities by identifying actors involved in NVDRS narratives. 

### Create a new environment

```bash
conda create -n suicide_intervention python=3.11
conda activate suicide_intervention
pip install -r requirements.txt
```

#### Download nltk data for sentence tokenization
```bash
python -m nltk.downloader punkt_tab
```

### Main commands 

#### Create a validation set for financial advisors that will be validated by humans 
```bash 
python form_financial_advisor_validation_set.py
```

#### Sample code for selecting diverse narratives based on coverage

```bash
python coverage_sampling.py
```

