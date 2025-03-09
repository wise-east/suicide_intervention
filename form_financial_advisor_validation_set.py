import pandas as pd 

data = pd.read_csv('financial_advisor_initial_llm_predictions.csv')
save_fp = 'financial_advisor_validation_set.csv'

financial_keywords = [
    "finance", 
    "financial", 
    "bank", 
    "investment", 
    "broker", 
    "account", 
    "loan", 
    "credit", 
    "wealth", 
    "insurance", 
    "tax", 
    "budget", 
    "mortgage", 
    "pension", 
    "bankruptcy", 
    "debt", 
    "agenc",
    "counselor",
]

positive_samples = [] 
negative_samples = [] 

target_size = 100 
pred_column_name = 'model_preds'
report_column_name = 'Report'
for index, row in data.iterrows():  
    # skip if Report is empty or not a string 
    if not isinstance(row[report_column_name], str): 
        continue 
    
    if eval(row[pred_column_name]) == [1]: 
        if len(positive_samples) < target_size and any(keyword in row[report_column_name].lower() for keyword in financial_keywords): 
            positive_samples.append(row)
    else: 
        if any(keyword in row[report_column_name].lower() for keyword in financial_keywords): 
            if len(negative_samples) < target_size: 
                negative_samples.append(row)
                
# go through samples again to add negatives that do not necessarily contain financial keywords 
for index, row in data.iterrows(): 
    if eval(row[pred_column_name]) == [0]: 
        if len(negative_samples) < target_size: 
            negative_samples.append(row)
            
print(f'Positive samples: {len(positive_samples)}')
# print(f'Negative samples: {len(negative_samples)}')
            
# concatenate positive and negative samples and save to csv 
# validation_set = pd.concat([pd.DataFrame(positive_samples), pd.DataFrame(negative_samples)])
validation_set = pd.DataFrame(positive_samples)
validation_set.to_csv(save_fp, index=False)

print(f'Saved validation set to {save_fp}')