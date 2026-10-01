import pandas as pd

# Load only required columns (faster)
df = pd.read_csv(
    "onlinefraud.csv",
    usecols=['step','type','amount','oldbalanceOrg','newbalanceOrig',
             'oldbalanceDest','newbalanceDest','isFraud']
)

# Get fraud and non-fraud
fraud = df[df['isFraud'] == 1]
non_fraud = df[df['isFraud'] == 0].sample(n=65000, random_state=42)

# Combine
final_df = pd.concat([fraud, non_fraud])

# Shuffle
final_df = final_df.sample(frac=1, random_state=42)

# Save
final_df.to_csv("final_fraud_dataset_73213.csv", index=False)

print("✅ DONE - File saved as final_fraud_dataset_73213.csv")
