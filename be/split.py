import pandas as pd
from sklearn.model_selection import train_test_split

# Load the metadata
df = pd.read_csv("fitmood_metadata.csv")

# Split into 80% training and 20% validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save to CSVs
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)

print("✅ Train/Validation split created successfully!")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")