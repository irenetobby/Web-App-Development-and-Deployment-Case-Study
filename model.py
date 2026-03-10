# Load and prepare data
df = pd.read_csv('data/drinks.csv')

# Drop the Unnamed: 0 column if it's just an index
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Handle missing values in features (X) and target (y) separately
print("\n" + "="*60)
print("DATA CLEANING")
print("="*60)

# Check missing values in target
target_col = 'total_litres_of_pure_alcohol'
if df[target_col].isnull().sum() > 0:
    # Option 1: Drop rows with missing target values (recommended for small dataset)
    df = df.dropna(subset=[target_col])
    print(f"Dropped {df[target_col].isnull().sum()} rows with missing target values")
    
    # Option 2: Alternatively, fill with median (use if you don't want to lose data)
    # median_val = df[target_col].median()
    # df[target_col].fillna(median_val, inplace=True)
    # print(f"Filled {df[target_col].isnull().sum()} missing values in target with median: {median_val}")

# Handle missing values in features
feature_cols = ['beer_servings', 'spirit_servings', 'wine_servings']
for col in feature_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Filled {df[col].isnull().sum()} missing values in {col} with median: {median_val}")

# Verify no missing values remain
print("\nMissing values after cleaning:")
print(df.isnull().sum())
print()

# Prepare features and target
X = df[feature_cols]  # or include other features as needed
y = df[target_col]

# Split the data (now with no NaN values)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print()
