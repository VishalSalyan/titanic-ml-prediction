
import pandas as pd

def clean_data(input_path):

    df = pd.read_csv(input_path)

    # -------------------------
    # Step 1: Basic columns first
    # -------------------------
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Name']].copy()

    # -------------------------
    # Step 2: Handle missing values
    # -------------------------
    df['Age'] = df['Age'].fillna(df['Age'].mean())

    # -------------------------
    # Step 3: Encode Sex
    # -------------------------
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0}).astype(int)

    # -------------------------
    # Step 4: Feature Engineering
    # -------------------------

    # FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # IsAlone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],
        'Rare'
    )
    df['Title'] = df['Title'].map({
        'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5
    }).fillna(0)

    # AgeGroup
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,20,40,60,80], labels=[0,1,2,3,4])

    # FareBin
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=[0,1,2,3])


    df['AgeGroup'] = df['AgeGroup'].astype(int)
    df['FareBin'] = df['FareBin'].astype(int)
    
    # -------------------------
    # Step 5: Final column selection
    # -------------------------
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare',
             'FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareBin']]

    # -------------------------
    # Save
    # -------------------------
    df.to_csv('data/clean_titanic.csv', index=False)


    # -------------------------------------
    # Separate features (X) and target (y)
    # -------------------------------------
    
    X = df.drop("Survived", axis = 1)
    y = df["Survived"]
    
    return X, y