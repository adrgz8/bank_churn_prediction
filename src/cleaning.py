
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def clean_data(path, scaler, test=False, verbose=True):
    df = pd.read_csv(path)
    if verbose:
        print('Shape of data is', df.shape)

    # Removing duplicates
    if verbose:
        print('Removing duplicate values and unnecessary columns...')

    dropped_cols = ['id', 'CustomerId', 'Surname']
    sel_dup = df.drop(dropped_cols, axis=1).duplicated().value_counts()
    if verbose:
        print('Duplicate values found: ', sel_dup.iloc[1])

    df = df.drop(['id', 'CustomerId', 'Surname'], axis=1)
    df = df.drop_duplicates()
    if verbose:
        print("Shape after removing duplicates is", df.shape)

    if verbose:
        if df.isnull().sum().sum() == 0:
            print('No null values found')
        else:
            print('Please check null values')
            print(df.isnull().sum())
            return

    # One-hot encoding gender
    if verbose:
        print('Converting categorical variables to One-hot encoding')
    gender_dummies = pd.get_dummies(
        df['Gender'],
        drop_first=False,
        dtype=int
        )
    df = df.join(gender_dummies)

    # One-hot encoding countries
    country_dummies = pd.get_dummies(
        df['Geography'],
        drop_first=False,
        dtype=int
        )
    df = df.join(country_dummies)

    # One-hot encoding Tenure and num of products
    tenure_options = pd.get_dummies(
        df['Tenure'],
        drop_first=True,
        dtype=int
        )
    tenure_cols = [f'Tenure_{i}' for i in tenure_options.columns]
    df[tenure_cols] = tenure_options

    num_prod_options = pd.get_dummies(
        df['NumOfProducts'],
        drop_first=False,
        dtype=int
        )
    num_prod_cols = [f'NumProds_{i}' for i in num_prod_options.columns]
    df[num_prod_cols] = num_prod_options

    # Adding boolean balance column
    if verbose:
        print('Adding boolean balance variable')
    df['zero_balance'] = (df['Balance'] == 0).astype(int)

    # Creating df for heatmap
    if not test:
        df_hm = df[[
            'CreditScore',
            'France',
            'Germany',
            'Spain',
            'Tenure',
            'NumOfProducts',
            'Male',
            'Age',
            'Balance',
            'HasCrCard',
            'IsActiveMember',
            'EstimatedSalary',
            'Exited']]

    # Removing redundant columns
    if verbose:
        print('Removing redundant columns...')
    red_cols = ['Geography',
                'Tenure',
                'NumOfProducts',
                'Female',
                'Gender']
    df = df.drop(red_cols, axis=1)

    # Making int columns
    int_cols = ['IsActiveMember', 'HasCrCard', 'Age']
    df[int_cols] = df[int_cols].astype(int)

    # Removing columns for test
    if test:
        df = df.drop(['id', 'CustomerId', 'Surname'], axis=1)
        trans_cols = ['CreditScore', 'Age', 'EstimatedSalary', 'Balance']
        df[trans_cols] = scaler.fit_transform(df[trans_cols])
        return df

    return df, df_hm


if __name__ == '__main__':
    path_train_data = 'data/raw/train.csv'
    data_scaler = MinMaxScaler(feature_range=(-1, 1))
    clean_data(
        path=path_train_data,
        scaler=data_scaler,
        test=False,
        verbose=True)
