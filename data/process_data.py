import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load csv files and merge them into one dataframe.
    
    ARGUMENTS:
    messages_filepath - csv file with message data
    categories_filepath - csv file with category data
    
    RETURN VALUE:
    df - merged dataframe
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on='id', how='inner')
    
    return df


def clean_data(df):
    """
    Clean merged dataframe.
    
    ARGUMENTS:
    df - merged dataframe
    
    RETURN VALUE:
    df - cleaned dataframe
    """
    
    categories = df.categories.str.split(';', expand=True)
    category_colnames = categories.loc[0].apply(lambda x: x[:-2])    
    df[category_colnames] = categories
        
    for col in category_colnames:
        df[col] = df[col].apply(lambda x: x[-1:])
        df[col] = df[col].astype(int)
    
    df.drop('categories', axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Due to error in original dataset, according to documentation
    # 1 = 'disaster-related' | 2 = 'not disaster-related'
    # Change 2's to 0's
    df.loc[df[df['related'] == 2].index, 'related'] = 0
            
    return df


def save_data(df, database_filename):
    """
    Save cleaned dataframe as table to SQLite database.
    
    ARGUMENTS:
    df - cleaned dataframe
    database_filename - filename of the SQLite database
    
    RETURN VALUE: 
    None
    """
    
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('categorized_messages', engine, index=False, if_exists='replace')  
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()