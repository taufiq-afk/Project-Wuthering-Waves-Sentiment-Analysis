#!/usr/bin/env python3
"""
Wuthering Waves Sentiment Analysis - Data Preprocessing
=====================================================
Menggabungkan dan membersihkan data dari:
1. Google Play Store (5k reviews)
2. App Store (3k reviews) 
3. Reddit (438+ posts)

Total target: 8,438+ data points untuk supervised learning
"""

import pandas as pd
import numpy as np
import re
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class WutheringWavesPreprocessor:
    def __init__(self):
        """
        Initialize preprocessor untuk Wuthering Waves sentiment data
        """
        self.raw_data_path = 'data/raw/'
        self.processed_data_path = 'data/processed/'
        
        # Create directories if not exist
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        self.combined_data = None
        print("🔧 Wuthering Waves Data Preprocessor initialized")
    
    def load_google_play_data(self):
        """
        Load dan standardisasi data Google Play Store
        """
        print("📱 Loading Google Play Store data...")
        
        try:
            # Find latest Google Play file
            google_files = glob.glob(f"{self.raw_data_path}*google_play*.csv")
            if not google_files:
                google_files = glob.glob(f"{self.raw_data_path}reviews_data.csv")
            
            if not google_files:
                print("❌ Google Play data not found!")
                return pd.DataFrame()
            
            latest_file = max(google_files, key=os.path.getctime)
            df = pd.read_csv(latest_file)
            
            print(f"✅ Loaded {len(df)} Google Play reviews from: {latest_file}")
            
            # Standardize columns
            df_standard = pd.DataFrame({
                'review_id': df.get('review_id', df.index).astype(str),
                'source': 'google_play',
                'title': df.get('title', ''),
                'text': df.get('content', df.get('text', df.get('review_text', ''))),
                'rating': df.get('score', df.get('rating', 3)).astype(float),
                'user_name': df.get('userName', df.get('user_name', 'Anonymous')),
                'date': df.get('at', df.get('date', pd.Timestamp.now())),
                'helpful_count': df.get('thumbsUpCount', df.get('helpful_count', 0))
            })
            
            return df_standard
            
        except Exception as e:
            print(f"❌ Error loading Google Play data: {str(e)}")
            return pd.DataFrame()
    
    def load_app_store_data(self):
        """
        Load dan standardisasi data App Store
        """
        print("🍎 Loading App Store data...")
        
        try:
            # Find latest App Store file
            app_store_files = glob.glob(f"{self.raw_data_path}*app_store*.csv")
            
            if not app_store_files:
                print("❌ App Store data not found!")
                return pd.DataFrame()
            
            latest_file = max(app_store_files, key=os.path.getctime)
            df = pd.read_csv(latest_file)
            
            print(f"✅ Loaded {len(df)} App Store reviews from: {latest_file}")
            
            # Standardize columns
            df_standard = pd.DataFrame({
                'review_id': df.get('review_id', df.index).astype(str),
                'source': 'app_store',
                'title': df.get('title', ''),
                'text': df.get('review_text', df.get('text', '')),
                'rating': df.get('rating', 3).astype(float),
                'user_name': df.get('user_name', 'Anonymous'),
                'date': df.get('date', pd.Timestamp.now()),
                'helpful_count': df.get('review_length', 0)  # Using length as proxy
            })
            
            return df_standard
            
        except Exception as e:
            print(f"❌ Error loading App Store data: {str(e)}")
            return pd.DataFrame()
    
    def load_reddit_data(self):
        """
        Load dan standardisasi data Reddit
        """
        print("💬 Loading Reddit data...")
        
        try:
            # Find latest Reddit file
            reddit_files = glob.glob(f"{self.raw_data_path}*reddit*.csv")
            
            if not reddit_files:
                print("❌ Reddit data not found!")
                return pd.DataFrame()
            
            latest_file = max(reddit_files, key=os.path.getctime)
            df = pd.read_csv(latest_file)
            
            print(f"✅ Loaded {len(df)} Reddit posts from: {latest_file}")
            
            # Convert Reddit scores to rating scale (1-5)
            # Score normalization: negative -> 1-2, neutral -> 3, positive -> 4-5
            def score_to_rating(score):
                if score < 0:
                    return 1
                elif score == 0:
                    return 2
                elif score <= 5:
                    return 3
                elif score <= 20:
                    return 4
                else:
                    return 5
            
            # Standardize columns
            df_standard = pd.DataFrame({
                'review_id': df.get('post_id', df.index).astype(str),
                'source': 'reddit',
                'title': df.get('title', ''),
                'text': df.get('text', ''),
                'rating': df.get('score', 0).apply(score_to_rating).astype(float),
                'user_name': df.get('author', 'Anonymous'),
                'date': pd.to_datetime(df.get('created_utc', 0), unit='s', errors='coerce'),
                'helpful_count': df.get('num_comments', 0)
            })
            
            return df_standard
            
        except Exception as e:
            print(f"❌ Error loading Reddit data: {str(e)}")
            return pd.DataFrame()
    
    def combine_all_data(self):
        """
        Menggabungkan semua data dari 3 sumber
        """
        print("\n🔗 Combining all data sources...")
        
        # Load all data
        google_data = self.load_google_play_data()
        app_store_data = self.load_app_store_data()
        reddit_data = self.load_reddit_data()
        
        # Combine all dataframes
        all_data = []
        
        if not google_data.empty:
            all_data.append(google_data)
            print(f"📱 Google Play: {len(google_data)} reviews")
        
        if not app_store_data.empty:
            all_data.append(app_store_data)
            print(f"🍎 App Store: {len(app_store_data)} reviews")
        
        if not reddit_data.empty:
            all_data.append(reddit_data)
            print(f"💬 Reddit: {len(reddit_data)} posts")
        
        if not all_data:
            print("❌ No data found to combine!")
            return pd.DataFrame()
        
        # Combine all
        self.combined_data = pd.concat(all_data, ignore_index=True)
        
        print(f"\n✅ Combined dataset: {len(self.combined_data)} total entries")
        print(f"📊 Sources breakdown:")
        print(self.combined_data['source'].value_counts())
        
        return self.combined_data
    
    def clean_text_data(self, df):
        """
        Membersihkan text data
        """
        print("\n🧹 Cleaning text data...")
        
        df = df.copy()
        
        # Combine title and text
        df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        
        # Clean text function
        def clean_text(text):
            if pd.isna(text) or text == '':
                return ''
            
            # Convert to string
            text = str(text)
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove special characters but keep basic punctuation
            text = re.sub(r'[^\w\s\.\!\?\,\-]', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Strip whitespace
            text = text.strip()
            
            return text
        
        # Apply cleaning
        df['clean_text'] = df['full_text'].apply(clean_text)
        
        # Remove empty texts
        df = df[df['clean_text'].str.len() > 5]  # At least 5 characters
        
        print(f"✅ Cleaned text data. Remaining entries: {len(df)}")
        
        return df
    
    def create_sentiment_labels(self, df):
        """
        Create sentiment labels dari rating
        """
        print("\n🎯 Creating sentiment labels...")
        
        df = df.copy()
        
        def rating_to_sentiment(rating):
            """
            Convert rating to sentiment:
            1-2: Negative
            3: Neutral  
            4-5: Positive
            """
            if rating <= 2:
                return 'negative'
            elif rating == 3:
                return 'neutral'
            else:
                return 'positive'
        
        # Create sentiment labels
        df['sentiment'] = df['rating'].apply(rating_to_sentiment)
        
        # Show distribution
        sentiment_dist = df['sentiment'].value_counts()
        print(f"📊 Sentiment distribution:")
        for sentiment, count in sentiment_dist.items():
            percentage = (count / len(df)) * 100
            print(f"   {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        return df
    
    def add_text_features(self, df):
        """
        Menambahkan fitur-fitur dari text
        """
        print("\n📝 Adding text features...")
        
        df = df.copy()
        
        # Text length
        df['text_length'] = df['clean_text'].str.len()
        
        # Word count
        df['word_count'] = df['clean_text'].str.split().str.len()
        
        # Exclamation marks (excitement indicator)
        df['exclamation_count'] = df['clean_text'].str.count('!')
        
        # Question marks
        df['question_count'] = df['clean_text'].str.count('\\?')
        
        # Capital letters ratio (intensity indicator)
        def capital_ratio(text):
            if len(text) == 0:
                return 0
            return sum(1 for c in text if c.isupper()) / len(text)
        
        df['capital_ratio'] = df['clean_text'].apply(capital_ratio)
        
        print(f"✅ Added text features")
        print(f"📊 Text length stats:")
        print(f"   Mean: {df['text_length'].mean():.1f} characters")
        print(f"   Mean words: {df['word_count'].mean():.1f} words")
        
        return df
    
    def remove_duplicates(self, df):
        """
        Remove duplicate entries
        """
        print("\n🔍 Removing duplicates...")
        
        initial_count = len(df)
        
        # Remove exact text duplicates
        df = df.drop_duplicates(subset=['clean_text'], keep='first')
        
        # Remove very similar texts (same first 50 characters)
        df['text_start'] = df['clean_text'].str[:50]
        df = df.drop_duplicates(subset=['text_start'], keep='first')
        df = df.drop('text_start', axis=1)
        
        final_count = len(df)
        removed = initial_count - final_count
        
        print(f"✅ Removed {removed} duplicate entries")
        print(f"📊 Final dataset: {final_count} entries")
        
        return df
    
    def save_processed_data(self, df):
        """
        Simpan data yang sudah diproses
        """
        print("\n💾 Saving processed data...")
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full processed dataset
        full_filename = f"{self.processed_data_path}wuthering_waves_combined_{timestamp}.csv"
        df.to_csv(full_filename, index=False, encoding='utf-8')
        
        # Save ready-for-ML dataset (hanya kolom yang diperlukan)
        ml_df = df[['review_id', 'source', 'clean_text', 'sentiment', 'rating', 
                   'text_length', 'word_count', 'exclamation_count', 'question_count', 'capital_ratio']].copy()
        
        ml_filename = f"{self.processed_data_path}ml_ready_dataset_{timestamp}.csv"
        ml_df.to_csv(ml_filename, index=False, encoding='utf-8')
        
        print(f"✅ Saved processed data:")
        print(f"   Full dataset: {full_filename}")
        print(f"   ML-ready dataset: {ml_filename}")
        
        # Show final stats
        print(f"\n📊 Final Dataset Summary:")
        print(f"   Total entries: {len(df)}")
        print(f"   Sources: {df['source'].nunique()}")
        print(f"   Sentiment distribution:")
        for sentiment, count in df['sentiment'].value_counts().items():
            print(f"      {sentiment.capitalize()}: {count}")
        
        return full_filename, ml_filename
    
    def run_preprocessing(self):
        """
        Jalankan seluruh proses preprocessing
        """
        print("🚀 Starting Wuthering Waves Data Preprocessing")
        print("=" * 60)
        
        # Step 1: Combine all data
        df = self.combine_all_data()
        
        if df.empty:
            print("❌ No data to process!")
            return None
        
        # Step 2: Clean text data
        df = self.clean_text_data(df)
        
        # Step 3: Create sentiment labels
        df = self.create_sentiment_labels(df)
        
        # Step 4: Add text features
        df = self.add_text_features(df)
        
        # Step 5: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 6: Save processed data
        full_file, ml_file = self.save_processed_data(df)
        
        print(f"\n🎉 Preprocessing completed successfully!")
        print(f"🎯 Ready for EDA & Model Development!")
        print(f"\n📋 Next steps:")
        print(f"1. Run EDA analysis dengan: notebooks/exploratory_analysis.ipynb")
        print(f"2. Start model training dengan: notebooks/model_training.ipynb")
        print(f"3. Use ML-ready dataset: {ml_file}")
        
        return df

def main():
    """
    Main function untuk menjalankan preprocessing
    """
    preprocessor = WutheringWavesPreprocessor()
    processed_data = preprocessor.run_preprocessing()
    
    if processed_data is not None:
        print(f"\n✅ Data preprocessing berhasil!")
        print(f"📊 Dataset siap untuk machine learning dengan {len(processed_data)} entries")
    else:
        print(f"\n❌ Data preprocessing gagal!")

if __name__ == "__main__":
    main()