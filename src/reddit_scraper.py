#!/usr/bin/env python3
"""
Wuthering Waves Reddit Posts Scraper
=====================================
Mengambil posts dari Reddit tentang Wuthering Waves untuk analisis sentiment.
Target: 2,000 posts dari berbagai subreddit gaming dan Wuthering Waves.
"""

# import praw  # Not needed, using direct JSON API
import pandas as pd
import time
import os
import requests
from datetime import datetime
import json

class RedditScraper:
    def __init__(self):
        """
        Initialize Reddit scraper.
        Menggunakan Reddit API tanpa authentication untuk public posts.
        """
        self.headers = {
            'User-Agent': 'WutheringWaves-Sentiment-Analysis/1.0'
        }
        self.data = []
        
    def scrape_subreddit_posts(self, subreddit_name, limit=500):
        """
        Scrape posts dari subreddit tertentu menggunakan Reddit JSON API
        """
        print(f"🔍 Scraping r/{subreddit_name}...")
        
        # Reddit JSON API endpoint
        url = f"https://www.reddit.com/r/{subreddit_name}/hot.json"
        
        try:
            params = {
                'limit': min(limit, 100),  # Reddit max 100 per request
                'raw_json': 1
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            posts = data['data']['children']
            
            collected = 0
            for post in posts:
                post_data = post['data']
                
                # Filter posts yang mengandung Wuthering Waves
                title = post_data.get('title', '').lower()
                selftext = post_data.get('selftext', '').lower()
                
                if self.is_wuthering_waves_related(title, selftext):
                    post_info = {
                        'post_id': f"reddit_{post_data.get('id', '')}",
                        'title': post_data.get('title', ''),
                        'text': post_data.get('selftext', ''),
                        'score': post_data.get('score', 0),
                        'num_comments': post_data.get('num_comments', 0),
                        'created_utc': post_data.get('created_utc', 0),
                        'subreddit': subreddit_name,
                        'author': post_data.get('author', '[deleted]'),
                        'upvote_ratio': post_data.get('upvote_ratio', 0.5),
                        'url': f"https://reddit.com{post_data.get('permalink', '')}"
                    }
                    
                    self.data.append(post_info)
                    collected += 1
                    
                    if collected >= limit:
                        break
            
            print(f"✅ Collected {collected} posts from r/{subreddit_name}")
            return collected
            
        except Exception as e:
            print(f"❌ Error scraping r/{subreddit_name}: {str(e)}")
            return 0
    
    def scrape_comments_from_post(self, subreddit_name, post_id, limit=50):
        """
        Scrape comments dari post tertentu
        """
        try:
            url = f"https://www.reddit.com/r/{subreddit_name}/comments/{post_id}.json"
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            if len(data) > 1:
                comments = data[1]['data']['children']
                
                collected = 0
                for comment in comments:
                    if comment['kind'] == 't1':  # Comment type
                        comment_data = comment['data']
                        
                        if comment_data.get('body') and comment_data.get('body') != '[deleted]':
                            comment_info = {
                                'post_id': f"reddit_comment_{comment_data.get('id', '')}",
                                'title': f"Comment on: {comment_data.get('link_title', '')}",
                                'text': comment_data.get('body', ''),
                                'score': comment_data.get('score', 0),
                                'num_comments': 0,
                                'created_utc': comment_data.get('created_utc', 0),
                                'subreddit': subreddit_name,
                                'author': comment_data.get('author', '[deleted]'),
                                'upvote_ratio': 0.5,
                                'url': f"https://reddit.com{comment_data.get('permalink', '')}"
                            }
                            
                            self.data.append(comment_info)
                            collected += 1
                            
                            if collected >= limit:
                                break
                
                return collected
                
        except Exception as e:
            print(f"❌ Error scraping comments: {str(e)}")
            return 0
    
    def is_wuthering_waves_related(self, title, text):
        """
        Check if post is related to Wuthering Waves
        """
        keywords = [
            'wuthering waves', 'wuwa', 'kuro games', 'kuro', 
            'jiyan', 'rover', 'tacet discord', 'solaris', 'huanglong',
            'resonator', 'forte', 'echo', 'waveplates'
        ]
        
        content = f"{title} {text}".lower()
        return any(keyword in content for keyword in keywords)
    
    def search_reddit_posts(self, query="wuthering waves", limit=1000):
        """
        Search posts using Reddit search
        """
        print(f"🔍 Searching Reddit for: {query}")
        
        try:
            url = "https://www.reddit.com/search.json"
            params = {
                'q': query,
                'sort': 'relevance',
                'limit': min(limit, 100),
                'type': 'link',
                'raw_json': 1
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            posts = data['data']['children']
            
            collected = 0
            for post in posts:
                post_data = post['data']
                
                post_info = {
                    'post_id': f"reddit_search_{post_data.get('id', '')}",
                    'title': post_data.get('title', ''),
                    'text': post_data.get('selftext', ''),
                    'score': post_data.get('score', 0),
                    'num_comments': post_data.get('num_comments', 0),
                    'created_utc': post_data.get('created_utc', 0),
                    'subreddit': post_data.get('subreddit', 'unknown'),
                    'author': post_data.get('author', '[deleted]'),
                    'upvote_ratio': post_data.get('upvote_ratio', 0.5),
                    'url': f"https://reddit.com{post_data.get('permalink', '')}"
                }
                
                self.data.append(post_info)
                collected += 1
            
            print(f"✅ Found {collected} posts from search")
            return collected
            
        except Exception as e:
            print(f"❌ Error searching Reddit: {str(e)}")
            return 0
    
    def save_data(self):
        """
        Save collected data to CSV
        """
        if not self.data:
            print("❌ No data to save!")
            return
        
        # Create data directory if not exists
        os.makedirs('data/raw', exist_ok=True)
        
        df = pd.DataFrame(self.data)
        
        # Add timestamp untuk konsistensi dengan format sebelumnya
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        filename = f"data/raw/reddit_wuthering_waves_{timestamp}.csv"
        
        df.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"💾 Data saved to: {filename}")
        print(f"📊 Total posts collected: {len(df)}")
        
        # Show preview
        print(f"\n📋 Data preview:")
        print(df[['title', 'score', 'subreddit', 'num_comments']].head())
        
        return filename
    
    def run_scraping(self, target_posts=2000):
        """
        Main scraping function
        """
        print("🎯 Wuthering Waves Reddit Posts Scraper")
        print("=" * 50)
        print(f"🎯 Target: {target_posts} posts")
        
        # Subreddits to scrape
        subreddits = {
            'WutheringWaves': 800,      # Main subreddit
            'gachagaming': 400,         # General gacha discussions
            'gaming': 200,              # General gaming
            'AndroidGaming': 200,       # Mobile gaming
            'iosgaming': 200,           # iOS gaming
            'mobilegaming': 200         # Mobile gaming general
        }
        
        total_collected = 0
        
        # 1. Scrape from specific subreddits
        for subreddit, limit in subreddits.items():
            if total_collected >= target_posts:
                break
                
            collected = self.scrape_subreddit_posts(subreddit, limit)
            total_collected += collected
            
            # Rate limiting
            time.sleep(2)
        
        # 2. Search for additional posts if needed
        if total_collected < target_posts:
            remaining = target_posts - total_collected
            print(f"\n🔍 Searching for {remaining} more posts...")
            
            search_queries = [
                "wuthering waves",
                "wuwa game",
                "kuro games wuthering",
                "wuthering waves review"
            ]
            
            for query in search_queries:
                if total_collected >= target_posts:
                    break
                    
                collected = self.search_reddit_posts(query, remaining // len(search_queries))
                total_collected += collected
                time.sleep(2)
        
        print(f"\n🎉 Scraping completed!")
        print(f"📊 Total collected: {total_collected} posts")
        
        # Save data
        filename = self.save_data()
        
        # Show rating distribution
        if self.data:
            df = pd.DataFrame(self.data)
            print(f"\n⭐ Score distribution:")
            print(f"High scores (>10): {len(df[df['score'] > 10])}")
            print(f"Medium scores (1-10): {len(df[(df['score'] >= 1) & (df['score'] <= 10)])}")
            print(f"Low scores (<1): {len(df[df['score'] < 1])}")
        
        print(f"\n🎯 Next steps:")
        print(f"1. Check file di: data/raw/")
        print(f"2. Lanjut ke Data combination & cleaning")
        print(f"3. Siap untuk EDA & preprocessing")
        
        return filename

def main():
    """
    Main function to run the scraper
    """
    scraper = RedditScraper()
    scraper.run_scraping(target_posts=2000)

if __name__ == "__main__":
    main()