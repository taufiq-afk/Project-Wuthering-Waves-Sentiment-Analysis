# 🎮 Wuthering Waves Sentiment Analysis

**Multi-Platform Sentiment Analysis with Original Web Scraping Dataset**

A comprehensive machine learning project that analyzes sentiment from user reviews and posts about Wuthering Waves across multiple platforms using supervised learning techniques.

![Project Demo](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Web](https://img.shields.io/badge/Web%20App-Streamlit-red)

## 🎯 Project Overview

This project implements a **supervised machine learning pipeline** that:
- 📊 **Scrapes original data** from Google Play Store, Apple App Store, and Reddit
- 🧹 **Preprocesses and cleans** 5,000+ user-generated reviews and posts
- 🤖 **Trains multiple ML models** with cross-validation and hyperparameter tuning
- 🚀 **Deploys a production web application** for real-time sentiment analysis

## ✨ Key Features

### 🔍 **Data Collection**
- **Google Play Store**: 3,400+ reviews via custom scraper
- **Apple App Store**: 1,300+ reviews using app-store-scraper
- **Reddit**: 300+ posts and comments via JSON API
- **Original Dataset**: 5,000+ authentic user opinions

### 🧠 **Machine Learning Pipeline**
- **Supervised Learning**: Multi-class classification (Positive/Negative/Neutral)
- **Feature Engineering**: TF-IDF vectorization with n-grams
- **Model Comparison**: Naive Bayes, SVM, Logistic Regression, Random Forest
- **Best Performance**: 79.5% accuracy with Random Forest

### 🚀 **Production Web App**
- **Real-time Predictions**: Instant sentiment analysis
- **Batch Processing**: Analyze multiple texts simultaneously
- **Interactive Visualizations**: Confidence scores and distribution charts
- **Professional UI**: Dark theme with responsive design

## 📂 Project Structure

```
wuthering_waves_sentiment/
├── 📁 data/
│   ├── raw/                    # Original scraped data
│   └── processed/              # Cleaned ML-ready datasets
├── 📁 src/
│   ├── scraping.py            # Google Play Store scraper
│   ├── app_store_scraper.py   # App Store scraper
│   ├── reddit_scraper.py      # Reddit posts scraper
│   └── preprocessing.py       # Data cleaning & preparation
├── 📁 notebooks/
│   ├── exploratory_analysis.ipynb  # EDA and insights
│   └── model_training.ipynb        # ML model development
├── 📁 models/
│   ├── best_sentiment_model.pkl    # Trained model
│   └── model_metadata.pkl          # Model information
├── 📄 app.py                  # Streamlit web application
├── 📄 requirements.txt        # Project dependencies
└── 📄 README.md              # Project documentation
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd wuthering_waves_sentiment
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Data Collection (Optional)
```bash
# Scrape Google Play Store reviews
python src/scraping.py

# Scrape App Store reviews  
python src/app_store_scraper.py

# Scrape Reddit posts
python src/reddit_scraper.py
```

### 4. Data Preprocessing
```bash
python src/preprocessing.py
```

### 5. Model Training
```bash
jupyter notebook notebooks/model_training.ipynb
```

### 6. Launch Web Application
```bash
streamlit run app.py
```

Visit `http://localhost:8501` to use the sentiment analysis app!

## 📊 Model Performance

| Model | Cross-Val Accuracy | Test Accuracy | Features |
|-------|-------------------|---------------|----------|
| **Random Forest** | **79.5%** | **79.5%** | 10,000 |
| Logistic Regression | 78.2% | 77.8% | 10,000 |
| SVM | 77.1% | 76.9% | 10,000 |
| Naive Bayes | 74.3% | 73.7% | 10,000 |

### 📈 Dataset Statistics
- **Total Samples**: 5,000+ reviews and posts
- **Sentiment Distribution**: 68% Positive, 26% Negative, 6% Neutral
- **Average Text Length**: 180 characters
- **Feature Engineering**: TF-IDF with 1-2 grams, 10k features

## 🎯 Use Cases

### 📱 **For Developers**
- Monitor user sentiment trends
- Identify common complaints and praise
- Guide feature development priorities

### 📊 **For Researchers** 
- Study sentiment analysis techniques
- Compare multi-platform user behavior
- Analyze gaming industry sentiment patterns

### 🎮 **For Gamers**
- Get quick sentiment overview of reviews
- Analyze community opinions before playing
- Track game reception over time

## 🛠️ Technical Details

### **Data Processing Pipeline**
1. **Web Scraping**: Custom scrapers for each platform
2. **Data Cleaning**: URL removal, text normalization, duplicate handling
3. **Feature Engineering**: TF-IDF vectorization with stopword removal
4. **Label Creation**: Rating-based sentiment mapping (1-2→Negative, 3→Neutral, 4-5→Positive)

### **Machine Learning Workflow**
1. **Train-Test Split**: 80/20 stratified split
2. **Cross-Validation**: 5-fold CV for robust evaluation
3. **Hyperparameter Tuning**: Grid search for optimal parameters
4. **Model Selection**: Best performing model based on test accuracy

### **Web Application Features**
- **Single Analysis**: Real-time prediction with confidence scores
- **Batch Analysis**: Process multiple texts with summary statistics
- **Model Insights**: Performance metrics and dataset information
- **Testing Suite**: Predefined test cases for validation

## 📋 Requirements

### **Python Version**
- Python 3.8 or higher

### **Key Dependencies**
- `pandas`, `numpy`, `scikit-learn` - Data science & ML
- `streamlit`, `plotly` - Web application & visualization
- `requests`, `beautifulsoup4` - Web scraping
- `nltk`, `wordcloud` - Text processing
- `jupyter` - Notebook environment

See `requirements.txt` for complete list.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Wuthering Waves** community for providing authentic reviews
- **Kuro Games** for creating an engaging game worthy of analysis
- **Open source libraries** that made this project possible

## 📞 Contact

- **Project Author**: Muhammad Taufiq Al Fikri
- **Email**: taufikalfikri28@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/taufiq-afk/
- **GitHub**: https://github.com/taufiq-afk

---

### 🎯 Project Highlights

✅ **Original Dataset**: 5,000+ scraped reviews (not pre-existing dataset)  
✅ **Multi-Platform**: Google Play + App Store + Reddit  
✅ **Supervised Learning**: Trained classification model  
✅ **Production Ready**: Deployed Streamlit web application  
✅ **Comprehensive Pipeline**: End-to-end ML workflow  
✅ **Professional Quality**: Industry-standard practices  

**🏆 This project demonstrates proficiency in data collection, machine learning, and web application development!**
