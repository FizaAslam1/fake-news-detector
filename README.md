# 📰 Fake News Detector: AI-Powered Misinformation Analysis

![Banner](https://img.shields.io/badge/AI--Powered-Misinformation%20Detection-2E86AB?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8%2B-306998?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?style=flat-square&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Best%20Model%20Accuracy-99.5%25-28A745?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

## Overview

**Fake News Detector** is a machine learning system engineered to identify and classify misinformation, benchmarked across **traditional ML and transformer-based approaches**. The best-performing model — **TF-IDF + Logistic Regression** — hits **99.5% accuracy**, outperforming a fine-tuned **BERT** model (92% accuracy) on this dataset. Paired with LIME explainability, this tool provides transparent, interpretable predictions for journalists, researchers, and content moderators.

### 🎯 Mission

To combat misinformation at scale by providing journalists, researchers, and content moderators with an interpretable, production-grade AI solution that distinguishes authentic journalism from fabricated claims.

---

## ✨ Core Features

| Feature | Description |
|---------|-------------|
| 🤖 **99.5% Accuracy** | Best model (TF-IDF + Logistic Regression), benchmarked against BERT |
| ⚖️ **Traditional ML vs Transformers** | Head-to-head comparison — classical NLP beat a fine-tuned BERT model on this task |
| 📝 **Long-Form Article Optimization** | Engineered specifically for comprehensive articles (100+ words) where claims are clearly defined |
| 🔍 **LIME Explainability** | Transparent predictions with visual word-importance analysis—understand *why* the AI made its decision |
| ⚡ **Real-Time Analysis** | Instant predictions with confidence scores |
| 🌐 **Web Application** | Intuitive Streamlit interface for seamless user interaction |
| 🔬 **Reproducible Research** | Full Jupyter notebooks documenting model development and validation |
| 📊 **Production-Ready** | Live deployment with scalable infrastructure |

---

## 🚀 Live Application

**[🔗 Access Fake News Detector](https://fake-news-detector-mkrbensp6tuvx6kigse8mx.streamlit.app/)**

*No installation required—start analyzing news articles immediately.*

---

## 🥊 Traditional ML vs BERT: Why Classical NLP Won

A common assumption is that transformer models automatically outperform "older" NLP techniques. This project tested that assumption directly — and the results say otherwise, at least for this dataset.

| Approach | Accuracy | Notes |
|----------|----------|-------|
| **TF-IDF + Logistic Regression** ✅ | **99.5%** | Best overall — fast, interpretable, cheap to run |
| Fine-tuned BERT | 92% | Slower, heavier, and underperformed on this dataset |

### Why did the simpler model win?

- **Lexical giveaways**: Fake vs. real news in this dataset is separable largely by word choice and phrasing patterns — exactly what TF-IDF captures well. BERT's contextual understanding didn't add much value when the signal is mostly lexical, not semantic.
- **Dataset size & style bias**: Classical datasets like this one often have strong source/style artifacts (certain outlets always show up as "fake" or "real"), which linear models exploit efficiently — sometimes too efficiently.
- **Overfitting risk**: BERT's larger capacity makes it more prone to overfitting on a dataset of this size without heavier regularization/tuning.
- **Compute vs. payoff**: BERT took significantly longer to train and infer, for a lower accuracy — a real-world cost/benefit case for choosing the simpler model in production.

**Takeaway**: Bigger/newer isn't always better. Model selection should be driven by the data and the problem, not by assumptions about which architecture is "state of the art." This comparison is intentionally included to show that reasoning, not just the winning number.

---

## 📚 Why Long-Form Articles?

This detector is **purpose-built for long-form journalism** because:

### Factual Clarity
- Long articles present substantive claims with evidence (or reveal the absence thereof)
- Headlines lack sufficient context for accurate misinformation detection
- Extended narratives enable pattern recognition of credibility markers

### Machine Learning Advantage
- **Feature Richness**: More text = richer linguistic features for classification
- **Statistical Significance**: Sufficient data volume for robust model decisions
- **Contextual Understanding**: Article body reveals intent, structure, and argumentative quality

### Reduced False Positives
- Short headlines are ambiguous and prone to misclassification
- Sensationalism in titles doesn't necessarily indicate fake news in substantive articles
- Long-form content allows separation of rhetorical style from factual accuracy

**Recommended Minimum**: 100+ words per article  
**Optimal Range**: 500–5,000 words

---

## 🏗️ Technical Architecture

### Machine Learning Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                  Raw News Article                        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│      Text Preprocessing & Cleaning                       │
│  • Lowercasing, tokenization, stopword removal          │
│  • Special character handling, normalization             │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│      Feature Extraction (Two Parallel Tracks)             │
│  • Track A: TF-IDF Vectorization (uni/bi/tri-grams)      │
│  • Track B: BERT Tokenization + Contextual Embeddings   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│      Model Benchmarking (Classical ML vs Transformer)    │
│  • TF-IDF → Logistic Regression, RF, XGBoost, SVM, NB   │
│  • BERT → Fine-tuned classification head                │
│  • Binary classification: Real vs. Fake                 │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│      Best Model Selected: TF-IDF + Logistic Regression   │
│      (99.5% accuracy vs. BERT's 92%)                     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│      LIME Explainability Analysis                        │
│  • Feature importance computation                       │
│  • Word-level influence visualization                   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│    Prediction + Interpretability Report                  │
│  • Classification label with confidence                 │
│  • Contributing features highlighted                    │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Classical ML** | scikit-learn | TF-IDF, Logistic Regression, RF, XGBoost, SVM, Naive Bayes |
| **Transformer Benchmark** | BERT (Hugging Face Transformers) | Contextual embedding baseline for comparison |
| **NLP Engine** | NLTK, spaCy | Text preprocessing & feature engineering |
| **Explainability** | LIME | Model-agnostic explanations |
| **Web Framework** | Streamlit | Interactive user interface |
| **Data Processing** | Pandas, NumPy | Efficient data manipulation |
| **Deployment** | Streamlit Cloud | Scalable, serverless hosting |
| **Language** | Python 3.8+ | Modern Python with type hints |

---

## 📊 Performance Metrics

| Model | Accuracy | Notes |
|-------|----------|-------|
| **TF-IDF + Logistic Regression** ✅ | **99.5%** | Deployed model — best accuracy, fastest inference |
| Fine-tuned BERT | 92% | Benchmarked for comparison — higher compute cost, lower accuracy on this dataset |
| Other classical combos (RF, XGBoost, SVM, NB × TF-IDF/CountVec/char n-grams) | Tested (20 combinations total) | TF-IDF + LogReg was the strongest performer |

| Metric | Value (Best Model) | Notes |
|--------|--------------------|-------|
| **Accuracy** | 99.5% | Overall correct predictions |
| **Precision** | High | Few false positives |
| **Recall** | Comprehensive | Strong fake news detection |
| **F1-Score** | Balanced | Excellent real-world performance |
| **Training Data** | 5,000+ articles | Diverse sources, balanced classes |

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git
- pip (Python package manager)

### Local Development Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/FizaAslam1/fake-news-detector.git
   cd fake-news-detector
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import streamlit, sklearn, lime; print('All dependencies installed!')"
   ```

5. **Launch Application**
   ```bash
   streamlit run appp.py
   ```

6. **Access Application**
   - Open browser to `http://localhost:8501`
   - Application loads automatically

---

## 📖 Usage Guide

### Via Web Application

1. Navigate to [Fake News Detector](https://fake-news-detector-mkrbensp6tuvx6kigse8mx.streamlit.app/)
2. **Paste or type** your news article in the text input area
3. **Review minimum length warning** (optimal: 100+ words)
4. Click **"Analyze Article"** button
5. **View Results**:
   - **Classification**: Real or Fake
   - **Confidence Score**: Probability (0-100%)
   - **Explanation**: Words highlighted by importance
   - **Feature Analysis**: Top contributing factors

### Programmatic Usage

```python
from sklearn.externals import joblib
import lime.lime_text

# Load trained model (best performer: TF-IDF + Logistic Regression)
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Prepare text
article_text = """
Your comprehensive news article here...
(minimum 100 words recommended)
"""

# Make prediction
transformed_text = vectorizer.transform([article_text])
prediction = model.predict(transformed_text)[0]
confidence = max(model.predict_proba(transformed_text)[0])

# Generate explanation
explainer = lime.lime_text.LimeTextExplainer(
    class_names=['Real', 'Fake']
)
explanation = explainer.explain_instance(
    article_text, 
    lambda x: model.predict_proba(vectorizer.transform(x))
)

print(f"Prediction: {'Real' if prediction == 1 else 'Fake'}")
print(f"Confidence: {confidence:.1%}")
print("Top contributing words:", explanation.as_list())
```

---

## 📁 Repository Structure

```
fake-news-detector/
│
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
│
├── appp.py                             # Streamlit web application
│
├── fakenews_detection.ipynb           # Complete model & notebook (incl. BERT benchmark)
│
├── model.pkl                           # Trained classification model (TF-IDF + LogReg)
├── vectorizer.pkl                      # Fitted TF-IDF vectorizer

```

---

## 🔬 Model Details

### Training Data
- **Size**: 5,000+ labeled articles
- **Balance**: 50/50 real/fake distribution
- **Sources**: News agencies, fact-check databases, academic datasets
- **Languages**: English (primary)

### Classical ML Track
- **Vectorization**: TF-IDF with sublinear term frequency
- **N-grams**: Unigrams, bigrams, trigrams
- **Vocabulary Size**: 10,000+ unique features
- **Algorithm**: Logistic Regression with L2 regularization (best of 20 combinations tested)
- **Training**: 80/20 train-test split with stratification
- **Validation**: Cross-validation with 5 folds
- **Hyperparameter Tuning**: Grid search optimization
- **Inference Time**: <100ms per article

### Transformer Track (Benchmark)
- **Model**: Fine-tuned BERT (base) with classification head
- **Result**: 92% accuracy — lower than the classical approach on this dataset
- **Why it's included**: To demonstrate that model choice should be evidence-driven, not assumption-driven — and to document the trade-off between compute cost and accuracy gain

### Explainability (LIME)
- **Method**: Local Interpretable Model-Agnostic Explanations
- **Granularity**: Word-level feature importance
- **Perturbations**: 1,000+ text variations per explanation
- **Visualization**: Interactive feature contribution charts

---

## 🚨 Important Considerations

### Accuracy & Limitations
- ⚠️ **99.5% accuracy reflects lab conditions**—real-world performance may vary
- ⚠️ **No system is 100% accurate**—always verify critical information independently
- ⚠️ **Dataset bias risk**: high accuracy on classical fake-news datasets can partly reflect source/style artifacts rather than pure factual reasoning — a known limitation being actively considered
- ⚠️ **Emerging misinformation tactics** may require model retraining
- ⚠️ **Language limitations**: Optimized for English-language articles

### Content Guidelines
- ✅ **Optimal**: 100–5,000 words per article
- ✅ **Content Type**: News, journalism, factual reporting
- ❌ **Poor Input**: Headlines alone, social media posts, short snippets
- ❌ **Out of Scope**: Non-English content, opinions without factual claims

### Responsible Use
- 📋 Use as a **supplementary tool**, not sole arbitrator
- 📋 **Verify** predictions with human editorial judgment
- 📋 **Disclose** that content was analyzed by AI
- 📋 **Respect** privacy and ethical guidelines in content analysis

---

## 🤝 Contributing

We welcome contributions to improve the Fake News Detector!

### Contribution Process

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/enhanced-model
   ```

3. **Make Changes & Commit**
   ```bash
   git commit -m "feat: improve model accuracy with new features"
   ```

4. **Push to Branch**
   ```bash
   git push origin feature/enhanced-model
   ```

5. **Open Pull Request**
   - Describe changes thoroughly
   - Reference related issues
   - Include test coverage

### Contribution Areas
- 🔧 Model improvements & optimization (including revisiting the BERT fine-tuning setup)
- 🌍 Support for additional languages
- 📊 Enhanced evaluation metrics
- 🎨 UI/UX improvements
- 📚 Documentation & tutorials
- 🧪 Test coverage expansion

### Code Standards
- Follow PEP 8 style guidelines
- Include docstrings for all functions
- Add unit tests for new features
- Update documentation accordingly

---

## 📄 License

This project is released under the **MIT License**. See LICENSE file for complete terms.

---

## 👩‍💻 Author & Contributors

**Primary Developer**
- **Fiza Aslam** — [@FizaAslam1](https://github.com/FizaAslam1)

---

## 🙏 Acknowledgments

### Libraries & Frameworks
- [scikit-learn](https://scikit-learn.org) — ML algorithms & utilities
- [Hugging Face Transformers](https://huggingface.co/transformers) — BERT benchmark
- [LIME](https://github.com/marcotcr/lime) — Model explainability
- [Streamlit](https://streamlit.io) — Interactive web applications
- [NLTK](https://www.nltk.org) — Natural language processing

### Research & Inspiration
- "Why Should I Trust You?" — LIME paper by Ribeiro et al.
- Fake News Challenge datasets
- Misinformation research community

---

## 📞 Support & Community

### Get Help

- 💬 Check [GitHub Discussions](https://github.com/FizaAslam1/fake-news-detector/discussions)
- 🐛 Report issues on [GitHub Issues](https://github.com/FizaAslam1/fake-news-detector/issues)

### Issue Reporting
Please include:
- Article text sample (or description)
- Expected vs. actual prediction
- Screenshots if applicable
- Python version & environment details

---

## 📊 Citation

If you use Fake News Detector in your research, please cite:

```bibtex
@software{aslam2026fakenews,
  title = {Fake News Detector: AI-Powered Misinformation Analysis},
  author = {Aslam, Fiza},
  year = {2026},
  url = {https://github.com/FizaAslam1/fake-news-detector}
}
```

---

## 🔮 Roadmap

- [ ] Improve BERT fine-tuning (regularization, larger dataset) to close the gap with classical ML
- [ ] Multi-language support (Spanish, French, Arabic)
- [ ] Real-time API for third-party integration
- [ ] Browser extension for in-article analysis
- [ ] Mobile application (iOS/Android)
- [ ] Advanced fact-checking integration
- [ ] Adversarial robustness testing
- [ ] User feedback loop for continuous improvement

---

## ⭐ Show Your Support

If this project helps you, please **star the repository**! Your support motivates continued development.

[![GitHub stars](https://img.shields.io/github/stars/FizaAslam1/fake-news-detector?style=social)](https://github.com/FizaAslam1/fake-news-detector)

---

<div align="center">

**Made with ❤️ for accurate information and trustworthy journalism**

[🔗 Try the Live Demo](https://fake-news-detector-mkrbensp6tuvx6kigse8mx.streamlit.app/)  • [🐛 Report Issues](https://github.com/FizaAslam1/fake-news-detector/issues)

*Last Updated: July 2026*

</div>
