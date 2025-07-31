import os
import re
import string
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from flask import Flask, render_template, request, redirect, url_for
from googleapiclient.discovery import build
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import google.generativeai as genai
from datetime import datetime
import html
app = Flask(__name__)

# Configure Google Generative AI
# GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
GOOGLE_API_KEY = ""
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')

YOUTUBE_API_KEY =""
if not YOUTUBE_API_KEY:
    raise ValueError("No YOUTUBE_API_KEY set for application")

def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    raise ValueError("Invalid YouTube URL")

def fetch_youtube_comments(video_id, max_comments=100):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments = []
    next_page_token = None
    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()
        for item in response.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(text)
            if len(comments) >= max_comments:
                return comments
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return comments

def analyze_sentiments(comments):
    stats = {"positive": 0, "neutral": 0, "negative": 0}
    for c in comments:
        polarity = TextBlob(c).sentiment.polarity
        if polarity > 0.1:
            stats["positive"] += 1
        elif polarity < -0.1:
            stats["negative"] += 1
        else:
            stats["neutral"] += 1
    return stats

def preprocess(comments):
    # Remove URLs
    comments = [re.sub(r'http\S+', '', c) for c in comments]
    # Remove HTML entities
    comments = [re.sub(r'&[a-z]+;', '', c) for c in comments]
    # Remove emojis and special characters
    comments = [re.sub(r'[^\w\s]', '', c) for c in comments]
    # Convert to lowercase
    processed = [c.lower() for c in comments]
    # Remove stopwords and short words
    stop_words = set([
        'the', 'and', 'to', 'of', 'i', 'a', 'you', 'my', 'in', 'it', 'that', 
        'is', 'im', 'for', 'on', 'dont', 'with', 'me', 'just', 'be', 'not', 
        'this', 'but', 'have', 'at', 'was', 'so', 't', 'are', 'if', 'your',
        'what', 'do', 'he', 'she', 'we', 'they', 'how', 'when', 'where', 'why',
        'an', 'as', 'or', 'its', 'am', 'are', 'were', 'his', 'her', 'their'
    ])
    processed = [" ".join([w for w in c.split() if len(w) > 2 and w not in stop_words]) for c in processed]
    return processed

def extract_topics(comments, n_topics=3, n_words=5):
    processed = preprocess(comments)
    
    # Filter out very short comments
    processed = [c for c in processed if len(c.split()) > 3]
    
    if not processed:
        return ["Not enough meaningful comments"]
        
    vectorizer = CountVectorizer(stop_words='english', max_df=0.90, min_df=2, ngram_range=(1, 2))
    X = vectorizer.fit_transform(processed)
    
    if X.shape[0] == 0 or X.shape[1] == 0:
        return ["Not enough data"]
        
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    words = np.array(vectorizer.get_feature_names_out())
    topics = []
    
    for topic in lda.components_:
        top_words = words[np.argsort(topic)[-n_words:]][::-1]
        # Filter out single characters
        top_words = [w for w in top_words if len(w) > 1]
        topics.append(", ".join(top_words))
        
    return topics



def extract_keywords(comments, n_keywords=10):
    try:
        processed = preprocess(comments)
        
        # If we don't have enough words, return empty list
        if not any(processed):
            return []
            
        # Use TF-IDF for better keyword extraction
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df=0.85, 
            min_df=1, 
            ngram_range=(1, 2)  # Include 1-word and 2-word phrases
        )
        X = vectorizer.fit_transform(processed)
        
        if X.shape[1] == 0:
            return []
        
        # Get top keywords based on TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        scores = X.sum(axis=0).A1  # Convert to 1D array
        sorted_indices = np.argsort(scores)[::-1]  # Sort descending
        
        # Get top N keywords
        keywords = [feature_names[i] for i in sorted_indices[:n_keywords]]
        
        # Filter out numbers and single characters
        keywords = [k for k in keywords if len(k) > 1 and not k.isdigit()]
        
        return keywords
        
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        return ["Error extracting keywords"]

def extract_suggestions(comments, n_suggestions=5):
    suggestion_patterns = [r'\bshould\b', r'\bplease\b', r'can you', r'would you', r'could you', r'i wish', r'i hope']
    suggestions = []
    for comment in comments:
        for pat in suggestion_patterns:
            if re.search(pat, comment, re.IGNORECASE):
                suggestions.append(comment)
                break
    most_common = [s for s, _ in Counter(suggestions).most_common(n_suggestions)]
    return most_common if most_common else ["No clear suggestions found"]

def get_llm_feedback(insights):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt_text = f"""
        You are a YouTube content advisor.
        Here are audience insights:
        {insights}

        Provide feedback in this structured HTML format:

        <h3>Strengths of the Video</h3>
        <ul>
            <li>Strength 1</li>
            <li>Strength 2</li>
            <li>Strength 3</li>
        </ul>

        <h3>Improvements Needed</h3>
        <ul>
            <li>Improvement 1</li>
            <li>Improvement 2</li>
            <li>Improvement 3</li>
        </ul>

        <h3>Content Suggestions</h3>
        <ul>
            <li>Suggestion 1</li>
            <li>Suggestion 2</li>
            <li>Suggestion 3</li>
        </ul>
        """
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_sentiment_pie(stats, filename):
    labels = list(stats.keys())
    sizes = list(stats.values())
    colors = ['#4CAF50', '#9E9E9E', '#F44336']
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title("Sentiment Distribution")
    plt.savefig(filename)
    plt.close()
def show_wordcloud(comments, filename):
    # Remove emojis and special characters
    text = " ".join(comments)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        collocations=False,  # Better for single words
        max_words=200
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_url = request.form['url']
        try:
            video_id = extract_video_id(video_url)
            comments = fetch_youtube_comments(video_id, max_comments=200)
            
            if not comments:
                 return render_template('index.html', error="No comments found or video ID invalid.")

            
            # Analyze sentiments
            stats = analyze_sentiments(comments)
            if not stats or sum(stats.values()) == 0:
                return render_template('index.html', error="Failed to analyze comments.")

            # Generate and save plots
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            sentiment_filename = f"sentiment_{timestamp}.png"
            wordcloud_filename = f"wordcloud_{timestamp}.png"
            plot_sentiment_pie(stats, os.path.join('static', sentiment_filename))
            show_wordcloud(comments, os.path.join('static', wordcloud_filename))
            
            # Extract topics, keywords, suggestions
            topics = extract_topics(comments)
            keywords = extract_keywords(comments)
            suggestions = extract_suggestions(comments)
            
            # Prepare insights for LLM feedback
            insights = {
                "sentiment_stats": stats,
                "topics": topics,
                "keywords": keywords,
                "suggestions": suggestions
            }
            
            # Get LLM feedback
            llm_feedback = get_llm_feedback(str(insights)) if GOOGLE_API_KEY else "Google API key not set. LLM feedback unavailable."
            
            # In your route handler, before rendering:
            cleaned_suggestions = [html.unescape(s) for s in suggestions]
            llm_feedback_cleaned = html.unescape(llm_feedback) if llm_feedback else None

            return render_template('results.html', 
                                video_url=video_url,
                                sentiment_stats=stats,
                                sentiment_img=sentiment_filename,
                                wordcloud_img=wordcloud_filename,
                                topics=topics,
                                keywords=keywords,
                                suggestions=cleaned_suggestions,  # Use cleaned version
                                llm_feedback=llm_feedback_cleaned)  # Use cleaned version
        except Exception as e:
            return render_template('index.html', error=f"Error processing video: {str(e)}")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)