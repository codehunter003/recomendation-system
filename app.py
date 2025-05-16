import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import streamlit as st
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
# Page config
st.set_page_config(
    page_title="FashionFinder - Smart Shopping",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Custom Styles ----------------------
def load_custom_css():
    dark_mode = st.sidebar.checkbox("🌙 Enable Dark Mode")
    background_color = '#1e1e1e' if dark_mode else 'white'
    text_color = 'white' if dark_mode else '#333'
    secondary_text = '#ccc' if dark_mode else '#555'
    st.markdown(f"""
    <style>
        
        body {{
            font-family: 'Segoe UI', sans-serif;
            background-color: {background_color};
            color: {text_color};
        }}
        .header {{
            background-color: #ff3f6c;
            padding: 20px;
            color: white;
            text-align: center;
            border-radius: 8px;
            position: relative;
        }}
        .nav-buttons {{
            position: absolute;
            top: 15px;
            right: 15px;
            display: flex;
            gap: 10px;
        }}
        .nav-button {{
            background-color: white;
            color: #ff3f6c;
            border: none;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }}
        .product-card {{
            border: 1px solid #eee;
            border-radius: 8px;
            padding: 15px;
            background-color: {'#2e2e2e' if dark_mode else 'white'};
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
            height: 450px;
            display: flex;
            flex-direction: column;
            margin-bottom: 20px;
            
        }}
        .product-card:hover {{
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        .product-card h4 {{
            font-size: 18px;
            color: {text_color};
            margin: 10px 0 5px 0;
        }}
        .product-card p {{
            color: {secondary_text};
            margin: 4px 0;
            font-size: 14px;
        }}
        .product-image-container {{
            height: 250px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .product-image {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        .product-info {{
            flex-grow: 1;
        }}
        .product-actions {{
            margin-top: auto;
        }}
        .rating-badge {{
            background-color: #14958f;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }}
        img.product-img:hover {{
            transform: scale(1.05);
            transition: all 0.3s ease-in-out;
        }}
        .action-button {{
            background-color: #ff3f6c;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            width: 100%;
            margin-top: 10px;
        }}
        .secondary-button {{
            background-color: white;
            color: #ff3f6c;
            border: 1px solid #ff3f6c;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            width: 100%;
            margin-top: 10px;
        }}
        .badge {{
            background-color: #ff3f6c;
            color: white;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
        }}
    </style>
    """, unsafe_allow_html=True) 

# ---------------------- Load Data ----------------------
@st.cache_data
def load_data():
    reviews = pd.read_csv("clothing_reviews.csv")
    products = pd.read_csv("clothing_description.csv")
    reviews.dropna(subset=['Customer ID'], inplace=True)
    reviews['Customer ID'] = reviews['Customer ID'].astype(int)
    products['product_id'] = products['product_id'].astype(int)
    
    # Add sentiment analysis
    reviews['final_review'] = reviews['Review Text'].fillna('')
    
    # Create sentiment scores for products
    sentiment_scores = generate_sentiment_scores(reviews)
    
    # Merge sentiment scores with products
    products = pd.merge(products, sentiment_scores[['product_id', 'sentiment_score', 'overall_rank']], 
                         on='product_id', how='left')
    
    # Fill NaN sentiment scores with neutral value
    products['sentiment_score'] = products['sentiment_score'].fillna(0.5)
    products['overall_rank'] = products['overall_rank'].fillna(999)  # Lower rank for products without reviews
    
    return reviews, products

@st.cache_resource
def generate_sentiment_scores(reviews):
    # For demonstration purposes, we'll simulate the sentiment analysis process
    # In a real implementation, this would use the full model training pipeline
    
    # Create a basic TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    
    # Check if there are enough reviews to train a model
    if len(reviews) > 100:
        # Create features from review text
        X = tfidf.fit_transform(reviews['final_review'])
        
        # Create target variable (for demo, we'll use rating > 3 as positive)
        reviews['sentiment'] = (reviews['Rating'] > 3).astype(int)
        y = reviews['sentiment']
        
        # Train a simple classifier
        etc = ExtraTreesClassifier(n_estimators=50, random_state=42)
        etc.fit(X, y)
        
        # Predict sentiment for all reviews
        reviews['predicted_sentiment'] = etc.predict_proba(X)[:,1]  # Get probability of positive class
    else:
        # Fallback if not enough data
        reviews['predicted_sentiment'] = reviews['Rating'] / 5.0  # Normalize ratings as sentiment
    
    # Group by product_id to compute average sentiment score
    overall_scores = reviews.groupby(['product_id'])['predicted_sentiment'].mean().reset_index()
    overall_scores.rename(columns={'predicted_sentiment': 'sentiment_score'}, inplace=True)
    
    # Rank all products based on sentiment score
    overall_scores['overall_rank'] = overall_scores['sentiment_score'].rank(method='dense', ascending=False)
    
    return overall_scores

# Initialize session state variables for wishlist and cart
def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    if 'selected_product' not in st.session_state:
        st.session_state.selected_product = None
    if 'wishlist' not in st.session_state:
        st.session_state.wishlist = []
    if 'cart' not in st.session_state:
        st.session_state.cart = []

# Navigation functions
def go_to_wishlist():
    st.session_state.page = 'wishlist'
    st.rerun()
    
def go_to_checkout():
    st.session_state.page = 'checkout'
    st.rerun()

def go_to_main():
    st.session_state.page = 'main'
    st.rerun()


# ---------------------- Header Navigation ----------------------
def header_with_navigation():
    st.markdown(
    """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Galada&display=swap');
            .header {
                background-image: url("https://media3.giphy.com/media/BLHlwFGnmPIP0yFKNl/giphy.webp?cid=ecf05e47rzatwsnq6epkf56kvoncrdqcqhyk80fnlmqydb93&ep=v1_gifs_search&rid=giphy.webp&ct=g");
                background-size: cover;
                background-position: center;
                padding: 40px;
                border-radius: 10px;
                text-align: center;
                color: white;
                margin-bottom: 20px;
            }
            .header h1 {
                font-weight: bold;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            }
            .header p {
                font-size: 18px;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
            }
            </style>
            
            <div class='header'>
                <h1>Find Your Fits</h1>
                <p>Your AI-powered Personal Stylist</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background-image: url("https://media3.giphy.com/media/BLHlwFGnmPIP0yFKNl/giphy.webp?cid=ecf05e47rzatwsnq6epkf56kvoncrdqcqhyk80fnlmqydb93&ep=v1_gifs_search&rid=giphy.webp&ct=g");
            background-size: cover;
            background-position: center;
        }
        
        [data-testid="stSidebar"]::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: white;
            opacity: 0.0; /* Adjust opacity here - higher value = less transparent */
            z-index: -1;
        }
        
        [data-testid="stSidebar"] > div {
            background-color: rgba(255, 255, 255, 0.1);
            position: relative;
            z-index: 1;
        }
        </style>
    """, unsafe_allow_html=True)
    # Use Streamlit's native components for navigation instead of JavaScript
    col1, col2, col3 = st.columns([5, 1, 1])
    with col2:
        if st.button(f"❤️ Wishlist ({len(st.session_state.wishlist)})", use_container_width=True):
            go_to_wishlist()
    with col3:
        if st.button(f"🛒 Cart ({len(st.session_state.cart)})", use_container_width=True):
            go_to_checkout()

# ---------------------- Recommendation Models ----------------------


# ---------------------- UI Components ----------------------

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import streamlit as st
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier

# ---------------------- Load Data ----------------------
@st.cache_data
def load_data():
    reviews = pd.read_csv("clothing_reviews.csv")
    products = pd.read_csv("clothing_description.csv")
    reviews.dropna(subset=['Customer ID'], inplace=True)
    reviews['Customer ID'] = reviews['Customer ID'].astype(int)
    products['product_id'] = products['product_id'].astype(int)
    
    # Add sentiment analysis
    reviews['final_review'] = reviews['Review Text'].fillna('')
    
    # Create sentiment scores for products
    sentiment_scores = generate_sentiment_scores(reviews)
    
    # Merge sentiment scores with products
    products = pd.merge(products, sentiment_scores[['product_id', 'sentiment_score', 'overall_rank']], 
                         on='product_id', how='left')
    
    # Fill NaN sentiment scores with neutral value
    products['sentiment_score'] = products['sentiment_score'].fillna(0.5)
    products['overall_rank'] = products['overall_rank'].fillna(999)  # Lower rank for products without reviews
    
    return reviews, products

@st.cache_resource
def generate_sentiment_scores(reviews):
    # For demonstration purposes, we'll simulate the sentiment analysis process
    # In a real implementation, this would use the full model training pipeline
    
    # Create a basic TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    
    # Check if there are enough reviews to train a model
    if len(reviews) > 100:
        # Create features from review text
        X = tfidf.fit_transform(reviews['final_review'])
        
        # Create target variable (for demo, we'll use rating > 3 as positive)
        reviews['sentiment'] = (reviews['Rating'] > 3).astype(int)
        y = reviews['sentiment']
        
        # Train a simple classifier
        etc = ExtraTreesClassifier(n_estimators=50, random_state=42)
        etc.fit(X, y)
        
        # Predict sentiment for all reviews
        reviews['predicted_sentiment'] = etc.predict_proba(X)[:,1]  # Get probability of positive class
    else:
        # Fallback if not enough data
        reviews['predicted_sentiment'] = reviews['Rating'] / 5.0  # Normalize ratings as sentiment
    
    # Group by product_id to compute average sentiment score
    overall_scores = reviews.groupby(['product_id'])['predicted_sentiment'].mean().reset_index()
    overall_scores.rename(columns={'predicted_sentiment': 'sentiment_score'}, inplace=True)
    
    # Rank all products based on sentiment score
    overall_scores['overall_rank'] = overall_scores['sentiment_score'].rank(method='dense', ascending=False)
    
    return overall_scores

# ---------------------- Recommendation Models ----------------------
@st.cache_resource
def build_models(reviews):
    ratings = reviews[['Customer ID', 'product_id', 'Rating']].dropna()
    matrix = ratings.pivot_table(index='Customer ID', columns='product_id', values='Rating').fillna(0)
    sparse = csr_matrix(matrix.values)
    idx_to_cust = dict(enumerate(matrix.index))
    cust_to_idx = {v: k for k, v in idx_to_cust.items()}
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
    model.fit(sparse)
    sim = cosine_similarity(csr_matrix(matrix.T.values))
    sim_df = pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)
    return matrix, sparse, model, sim_df, idx_to_cust, cust_to_idx

# ---------------------- Recommendation Logic ----------------------
def hybrid_recommend(customer_id, matrix, sparse, model, sim_df, idx_to_cust, cust_to_idx, products, top_n=10):
    if customer_id not in cust_to_idx:
        return pd.DataFrame()

    idx = cust_to_idx[customer_id]
    dists, inds = model.kneighbors(sparse[idx], n_neighbors=6)
    neighbors = [idx_to_cust[i] for i in inds.flatten()[1:]]
    neighbor_scores = matrix.loc[neighbors].mean()

    user_scores = matrix.loc[customer_id]
    rated_items = user_scores[user_scores > 0].index
    item_scores = pd.Series(0, index=matrix.columns)
    for item in rated_items:
        item_scores += sim_df[item] * user_scores[item]

    combined = 0.5 * neighbor_scores + 0.5 * item_scores
    combined = combined.drop(rated_items, errors='ignore')
    
    # Get products with combined scores
    recommended_products = pd.DataFrame({
        'product_id': combined.index,
        'rec_score': combined.values
    })
    
    # Merge with products to get sentiment scores
    rec_with_sentiment = pd.merge(
        recommended_products,
        products[['product_id', 'sentiment_score', 'overall_rank']],
        on='product_id',
        how='left'
    )
    
    # Create a final score that combines recommendation and sentiment
    # 60% weight to recommendation score (normalized) and 40% to sentiment
    rec_with_sentiment['rec_score_norm'] = (rec_with_sentiment['rec_score'] - rec_with_sentiment['rec_score'].min()) / \
                                      (rec_with_sentiment['rec_score'].max() - rec_with_sentiment['rec_score'].min() + 1e-10)
    
    rec_with_sentiment['final_score'] = (0.6 * rec_with_sentiment['rec_score_norm']) + \
                                   (0.4 * rec_with_sentiment['sentiment_score'])
    
    # Sort by final score and get top_n products
    top_recommendations = rec_with_sentiment.sort_values('final_score', ascending=False).head(top_n)
    
    # Return the full product information for these recommendations
    return products[products['product_id'].isin(top_recommendations['product_id'])]

# ---------------------- UI Components ----------------------
def render_product(product, avg_rating):
    prod_id = product['product_id']
    in_wishlist = prod_id in [p['product_id'] for p in st.session_state.wishlist] if st.session_state.wishlist else False
    in_cart = prod_id in [p['product_id'] for p in st.session_state.cart] if st.session_state.cart else False
    
    # Format sentiment score for display
    sentiment_score = product.get('sentiment_score', 0)
    sentiment_display = f"{sentiment_score:.2f}" if sentiment_score else "N/A"
    
    # Determine sentiment badge color
    if sentiment_score >= 0.8:
        sentiment_color = "#28a745"  # Green for high sentiment
    elif sentiment_score >= 0.6:
        sentiment_color = "#17a2b8"  # Blue for good sentiment
    elif sentiment_score >= 0.4:
        sentiment_color = "#ffc107"  # Yellow for neutral sentiment
    else:
        sentiment_color = "#dc3545"  # Red for low sentiment
    
    with st.container():
        st.markdown(f"""
        <div class='product-card'>
            <div class='product-image-container'>
                <img class='product-image' src='{product['image_url']}' onerror="this.src='https://via.placeholder.com/300x400?text=No+Image';">
            </div>
            <div class='product-info'>
                <h4>{product['product_name']}</h4>
                <p><strong>Brand:</strong> {product['product_brand']}</p>
                <p><strong>Price:</strong> ₹{product['price']:.2f} <span class='rating-badge'>★ {avg_rating}</span></p>
                <p><strong>Sentiment:</strong> <span style="background-color: {sentiment_color}; color: white; padding: 3px 8px; border-radius: 4px; font-size: 12px;">{sentiment_display}</span></p>
            </div>
            <div class='product-actions'>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"{'❤️ Added' if in_wishlist else '🤍 Wishlist'}", key=f"wish_{prod_id}"):
                if not in_wishlist:
                    st.session_state.wishlist.append(dict(product))
                    st.success(f"Added {product['product_name']} to wishlist!")
                else:
                    st.session_state.wishlist = [p for p in st.session_state.wishlist if p['product_id'] != prod_id]
                    st.info(f"Removed from wishlist")
                st.rerun()
        with col2:
            if st.button(f"{'✓ In Cart' if in_cart else '🛒 Add to Cart'}", key=f"cart_{prod_id}"):
                if not in_cart:
                    st.session_state.cart.append(dict(product))
                    st.success(f"Added {product['product_name']} to cart!")
                else:
                    st.session_state.cart = [p for p in st.session_state.cart if p['product_id'] != prod_id]
                    st.info(f"Removed from cart")
                st.rerun()
        
        if st.button(f"View Details", key=f"view_{prod_id}"):
            st.session_state.selected_product = prod_id
            st.session_state.page = 'details'
            st.rerun()

def get_avg_rating(pid, reviews):
    subset = reviews[reviews['product_id'] == pid]
    return round(subset['Rating'].mean(), 1) if not subset.empty else 0

def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ---------------------- Login Page ----------------------
def login_page():
    # Apply custom CSS
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://images.pexels.com/photos/1884584/pexels-photo-1884584.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;}

    .main-container {
        display: flex;
        background: #e9ebee;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0,0,0,0.1);
        margin: 0 auto;
        max-width: 900px;
    }
    
    .login-section {
        background-color: #fafafa;
        padding: 40px;
        text-align: center;
        width: 60%;
        border-radius: 10px 0 0 10px;
    }
    
    .login-section h1 {
        margin-bottom: 40px;
        font-size: 2.5em;
        color: #333;
    }
    
    .form-input {
        background-color: #eeeeef !important;
        border: none !important;
        padding: 10px !important;
        margin-bottom: 20px !important;
        border-radius: 4px !important;
        width: 100% !important;
    }
    
    .remember-forgot {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .login-btn {
        background-color: #9526a9 !important;
        color: white !important;
        width: 100% !important;
        padding: 10px 0 !important;
        font-size: 18px !important;
        margin: 20px 0 !important;
        transition: all 0.3s ease !important;
        border: none !important;
    }
    
    .login-btn:hover {
        background-color: #7d1e91 !important;
        transform: scale(0.98);
    }
                
    .admin-btn {
        background-color: #9526a9 !important;
        color: white !important;
        width: 100% !important;
        padding: 10px 0 !important;
        font-size: 18px !important;
        margin: 20px 0 !important;
        transition: all 0.3s ease !important;
        border: none !important;
    }
    
    .admin-btn:hover {
        background-color: #7d1e91 !important;
        transform: scale(0.98);
    }
    
    .social-login {
        margin-top: 20px;
    }
    
    .social-divider {
        display: flex;
        align-items: center;
        margin: 20px 0;
    }
    
    .social-divider hr {
        flex-grow: 1;
        border: none;
        height: 1px;
        background-color: #ddd;
    }
    
    .social-divider p {
        margin: 0 10px;
        color: #777;
    }
    
    .social-icons {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 20px 0;
    }
    
    .social-icon {
        background-color: #f5f5f5;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .social-icon:hover {
        background-color: #e0e0e0;
    }
    
    .stButton button {
        border-radius: 4px;
    }
    
    /* Hide default Streamlit elements we don't need */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    .stCheckbox {
        margin-bottom: 0 !important;
    }
    
    /* Responsive adjustments */
    @media screen and (max-width: 768px) {
        .main-container {
            flex-direction: column;
        }
        .login-section, .register-section {
            width: 100%;
            border-radius: 0;
        }
        .register-section {
            border-radius: 0 0 10px 10px;
        }
        .login-section {
            border-radius: 10px 10px 0 0;
        }
    }
    </style>
    """, unsafe_allow_html=True)



    # Insert form elements into the layout
    # Put these in columns to place them where we want in the custom layout
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        customer_id = st.text_input("", placeholder="Customer ID", key="customer_id", help="Enter your customer ID")
        st.markdown('<style>div[data-testid="stFormSubmitButton"] {display: none}</style>', unsafe_allow_html=True)
        
        password = st.text_input("", placeholder="Password", type="password", key="password", help="Enter your password")
        
        col_remember, col_forgot = st.columns(2)
        with col_remember:
            st.checkbox("Remember me", key="remember_me")
        with col_forgot:
            st.markdown('<div style="text-align: right;"><a href="#" style="color: #333; text-decoration: none;">Forgot password?</a></div>', unsafe_allow_html=True)
        
        login_button = st.button("Log in", key="login_btn", use_container_width=True)

        admin_login = st.button("Sign in as Admin", key="admin_btn", use_container_width=True)
        if admin_login:
            st.session_state.page = 'admin'
            st.rerun()      

        
        # Social login section
        st.markdown("""
        <div class="social-divider">
            <hr><p>Or Connect With</p><hr>
        </div>
        <div class="social-icons">
            <div class="social-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" fill="#3b5998" viewBox="0 0 320 512">
                    <path d="M279.14 288l14.22-92.66h-88.91v-60.13c0-25.35 12.42-50.06 52.24-50.06h40.42V6.26S260.43 0 225.36 0c-73.22 0-121.08 44.38-121.08 124.72v70.62H22.89V288h81.39v224h100.17V288z"/>
                </svg>
            </div>
            <div class="social-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" fill="#1DA1F2" viewBox="0 0 512 512">
                    <path d="M459.37 151.716c.325 4.548.325 9.097.325 13.645 0 138.72-105.583 298.558-298.558 298.558-59.452 0-114.68-17.219-161.137-47.106 8.447.974 16.568 1.299 25.34 1.299 49.055 0 94.213-16.568 130.274-44.832-46.132-.975-84.792-31.188-98.112-72.772 6.498.974 12.995 1.624 19.818 1.624 9.421 0 18.843-1.3 27.614-3.573-48.081-9.747-84.143-51.98-84.143-102.985v-1.299c13.969 7.797 30.214 12.67 47.431 13.319-28.264-18.843-46.781-51.005-46.781-87.391 0-19.492 5.197-37.36 14.294-52.954 51.655 63.675 129.3 105.258 216.365 109.807-1.624-7.797-2.599-15.918-2.599-24.04 0-57.828 46.782-104.934 104.934-104.934 30.213 0 57.502 12.67 76.67 33.137 23.715-4.548 46.456-13.32 66.599-25.34-7.798 24.366-24.366 44.833-46.132 57.827 21.117-2.273 41.584-8.122 60.426-16.243-14.292 20.791-32.161 39.308-52.628 54.253z"/>
                </svg>
            </div>
            <div class="social-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" fill="#171515" viewBox="0 0 496 512">
                    <path d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3.3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5.3-6.2 2.3zm44.2-1.7c-2.9.7-4.9 2.6-4.6 4.9.3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3.7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3.3 2.9 2.3 3.9 1.6 1 3.6.7 4.3-.7.7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3.7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3.7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"/>
                </svg>
            </div>
            <div class="social-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" fill="#0077B5" viewBox="0 0 448 512">
                    <path d="M100.28 448H7.4V148.9h92.88zM53.79 108.1C24.09 108.1 0 83.5 0 53.8a53.79 53.79 0 0 1 107.58 0c0 29.7-24.1 54.3-53.79 54.3zM447.9 448h-92.68V302.4c0-34.7-.7-79.2-48.29-79.2-48.29 0-55.69 37.7-55.69 76.7V448h-92.78V148.9h89.08v40.8h1.3c12.4-23.5 42.69-48.3 87.88-48.3 94 0 111.28 61.9 111.28 142.3V448z"/>
                </svg>
            </div>
        </div>
        <div style="text-align: center; margin-top: 20px;">
            <span style="color: #777; font-size: 12px;">&copy; 2025 FashionFinder</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Register button in the right section
    

    # Process login logic
    if login_button:
        if password == "123" and customer_id.isdigit():
            cid = int(customer_id)
            reviews, _ = load_data()
            if cid in reviews['Customer ID'].values:
                st.session_state.logged_in = True
                st.session_state.cid = cid
                st.session_state.page = 'main'
                st.rerun()
            else:
                st.error("Customer not found.")
        else:
            st.error("Invalid credentials.")
    
    # Handle register button click

def get_trending_products():
    # Load the datasets
    products_df = _, products_df = load_data()
    reviews_df = pd.read_csv('clothing_reviews.csv')

    # Find Best-Selling Products
    best_sellers = reviews_df.groupby('product_id')['Quantity'].sum().sort_values(ascending=False)

    # Top 10 Trending Product IDs
    top_trending_ids = best_sellers.head(10).index.tolist()

    # Count number of non-empty reviews per product
    review_counts = reviews_df[reviews_df['Review Text'].notnull()].groupby('product_id')['Review Text'].count().reset_index()
    review_counts.rename(columns={'Review Text': 'num_reviews'}, inplace=True)

    # Get product details for trending products
    trending_products = products_df[products_df['product_id'].isin(top_trending_ids)]

    # Add review count
    trending_products = pd.merge(trending_products, review_counts, on='product_id', how='left')
    trending_products['num_reviews'].fillna(0, inplace=True)
    trending_products['num_reviews'] = trending_products['num_reviews'].astype(int)

    # Add total quantity sold
    trending_products['total_quantity'] = trending_products['product_id'].map(best_sellers)

    # Order by sales volume
    trending_products = trending_products.sort_values(by='total_quantity', ascending=False)

    return trending_products




# ---------------------- Main Page ----------------------
def main_page():
    load_custom_css()
    reviews, products = load_data()
    matrix, sparse, model, sim_df, idx_to_cust, cust_to_idx = build_models(reviews)
    
    if st.button("👤 My Profile"):
        st.session_state.page = 'profile'
        st.rerun()
        return

    # --- Extract all possible keywords ---
    # --- Autocomplete-style Search Bar ---
    product_names = products['product_name'].dropna().unique().tolist()
    brand_names = products['product_brand'].dropna().unique().tolist()
    all_keywords = sorted(set(product_names + brand_names))

    search_query = st.text_input("🔍 Search Products or Brands", key="search_bar")

    if search_query:
        suggestions = [kw for kw in all_keywords if kw.lower().startswith(search_query.lower())]
        for match in suggestions[:5]:
            if st.button(match, key=f"suggest_{match}"):
                search_query = match
                st.session_state.search_bar = match  # update input field visually
                break

    # --- Show Search Results ---
    if search_query:
        filtered = products[
            products['product_name'].str.contains(search_query, case=False, na=False) |
            products['product_brand'].str.contains(search_query, case=False, na=False)
        ]

        st.markdown(f"### 🔎 Search Results for '**{search_query}**'")
        if filtered.empty:
            st.warning("No products found.")
        else:
            for _, row in filtered.iterrows():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(row['image_url'], width=100)
                with col2:
                    st.markdown(f"**{row['product_name']}**")
                    st.markdown(f"Brand: {row['product_brand']}")
                    if st.button("View Details", key=f"view_{row['product_id']}"):
                        st.session_state.selected_product = row['product_id']
                        st.session_state.page = 'details'
                        st.rerun()


    
    
    header_with_navigation()

    st.sidebar.header("Filters")
    brand = st.sidebar.multiselect("Brand", options=products['product_brand'].unique(), key="filter_brand")
    sub_category = st.sidebar.multiselect("Sub-Category", options=products['sub_category'].unique(), key="filter_subcategory")
    gender = st.sidebar.multiselect("Gender", options=products['gender'].unique(), key="filter_gender")
    price_range = st.sidebar.slider("Price Range", float(products['price'].min()), float(products['price'].max()), 
                                    (float(products['price'].min()), float(products['price'].max())), key="filter_price")
    sort_option = st.sidebar.selectbox("Sort By", ["Recommended", "Price: Low to High", "Price: High to Low", "Rating"], key="filter_sort")
    top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10, key="filter_topn")

    #st.sidebar.header("Recommendation Type")
    #method = st.sidebar.radio("Select Method", ["Hybrid", "User-Based", "Item-Based"])

    #if method == "Hybrid":
    recs = hybrid_recommend(st.session_state.cid, matrix, sparse, model, sim_df, idx_to_cust, cust_to_idx, products, top_n)
 

    if not recs.empty:
        if brand:
            recs = recs[recs['product_brand'].isin(brand)]
        if sub_category:
            recs = recs[recs['sub_category'].isin(sub_category)]
        if gender:
            recs = recs[recs['gender'].isin(gender)]
        recs = recs[(recs['price'] >= price_range[0]) & (recs['price'] <= price_range[1])]

        if sort_option == "Price: Low to High":
            recs = recs.sort_values(by="price")
        elif sort_option == "Price: High to Low":
            recs = recs.sort_values(by="price", ascending=False)
        elif sort_option == "Rating":
            recs['avg_rating'] = recs['product_id'].apply(lambda x: get_avg_rating(x, reviews))
            recs = recs.sort_values(by='avg_rating', ascending=False)

        # Load reviews to get customer name
        reviews, _ = load_data()
        customer_info = reviews[reviews['Customer ID'] == st.session_state.cid]

        if not customer_info.empty:
            customer_name = customer_info.iloc[0]['Customer Name']
            st.success(f"Welcome back, {customer_name}!")
        else:
            st.success(f"Welcome back, Customer #{st.session_state.cid}")

        cols = st.columns(3)
        for i, (_, prod) in enumerate(recs.iterrows()):
            with cols[i % 3]:
                render_product(prod, get_avg_rating(prod['product_id'], reviews))
    else:
        st.warning("No recommendations found for your profile.")

    st.markdown("<h2 class='trending-title'>🔥 Trending Products</h2>", unsafe_allow_html=True)
    
    # Get trending products
    trending_products = get_trending_products()
    
    
    if not trending_products.empty:
        with st.container():
            st.markdown("<div class='trending-container'>", unsafe_allow_html=True)
            
            # Create scrollable row of products using Streamlit columns
            # Show them 3 at a time in a row for better visibility
            num_products = min(9, len(trending_products))  # Limit to 9 products
            products_per_row = 3
            
            # Create rows as needed
            for i in range(0, num_products, products_per_row):
                cols = st.columns(products_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < num_products:
                        prod = trending_products.iloc[idx]
                        avg_rating = get_avg_rating(prod['product_id'], reviews)
                        sentiment_score = prod.get('sentiment_score', 0)
                        sentiment_display = f"{sentiment_score:.2f}" if sentiment_score else "N/A"
                        if sentiment_score >= 0.8:
                            sentiment_color = "#28a745"
                        elif sentiment_score >= 0.6:
                            sentiment_color = "#17a2b8"
                        elif sentiment_score >= 0.4:
                            sentiment_color = "#ffc107"
                        else:
                            sentiment_color = "#dc3545"
                        with col:
                            # Create product card
                            render_product(prod, get_avg_rating(prod['product_id'], reviews))
            
            st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ Discounted Products Section ------------------
    st.markdown("<h2 class='trending-title'>💸 Discounted Products</h2>", unsafe_allow_html=True)

    # Prepare discount logic
    reviews_df = pd.read_csv("clothing_reviews.csv")
    _, products_df = load_data()


    reviews_df['Purchase Date'] = pd.to_datetime(reviews_df['Purchase Date'], dayfirst=True)
    latest_date = reviews_df['Purchase Date'].max()
    three_months_ago = latest_date - pd.DateOffset(months=3)
    recent_df = reviews_df[reviews_df['Purchase Date'] >= three_months_ago]

    # Aggregate demand (product_id only)
    recent_demand = recent_df.groupby('product_id')['Quantity'].sum().reset_index()
    low_recent_demand = recent_demand.sort_values(by='Quantity').reset_index(drop=True).head(10)

    # Apply discount logic
    def get_discount_percentage(qty):
        if qty <= 1:
            return 0.30
        elif qty <= 2:
            return 0.20
        elif qty <= 3:
            return 0.10
        else:
            return 0.00

    low_recent_demand['Discount %'] = low_recent_demand['Quantity'].apply(get_discount_percentage)

    # Merge with products and calculate discounted price
    discounted_products = pd.merge(low_recent_demand, products_df, on='product_id', how='left')
    discounted_products['discounted_price'] = discounted_products['price'] * (1 - discounted_products['Discount %'])

    # Display products
    if not discounted_products.empty:
        with st.container():
            products_per_row = 3
            for i in range(0, len(discounted_products), products_per_row):
                cols = st.columns(products_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(discounted_products):
                        prod = discounted_products.iloc[idx]
                        avg_rating = get_avg_rating(prod['product_id'], reviews)

                        with col:
                            sentiment_score = prod.get('sentiment_score', 0)
                            sentiment_display = f"{sentiment_score:.2f}" if not pd.isna(sentiment_score) else "N/A"

                            if sentiment_score >= 0.8:
                                sentiment_color = "#28a745"
                            elif sentiment_score >= 0.6:
                                sentiment_color = "#17a2b8"
                            elif sentiment_score >= 0.4:
                                sentiment_color = "#ffc107"
                            else:
                                sentiment_color = "#dc3545"

                            st.markdown(f"""
                            <div class='product-card'>
                                <div class='product-image-container'>
                                    <img class='product-image' src='{prod['image_url']}' onerror="this.src='https://via.placeholder.com/300x400?text=No+Image';">
                                </div>
                                <div class='product-info'>
                                    <h4>{prod['product_name']}</h4>
                                    <p><strong>Brand:</strong> {prod['product_brand']}</p>
                                    <p>
                                        <strong>Price:</strong>
                                        <del style='color:#999;'>₹{prod['price']:.2f}</del>
                                        <span style='color: #28a745; font-weight: bold;'>₹{prod['discounted_price']:.2f}</span><br>
                                        <span style='font-size: 12px; color: #dc3545;'>
                                            You Save: ₹{prod['price'] - prod['discounted_price']:.2f} ({int(prod['Discount %'] * 100)}% OFF)
                                        </span>
                                    </p>
                                    <p><strong>Sentiment:</strong> 
                                        <span style="background-color: {sentiment_color}; color: white; padding: 3px 8px; border-radius: 4px; font-size: 12px;">
                                            {sentiment_display}
                                        </span>
                                    </p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            prod_id = prod['product_id']
                            in_wishlist = prod_id in [p['product_id'] for p in st.session_state.wishlist]
                            in_cart = prod_id in [p['product_id'] for p in st.session_state.cart]

                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"{'❤️ Added' if in_wishlist else '🤍 Wishlist'}", key=f"wish_discount_{prod_id}"):
                                    if not in_wishlist:
                                        st.session_state.wishlist.append(dict(prod))
                                        st.success(f"Added {prod['product_name']} to wishlist!")
                                    else:
                                        st.session_state.wishlist = [p for p in st.session_state.wishlist if p['product_id'] != prod_id]
                                        st.info(f"Removed from wishlist")
                                    st.rerun()
                            with col2:
                                if st.button(f"{'✓ In Cart' if in_cart else '🛒 Add to Cart'}", key=f"cart_discount_{prod_id}"):
                                    if not in_cart:
                                        st.session_state.cart.append(dict(prod))
                                        st.success(f"Added {prod['product_name']} to cart!")
                                    else:
                                        st.session_state.cart = [p for p in st.session_state.cart if p['product_id'] != prod_id]
                                        st.info(f"Removed from cart")
                                    st.rerun()

                            if st.button("View Details", key=f"view_discount_{prod_id}"):
                                prod_copy = dict(prod)
                                prod_copy['discount_percent'] = prod['Discount %']
                                prod_copy['discounted_price'] = prod['discounted_price']
                                st.session_state.selected_product = prod_id
                                st.session_state.discounted_product = prod_copy
                                st.session_state.page = 'details'
                                st.rerun()



    # --- Footer Section ---
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px; font-size: 14px;'>
        <p><strong>FashionFinder</strong> | Your AI-Powered Fashion Companion</p>
        <p>📞 Contact: +91-9876543210 | ✉️ Email: support@fashionfinder.com</p>
        <p>© 2025 FashionFinder. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
    

# ---------------------- Product Details Page ----------------------
def product_details_page():
    reviews, products = load_data()
    matrix, sparse, model, sim_df, idx_to_cust, cust_to_idx = build_models(reviews)

    # Ensure sim_df uses float product IDs
    sim_df.columns = sim_df.columns.astype(float)
    sim_df.index = sim_df.index.astype(float)

    pid = st.session_state.selected_product
    # Load discount-aware product if present
    if 'discounted_product' in st.session_state and st.session_state.discounted_product['product_id'] == pid:
        product = st.session_state.discounted_product
    else:
        product = products[products['product_id'] == pid].iloc[0]
        product = dict(product)  # Convert Series to dict for consistency


    
    # Get sentiment score for this product
    sentiment_score = product.get('sentiment_score', 0)
    sentiment_display = f"{sentiment_score:.2f}" if sentiment_score else "N/A"
    
    # Determine sentiment label
    if sentiment_score >= 0.8:
        sentiment_label = "Very Positive"
        sentiment_color = "#28a745"
    elif sentiment_score >= 0.6:
        sentiment_label = "Positive"
        sentiment_color = "#17a2b8"
    elif sentiment_score >= 0.4:
        sentiment_label = "Neutral"
        sentiment_color = "#ffc107"
    elif sentiment_score >= 0.2:
        sentiment_label = "Negative"
        sentiment_color = "#fd7e14"
    else:
        sentiment_label = "Very Negative"
        sentiment_color = "#dc3545"

    # Top navigation
    nav_col1, nav_col2, nav_col3 = st.columns([1, 8, 1])
    with nav_col1:
        if st.button("← Back to Products"):
            go_to_main()
    with nav_col3:
        col_wish, col_cart = st.columns(2)
        with col_wish:
            if st.button("❤️ "):
                go_to_wishlist()
        with col_cart:
            if st.button("🛒"):
                go_to_checkout()
    
    # --- Product Info ---
    # CSS for responsive image with height restriction
    # --- CSS to constrain image inside container properly ---
    st.markdown("""
    <style>
    .product-image-box {
        height: 720px;
        width: 100%;
        background-color: #fff;
        padding: 10px;
        border: 1px solid #eee;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }
    .product-image-box img {
        height: 100%;
        width: auto;
        object-fit: contain;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Layout with image and product info ---
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"""
        <div class="product-image-box">
            <img src="{product['image_url']}" onerror="this.src='https://via.placeholder.com/400x600?text=No+Image';" />
        </div>
        """, unsafe_allow_html=True)


    with right:
        st.title(product['product_name'])
        st.subheader(product['product_brand'])
        # --- Price Display with Discount Formatting ---
        original_price = product['price']
        discount = product.get('discount_percent', 0)
        discounted_price = product.get('discounted_price', original_price)

        if discount > 0:
            st.markdown(f"""
            <div style="margin-top: 10px;">
                <span style="font-size: 28px; font-weight: bold; color: #28a745;">₹{discounted_price:.2f}</span>
                <span style="text-decoration: line-through; color: #888; margin-left: 10px;">₹{original_price:.2f}</span><br>
                <span style="color: #dc3545; font-size: 14px;">You save ₹{original_price - discounted_price:.2f} ({int(discount * 100)}% OFF)</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="margin-top: 10px;">
                <span style="font-size: 26px; font-weight: bold; color: #ff3f6c;">₹{original_price:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        # --- Product Rating Display ---
        avg_rating = get_avg_rating(product['product_id'], reviews)

        # Rating badge
        rating_color = "#28a745" if avg_rating >= 4 else "#ffc107" if avg_rating >= 3 else "#dc3545"
        st.markdown(f"""
        <div style="display: inline-block; padding: 6px 12px; background-color: {rating_color}; color: white; border-radius: 6px; font-weight: bold; margin-top: 10px;">
            ⭐ {avg_rating:.1f} / 5
        </div>
        """, unsafe_allow_html=True)



        st.markdown(f"**Material:** {product['material']}")
        st.markdown(f"**Gender:** {product['gender']}")
        
        # --- Size Selection Buttons ---
        sizes = product['size'].split(',') if isinstance(product['size'], str) else []

        if 'selected_size' not in st.session_state:
            st.session_state.selected_size = sizes[0] if sizes else None

        # CSS for round bu
        st.markdown("**Select Size:**")

        # Use columns with tighter layout (e.g., 1 unit per size)
        cols = st.columns(len(sizes))
        for i, sz in enumerate(sizes):
            with cols[i]:
                if st.button(sz, key=f"size_btn_{sz}"):
                    st.session_state.selected_size = sz


        
        # Display sentiment information prominently
        st.markdown(f"""
        <div style="margin: 15px 0; padding: 10px; border-radius: 5px; background-color: {sentiment_color}20; border-left: 5px solid {sentiment_color};">
            <h3 style="margin: 0; color: {sentiment_color};">Customer Sentiment: {sentiment_label}</h3>
            <p style="margin: 5px 0 0 0;">Sentiment Score: {sentiment_display}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Description:** {product['product_description']}")
        col1, col2 = st.columns(2)
        # --- Action Buttons ---
        with col1:
            if st.button("🧡 Add to Wishlist"):
                pid = st.session_state.selected_product
                reviews, products = load_data()
                product = products[products['product_id'] == pid].iloc[0]

                # Only add if not already in wishlist
                if pid not in [p['product_id'] for p in st.session_state.wishlist]:
                    st.session_state.wishlist.append(dict(product))
                    st.success("Added to wishlist!")
                else:
                    st.info("Already in wishlist.")

        with col2:
            if st.button("🛒 Add to Cart"):
                if pid not in [p['product_id'] for p in st.session_state.cart]:
                    # Add selected size if available
                    cart_item = dict(product)
                    cart_item['selected_size'] = st.session_state.get('selected_size', '')
                    
                    st.session_state.cart.append(cart_item)
                    st.success("Added to cart!")
                else:
                    st.info("Already in cart.")

    # --- Top 5 Reviews ---
    st.subheader("Top 5 Reviews")
    top_reviews = reviews[reviews['product_id'] == pid].head(5)
    if not top_reviews.empty:
        for _, r in top_reviews.iterrows():
            # Calculate individual review sentiment
            review_sentiment = r.get('predicted_sentiment', r['Rating']/5.0)
            sent_color = "#28a745" if review_sentiment >= 0.7 else "#ffc107" if review_sentiment >= 0.4 else "#dc3545"
            
            st.markdown(f"""
            <div style="margin-bottom: 15px; padding: 10px; border-radius: 5px; background-color: #000000; border-left: 3px solid {sent_color};">
                <div style="display: flex; justify-content: space-between;">
                    <span>⭐ {r['Rating']}</span>
                    <span style="background-color: {sent_color}; color: black; padding: 2px 8px; border-radius: 10px; font-size: 12px;">
                        Sentiment: {review_sentiment:.2f}
                    </span>
                </div>
                <p><strong>{r['Customer Name']}</strong>: {r['Review Text']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No reviews for this product yet.")

    # --- Action Buttons ---


    st.markdown("### 🧥 Similar Products You Might Like")

    similar_items = products[
        (products['sub_category'] == product['sub_category']) &
        (products['product_id'] != product['product_id']) &
        (
            (products['material'] == product['material']) |
            (products['color'] == product['color'])
        )
    ].head(6)

    if not similar_items.empty:
        products_per_row = 3
        for i in range(0, len(similar_items), products_per_row):
            cols = st.columns(products_per_row)
            for j in range(products_per_row):
                idx = i + j
                if idx < len(similar_items):
                    prod = similar_items.iloc[idx]
                    avg_rating = get_avg_rating(prod['product_id'], reviews)
                    with cols[j]:
                        with st.container():  # ✅ This is critical
                            render_product(prod, avg_rating)
    else:
        st.info("No similar items found for this product.")









# ---------------------- Wishlist Page ----------------------
def wishlist_page():
    load_custom_css()
    reviews, _ = load_data()
    
    header_with_navigation()
    
    st.markdown("## My Wishlist")
    
    if not st.session_state.wishlist:
        st.info("Your wishlist is empty. Add products to your wishlist to see them here.")
        if st.button("Browse Products"):
            go_to_main()
    else:
        cols = st.columns(3)
        for i, product in enumerate(st.session_state.wishlist):
            with cols[i % 3]:
                avg_rating = get_avg_rating(product['product_id'], reviews)
                with st.container():
                    st.markdown(f"""
                    <div class='product-card'>
                        <div class='product-image-container'>
                            <img class='product-image' src='{product['image_url']}' onerror="this.src='https://via.placeholder.com/300x400?text=No+Image';">
                        </div>
                        <div class='product-info'>
                            <h4>{product['product_name']}</h4>
                            <p><strong>Brand:</strong> {product['product_brand']}</p>
                            <p><strong>Price:</strong> ₹{product['price']:.2f} <span class='rating-badge'>★ {avg_rating}</span></p>
                        </div>
                        <div class='product-actions'>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Remove", key=f"remove_wish_{product['product_id']}"):
                            st.session_state.wishlist = [p for p in st.session_state.wishlist if p['product_id'] != product['product_id']]
                            st.rerun()
                    with col2:
                        if st.button("Add to Cart", key=f"cart_wish_{product['product_id']}"):
                            if product['product_id'] not in [p['product_id'] for p in st.session_state.cart]:
                                st.session_state.cart.append(product)
                                st.success(f"Added to cart!")
                            else:
                                st.info("Already in cart")
                            st.rerun()
                    
                    if st.button("View Details", key=f"view_wish_{product['product_id']}"):
                        st.session_state.selected_product = product['product_id']
                        st.session_state.page = 'details'
                        st.rerun()
        
        if st.button("Continue Shopping", type="primary"):
            go_to_main()

# ---------------------- Checkout Page ----------------------
def checkout_page():
    load_custom_css()
    
    header_with_navigation()
    
    st.markdown("## Shopping Cart")
    # Auto-clear order flag if user left the checkout and came back
    if st.session_state.get("order_placed") and not st.session_state.cart:
        st.session_state.order_placed = False
    if not st.session_state.cart:
        st.info("Your cart is empty. Add products to your cart to checkout.")
    if st.button("Browse Products"):
        go_to_main()
    else:
        total = 0
        savings = 0

        for idx, product in enumerate(st.session_state.cart):
            # Ensure quantity exists
            if 'quantity' not in product:
                product['quantity'] = 1

            original_price = product['price']
            discounted_price = product.get('discounted_price', original_price)
            final_price = discounted_price
            quantity = product['quantity']

            with st.container():
                col1, col2, col3 = st.columns([1, 3, 2])
                with col1:
                    st.image(product['image_url'], width=100)
                with col2:
                    st.markdown(f"### {product['product_name']}")
                    st.markdown(f"**Brand:** {product['product_brand']}")
                    st.markdown(f"**Size:** {product.get('selected_size', 'N/A')}")
                    if original_price != final_price:
                        st.markdown(f"""
                        <span style='font-size:18px; color:#28a745; font-weight:bold;'>₹{final_price:.2f}</span>
                        <del style='color:#888; margin-left:10px;'>₹{original_price:.2f}</del>
                        <br><span style='color:#dc3545;'>You save ₹{(original_price - final_price):.2f} per item</span>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Price:** ₹{final_price:.2f}")

                    st.markdown(f"**Quantity:** {quantity}")
                    q_col1, q_col2, q_col3 = st.columns([1, 1, 2])
                    with q_col1:
                        if st.button("➕", key=f"increase_{product['product_id']}"):
                            product['quantity'] += 1
                            st.rerun()
                    with q_col2:
                        if quantity > 1:
                            if st.button("➖", key=f"decrease_{product['product_id']}"):
                                product['quantity'] -= 1
                                st.rerun()
                with col3:
                    if st.button("Remove", key=f"remove_cart_{product['product_id']}"):
                        st.session_state.cart = [p for p in st.session_state.cart if p['product_id'] != product['product_id']]
                        st.rerun()

            total += final_price * quantity
            savings += (original_price - final_price) * quantity



        st.markdown("### Order Summary")
        total_items = sum(p.get('quantity', 1) for p in st.session_state.cart)
        st.markdown(f"**Subtotal ({total_items} items):** ₹{total:.2f}")
        st.markdown(f"**You Saved:** ₹{savings:.2f}")
        shipping = 99 if total < 1000 else 0
        tax = total * 0.18
        grand_total = total + shipping + tax
        st.markdown(f"**Shipping:** ₹{shipping:.2f}")
        st.markdown(f"**Tax (18%):** ₹{tax:.2f}")
        st.markdown(f"**Grand Total:** ₹{grand_total:.2f}")



        # Shipping form
        st.markdown("### Shipping Information")
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Full Name")
            address1 = st.text_input("Address Line 1")
            address2 = st.text_input("Address Line 2")     
        with col2:
            city = st.text_input("City")
            state = st.text_input("State")
            pin = st.text_input("Pin Code")
            phone = st.text_input("Phone Number")

        st.markdown("### Payment Method")
        payment_method = st.radio("Select Payment Method", ["Credit/Debit Card", "UPI", "Cash on Delivery"])

        if payment_method == "Credit/Debit Card":
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Card Number")
                st.text_input("Name on Card")
            with col2:
                st.text_input("Expiry Date")
                st.text_input("CVV", type="password")
        elif payment_method == "UPI":
            st.text_input("UPI ID")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Continue Shopping"):
                go_to_main()
        with col2:
            if "order_placed" not in st.session_state:
                st.session_state.order_placed = False

            if not st.session_state.order_placed:
                if st.button("Place Order", key="place_order_button"):
                    st.session_state.cart = []
                    st.session_state.order_placed = True
                    st.success("✅ Order placed successfully! Your items will arrive in 3–5 business days.")
            else:
                
                if st.button("Continue Shopping", key="order_placed_continue"):
                    #st.session_state.order_placed = False
                    go_to_main()




def profile_page():
    reviews, _ = load_data()
    customer_data = reviews[reviews['Customer ID'] == st.session_state.cid].iloc[0]

    st.title("👤 My Profile")
    st.subheader("Customer Info")
    st.write(f"**Name:** {customer_data['Customer Name']}")
    st.write(f"**Age:** {customer_data['Customer Age']}")
    st.write(f"**Gender:** {customer_data['Gender']}")
    st.write(f"**Customer ID:** {customer_data['Customer ID']}")

    st.subheader("🛍️ Order History")
    orders = reviews[reviews['Customer ID'] == st.session_state.cid]
    if not orders.empty:
        st.dataframe(orders[['Purchase Date', 'product_name', 'Product Category', 'Quantity', 'Payment Method', 'Rating']])
    else:
        st.info("No order history available.")
    if st.button("🛍️ Continue Shopping"):
        st.session_state.page = 'main'
        st.rerun()

import subprocess

def run_admin_dashboard():
    st.markdown("### 🔒 Admin Dashboard")
    subprocess.Popen(["streamlit", "run", "das.py"])
    st.success("Admin dashboard launched in a new window.")

# ---------------------- Entry Point ----------------------
def main():
    init_session_state()
    
    # Handle URL parameters for page navigation
    query_params = st.query_params
    if 'page' in query_params:
        page_value = query_params['page']
        if isinstance(page_value, list):
            page_value = page_value[0]
        if page_value == 'wishlist':
            st.session_state.page = 'wishlist'
        elif page_value == 'checkout':
            st.session_state.page = 'checkout'
    
    if st.session_state.logged_in:
        if st.session_state.page == 'main':
            main_page()
        elif st.session_state.page == 'details':
            product_details_page()
        elif st.session_state.page == 'wishlist':
            wishlist_page()
        elif st.session_state.page == 'checkout':
            checkout_page()
        elif st.session_state.page == 'profile':
            profile_page()
    elif st.session_state.page == 'admin':
        run_admin_dashboard()
    else:
        login_page()

if __name__ == "__main__":
    main()