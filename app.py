import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import streamlit as st
import base64

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
    return reviews, products

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
        f"""
        <div class='header'>
            <h1>🛍️ FashionFinder</h1>
            <p>Your AI-powered Personal Stylist</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Use Streamlit's native components for navigation instead of JavaScript
    col1, col2, col3 = st.columns([5, 1, 1])
    with col2:
        if st.button(f"❤️ Wishlist ({len(st.session_state.wishlist)})", use_container_width=True):
            go_to_wishlist()
    with col3:
        if st.button(f"🛒 Cart ({len(st.session_state.cart)})", use_container_width=True):
            go_to_checkout()

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
    combined = combined.drop(rated_items, errors='ignore').sort_values(ascending=False)
    return products[products['product_id'].isin(combined.head(top_n).index)]

def user_based_recommend(customer_id, matrix, sparse, model, idx_to_cust, cust_to_idx, products, top_n=10):
    if customer_id not in cust_to_idx:
        return pd.DataFrame()
    idx = cust_to_idx[customer_id]
    dists, inds = model.kneighbors(sparse[idx], n_neighbors=6)
    neighbors = [idx_to_cust[i] for i in inds.flatten()[1:]]
    neighbor_scores = matrix.loc[neighbors].mean()
    user_scores = matrix.loc[customer_id]
    unrated = user_scores[user_scores == 0]
    scores = neighbor_scores[unrated.index].sort_values(ascending=False)
    return products[products['product_id'].isin(scores.head(top_n).index)]

def item_based_recommend(customer_id, matrix, sim_df, products, top_n=10):
    if customer_id not in matrix.index:
        return pd.DataFrame()
    user_scores = matrix.loc[customer_id]
    rated_items = user_scores[user_scores > 0].index
    item_scores = pd.Series(0, index=matrix.columns)
    for item in rated_items:
        item_scores += sim_df[item] * user_scores[item]
    item_scores = item_scores.drop(rated_items, errors='ignore').sort_values(ascending=False)
    return products[products['product_id'].isin(item_scores.head(top_n).index)]

# ---------------------- UI Components ----------------------
def render_product(product, avg_rating):
    prod_id = product['product_id']
    in_wishlist = prod_id in [p['product_id'] for p in st.session_state.wishlist] if st.session_state.wishlist else False
    in_cart = prod_id in [p['product_id'] for p in st.session_state.cart] if st.session_state.cart else False
    
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
    background-image: url("https://t4.ftcdn.net/jpg/05/96/62/65/360_F_596626503_jrzjZNYStDexiWxQFqO7oCh6M8PdMlJs.jpg");
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
  

# ---------------------- Main Page ----------------------
def main_page():
    load_custom_css()
    reviews, products = load_data()
    matrix, sparse, model, sim_df, idx_to_cust, cust_to_idx = build_models(reviews)

    if st.button("👤 My Profile"):
        st.session_state.page = 'profile'
        st.rerun()
        return

    search = st.text_input("🔎 Search for products", "")
    if search:
        filtered = products[products['product_name'].str.contains(search, case=False, na=False)]
        if not filtered.empty:
            st.subheader(f"Search Results for '{search}'")
            for _, prod in filtered.iterrows():
                if st.button(f"View Details - {prod['product_id']}"):
                    st.session_state.selected_product = prod['product_id']
                    st.session_state.page = 'details'
                render_product(prod, reviews)
        else:
            st.warning("No matching products found.")
        return
    header_with_navigation()

    st.sidebar.header("Filters")
    brand = st.sidebar.multiselect("Brand", options=products['product_brand'].unique())
    sub_category = st.sidebar.multiselect("Sub-Category", options=products['sub_category'].unique())
    gender = st.sidebar.multiselect("Gender", options=products['gender'].unique())
    price_range = st.sidebar.slider("Price Range", float(products['price'].min()), float(products['price'].max()), (float(products['price'].min()), float(products['price'].max())))
    sort_option = st.sidebar.selectbox("Sort By", ["Recommended", "Price: Low to High", "Price: High to Low", "Rating"])
    top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

    st.sidebar.header("Recommendation Type")
    method = st.sidebar.radio("Select Method", ["Hybrid", "User-Based", "Item-Based"])

    if method == "Hybrid":
        recs = hybrid_recommend(st.session_state.cid, matrix, sparse, model, sim_df, idx_to_cust, cust_to_idx, products, top_n)
    elif method == "User-Based":
        recs = user_based_recommend(st.session_state.cid, matrix, sparse, model, idx_to_cust, cust_to_idx, products, top_n)
    else:
        recs = item_based_recommend(st.session_state.cid, matrix, sim_df, products, top_n)

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

        st.success(f"Welcome back, Customer #{st.session_state.cid}")
        cols = st.columns(3)
        for i, (_, prod) in enumerate(recs.iterrows()):
            with cols[i % 3]:
                render_product(prod, get_avg_rating(prod['product_id'], reviews))
    else:
        st.warning("No recommendations found for your profile.")

    

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
    product = products[products['product_id'] == pid].iloc[0]

    st.title(product['product_name'])

    # --- Product Info ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(product['image_url'], width=300)
    with col2:
        st.subheader(product['product_brand'])
        st.markdown(f"**Price:** ₹{product['price']:.2f}")
        st.markdown(f"**Material:** {product['material']}")
        st.markdown(f"**Gender:** {product['gender']}")
        st.markdown(f"**Sizes:** {product['size']}")
        st.markdown(f"**Description:** {product['product_description']}")

    # --- Top 5 Reviews ---
    st.subheader("Top 5 Reviews")
    top_reviews = reviews[reviews['product_id'] == pid].head(5)
    if not top_reviews.empty:
        for _, r in top_reviews.iterrows():
            st.write(f"⭐ {r['Rating']} - {r['Customer Name']}: {r['Review Text']}")
    else:
        st.info("No reviews for this product yet.")

    # --- Action Buttons ---
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧡 Add to Wishlist"):
            st.session_state.page = 'wishlist'
            st.rerun()
    with col2:
        if st.button("🛒 Buy Now"):
            st.session_state.page = 'checkout'
            st.rerun()
    if st.button("Back to Products"):
        st.session_state.page = 'main'
        st.rerun()

    # --- Similar Products Section ---
    st.markdown("### 🧥 Similar Products You Might Like")

    similar_items = products[
        (products['sub_category'] == product['sub_category']) &
        (products['product_id'] != product['product_id']) &  # exclude current
        (
            (products['material'] == product['material']) |
            (products['color'] == product['color'])
        )
    ].head(6)

    if not similar_items.empty:
        cols = st.columns(3)
        for i, (_, prod) in enumerate(similar_items.iterrows()):
            with cols[i % 3]:
                render_product(prod, get_avg_rating(prod['product_id'], reviews))

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
    
    if not st.session_state.cart:
        st.info("Your cart is empty. Add products to your cart to checkout.")
        if st.button("Browse Products"):
            go_to_main()
    else:
        total = 0
        for product in st.session_state.cart:
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    st.image(product['image_url'], width=100)
                with col2:
                    st.markdown(f"### {product['product_name']}")
                    st.markdown(f"**Brand:** {product['product_brand']}")
                    st.markdown(f"**Price:** ₹{product['price']:.2f}")
                with col3:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if st.button("Remove", key=f"remove_cart_{product['product_id']}"):
                        st.session_state.cart = [p for p in st.session_state.cart if p['product_id'] != product['product_id']]
                        st.rerun()
                st.markdown("---")
                total += product['price']
        
        st.markdown("### Order Summary")
        st.markdown(f"**Subtotal ({len(st.session_state.cart)} items):** ₹{total:.2f}")
        st.markdown(f"**Shipping:** ₹{99 if total < 1000 else 0:.2f}")
        st.markdown(f"**Tax:** ₹{total * 0.18:.2f}")
        grand_total = total + (99 if total < 1000 else 0) + (total * 0.18)
        st.markdown(f"**Grand Total:** ₹{grand_total:.2f}")
        
        st.markdown("### Shipping Information")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Full Name")
            st.text_input("Address Line 1")
            st.text_input("Address Line 2")
        with col2:
            st.text_input("City")
            st.text_input("State")
            st.text_input("Pin Code")
            st.text_input("Phone Number")
        
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
            if st.button("Place Order", type="primary"):
                st.success("Order placed successfully! Your order will be delivered in 3-5 business days.")
                st.session_state.cart = []
                st.rerun()

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
    else:
        login_page()

if __name__ == "__main__":
    main()
