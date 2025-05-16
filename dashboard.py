import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
from datetime import datetime, timedelta
import warnings
import base64
from io import BytesIO

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Fashion Retail Analytics Dashboard",
    page_icon="👚",
    layout="wide",
)

# Custom styling
st.markdown("""
<style>
        
    .main-header {
        font-size: 2.5rem;
        color: #1a237e;
        font-weight: 800;
        margin-bottom: 1.5rem;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0d47a1;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: transparent;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #e8eaf6;
        border-radius: 7px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1565c0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #263238;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    .alert-box {
        background-color: #ffecb3;
        border-left: 5px solid #ffa000;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .discount-box {
        background-color: #c8e6c9;
        border-left: 5px solid #2e7d32;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for data loading and preprocessing
@st.cache_data
def load_data():
    """Load the dataset files"""
    products_df = pd.read_csv('clothing_description.csv')
    reviews_df = pd.read_csv('clothing_reviews.csv')
    
    # Clean and preprocess data
    reviews_df['Purchase Date'] = pd.to_datetime(reviews_df['Purchase Date'], format='%d-%m-%Y %H:%M')
    reviews_df.dropna(subset=['product_id', 'Quantity', 'Purchase Date'], inplace=True)
    
    return products_df, reviews_df

# Function to create a downloadable link
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href

# Load data
try:
    products_df, reviews_df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# -----------------------------------------------
# Customer Dashboard Components
# -----------------------------------------------

def trending_products_section():
    """Display the trending products section"""
    st.markdown('<div class="sub-header">🔥 Trending Products</div>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        # Calculate trending products
        best_sellers = reviews_df.groupby('product_id')['Quantity'].sum().sort_values(ascending=False)
        top_trending_ids = best_sellers.head(10).index.tolist()
        
        # Count number of non-empty reviews per product
        review_counts = reviews_df[reviews_df['Review Text'].notnull()].groupby('product_id')['Review Text'].count().reset_index()
        review_counts.rename(columns={'Review Text': 'num_reviews'}, inplace=True)
        
        # Get product details for trending products
        trending_products = products_df[products_df['product_id'].isin(top_trending_ids)].copy()
        
        # Merge with review counts
        trending_products = trending_products.merge(review_counts, on='product_id', how='left')
        trending_products['num_reviews'].fillna(0, inplace=True)
        trending_products['num_reviews'] = trending_products['num_reviews'].astype(int)
        
        # Add sales quantity
        trending_products['total_quantity'] = trending_products['product_id'].map(best_sellers)
        trending_products = trending_products.sort_values(by='total_quantity', ascending=False)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Create a bar chart for trending products
            fig = px.bar(
                trending_products.head(10),
                x='product_name',
                y='total_quantity',
                color='product_category',
                title='Top 10 Best-Selling Products',
                labels={'total_quantity': 'Units Sold', 'product_name': 'Product'},
                height=500,
                text='total_quantity'
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(255, 255, 255, 0.9)',
                margin=dict(l=20, r=20, t=50, b=20),
                yaxis_title='Total Quantity Sold',
                xaxis_title='',
                colorway=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Display metrics
            total_sales = trending_products['total_quantity'].sum()
            avg_rating = reviews_df['Rating'].mean()
            
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-value">{total_sales:,}</div>
                    <div class="metric-label">Total Units Sold</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"""
                <div class="metric-container" style="margin-top: 1rem;">
                    <div class="metric-value">{avg_rating:.1f}</div>
                    <div class="metric-label">Average Rating</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Display table of trending products with details
        st.markdown("<h4>Trending Products Details</h4>", unsafe_allow_html=True)
        st.dataframe(
            trending_products[['product_name', 'product_category', 'price', 'total_quantity', 'num_reviews']],
            hide_index=True,
            column_config={
                "product_name": "Product Name",
                "product_category": "Category",
                "price": st.column_config.NumberColumn("Price ($)", format="$%.2f"),
                "total_quantity": "Units Sold",
                "num_reviews": "Reviews"
            },
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

def product_category_analysis():
    """Display product category analysis"""
    st.markdown('<div class="sub-header">📊 Product Category Analysis</div>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns(2)
        
        # Category sales analysis
        category_sales = reviews_df.groupby('Product Category')['Quantity'].sum().reset_index()
        category_sales = category_sales.sort_values('Quantity', ascending=False)
        
        # Category ratings analysis
        category_ratings = reviews_df.groupby('Product Category')['Rating'].mean().reset_index()
        category_ratings['Rating'] = category_ratings['Rating'].round(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig = px.pie(
                category_sales,
                values='Quantity',
                names='Product Category',
                title='Sales by Product Category',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(
                legend_title='Category',
                margin=dict(l=20, r=20, t=50, b=20)
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig = px.bar(
                category_ratings,
                x='Product Category',
                y='Rating',
                title='Average Rating by Category',
                labels={'Rating': 'Average Rating'},
                color='Rating',
                color_continuous_scale='Viridis',
                text='Rating'
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(255, 255, 255, 0.9)',
                margin=dict(l=20, r=20, t=50, b=20),
                coloraxis_showscale=False
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

def customer_review_insights():
    """Display customer review insights"""
    st.markdown('<div class="sub-header">💬 Customer Review Insights</div>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns(2)
        
        # Calculate ratings distribution
        ratings_dist = reviews_df['Rating'].value_counts().sort_index()
        
        # Calculate age demographics
        age_groups = pd.cut(reviews_df['Customer Age'], bins=[0, 18, 25, 35, 50, 100], labels=['Under 18', '18-25', '26-35', '36-50', 'Over 50'])
        age_distribution = age_groups.value_counts().sort_index()
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig = px.bar(
                x=ratings_dist.index,
                y=ratings_dist.values,
                title='Rating Distribution',
                labels={'x': 'Rating', 'y': 'Number of Reviews'},
                color=ratings_dist.index,
                color_continuous_scale='RdYlGn',
                text=ratings_dist.values
            )
            fig.update_layout(
                plot_bgcolor='rgba(255, 255, 255, 0.9)',
                margin=dict(l=20, r=20, t=50, b=20),
                coloraxis_showscale=False
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig = px.pie(
                names=age_distribution.index,
                values=age_distribution.values,
                title='Customer Age Demographics',
                color_discrete_sequence=px.colors.sequential.Plasma_r
            )
            fig.update_layout(
                legend_title='Age Group',
                margin=dict(l=20, r=20, t=50, b=20)
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Display recent reviews
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h4>Recent Customer Reviews</h4>", unsafe_allow_html=True)
    
    # Filter for non-empty reviews and sort by date
    recent_reviews = reviews_df[reviews_df['Review Text'].notna()].sort_values('Purchase Date', ascending=False).head(5)
    
    for _, review in recent_reviews.iterrows():
        col1, col2 = st.columns([1, 4])
        with col1:
            # Display rating stars
            rating = int(review['Rating'])
            stars = "⭐" * rating
            st.markdown(f"#### {stars}")
            st.caption(f"{review['Purchase Date'].strftime('%b %d, %Y')}")
        
        with col2:
            st.markdown(f"**{review['product_name']}**")
            st.markdown(f"*{review['Customer Name']}*: {review['Review Text']}")
        
        st.divider()
    
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------
# Admin Dashboard Components
# -----------------------------------------------

def demand_forecast_section():
    """Display demand forecast section"""
    st.markdown('<div class="sub-header">📈 Demand Forecast (Next 6 Months)</div>', unsafe_allow_html=True)
    
    with st.container():
        # Add monthly aggregation and forecasting
        sales_monthly = reviews_df.copy()
        sales_monthly.set_index('Purchase Date', inplace=True)
        monthly_total_sales = sales_monthly.resample('M')['Quantity'].sum()
        
        # Drop the last month if it's not a full month
        last_date = sales_monthly.index.max()
        if last_date.day < 28:
            monthly_total_sales = monthly_total_sales[:-1]
        
        # Interpolate any 0s that may have been caused by missing data
        monthly_total_sales = monthly_total_sales.replace(0, pd.NA)
        monthly_total_sales = monthly_total_sales.interpolate()
        
        # Fit Exponential Smoothing model
        try:
            model = ExponentialSmoothing(monthly_total_sales, trend='add', seasonal='add', seasonal_periods=12).fit()
            
            # Forecast next 6 months
            forecast = model.forecast(6)
            
            # Merge historical and forecast for smoother plot
            full_series = pd.concat([monthly_total_sales, forecast])
            
            # Create plotly visualization
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=monthly_total_sales.index,
                y=monthly_total_sales.values,
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='#3498db', width=2),
                marker=dict(size=6)
            ))
            
            # Add forecast data
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode='lines+markers',
                name='Forecasted Sales',
                line=dict(color='#e74c3c', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            # Update layout
            fig.update_layout(
                title='Monthly Sales Forecast',
                xaxis_title='Month',
                yaxis_title='Total Units Sold',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                plot_bgcolor='rgba(255, 255, 255, 0.9)',
                margin=dict(l=20, r=20, t=50, b=20),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display forecast values in a table
            st.markdown("<h4>Forecasted Sales for Next 6 Months</h4>", unsafe_allow_html=True)
            
            forecast_df = forecast.reset_index()
            forecast_df.columns = ['Month', 'Forecasted Units']
            forecast_df['Month'] = forecast_df['Month'].dt.strftime('%B %Y')
            forecast_df['Forecasted Units'] = forecast_df['Forecasted Units'].round().astype(int)
            
            st.dataframe(
                forecast_df,
                hide_index=True,
                use_container_width=True
            )
            
            # Calculate the expected revenue based on average price
            avg_price = products_df['price'].mean()
            total_forecast = forecast.sum()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-container">
                        <div class="metric-value">{total_forecast:.0f}</div>
                        <div class="metric-label">Total Forecasted Units</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f"""
                    <div class="metric-container">
                        <div class="metric-value">${total_forecast * avg_price:,.2f}</div>
                        <div class="metric-label">Estimated Revenue</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
        except Exception as e:
            st.error(f"Error creating forecast: {e}")

def high_demand_products():
    """Display high demand products section"""
    st.markdown('<div class="sub-header">⚠️ High-Demand Products Alert</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="alert-box">', unsafe_allow_html=True)
        st.markdown("### Stock Alert for Next Month")
        st.markdown("The following products are predicted to have high demand next month and may require additional inventory.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prepare data
        reviews_tmp = reviews_df.copy()
        reviews_tmp.set_index('Purchase Date', inplace=True)
        
        # Group by product and month
        monthly_sales = reviews_tmp.groupby(['product_id', pd.Grouper(freq='M')])['Quantity'].sum().reset_index()
        
        alerts = []
        
        # Simplified forecasting approach for demo
        for product_id in monthly_sales['product_id'].unique():
            product_data = monthly_sales[monthly_sales['product_id'] == product_id].copy()
            
            if len(product_data) < 3:  # Need at least 3 months of data
                continue
                
            try:
                # Calculate recent growth rate
                product_data = product_data.sort_values('Purchase Date')
                recent_months = min(3, len(product_data))
                recent_data = product_data.tail(recent_months)
                
                avg_recent_sales = recent_data['Quantity'].mean()
                growth_rate = 1.0
                
                if len(recent_data) > 1:
                    first_month = recent_data['Quantity'].iloc[0]
                    last_month = recent_data['Quantity'].iloc[-1]
                    if first_month > 0:
                        growth_rate = last_month / first_month
                
                # Forecast next month with simple growth model
                predicted_demand = avg_recent_sales * growth_rate
                
                if predicted_demand >= 10:  # Threshold for high demand
                    alerts.append((product_id, round(predicted_demand, 2)))
                    
            except Exception as e:
                continue
        
        # Merge with product details
        if alerts:
            alerts_df = pd.DataFrame(alerts, columns=['product_id', 'forecasted_quantity'])
            final_alerts = pd.merge(alerts_df, products_df, on='product_id', how='left')
            
            # Create a new column for inventory suggestion
            final_alerts['Suggested Inventory'] = (final_alerts['forecasted_quantity'] * 1.2).round().astype(int)
            
            # Display alerts table
            st.dataframe(
                final_alerts[['product_id', 'product_name', 'product_category', 'forecasted_quantity', 'Suggested Inventory']],
                hide_index=True,
                column_config={
                    "product_id": "Product ID",
                    "product_name": "Product Name",
                    "product_category": "Category",
                    "forecasted_quantity": "Forecasted Demand",
                    "Suggested Inventory": "Suggested Inventory"
                },
                use_container_width=True
            )
            
            # Create downloadable report
            st.markdown(get_download_link(final_alerts, "high_demand_products_alert", "📥 Download High Demand Products Report"), unsafe_allow_html=True)
        else:
            st.info("No high-demand products detected for next month.")

def low_demand_products():
    """Display low demand products section"""
    st.markdown('<div class="sub-header">📉 Low-Demand Products (Discount Recommendations)</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="discount-box">', unsafe_allow_html=True)
        st.markdown("### Discount Strategy for Slow-Moving Inventory")
        st.markdown("The following products have shown low demand in the last 3 months and may require promotional discounts.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Determine the latest date in the dataset
        latest_date = reviews_df['Purchase Date'].max()
        
        # Filter data for the last 3 months from the latest purchase date
        three_months_ago = latest_date - pd.DateOffset(months=3)
        recent_df = reviews_df[reviews_df['Purchase Date'] >= three_months_ago]
        
        # Group by product_id and product_name, summing the total quantity sold
        recent_demand = recent_df.groupby(['product_id'])['Quantity'].sum().reset_index()
        
        # Merge with product names
        recent_demand = recent_demand.merge(products_df[['product_id', 'product_name', 'product_category', 'price']], on='product_id', how='left')
        
        # Sort by quantity to find low-demand products
        low_recent_demand = recent_demand.sort_values(by='Quantity').reset_index(drop=True)
        
        # Define discount strategy function
        def suggest_discount(qty):
            if qty <= 1:
                return "30% OFF"
            elif qty <= 2:
                return "20% OFF"
            elif qty <= 3:
                return "10% OFF"
            else:
                return "No Discount"
        
        # Apply the discount suggestion
        low_recent_demand['Suggested Discount'] = low_recent_demand['Quantity'].apply(suggest_discount)
        
        # Add a column for potential revenue impact
        low_recent_demand['Discount Percentage'] = low_recent_demand['Suggested Discount'].map({
            "30% OFF": 0.3, "20% OFF": 0.2, "10% OFF": 0.1, "No Discount": 0
        })
        
        low_recent_demand['Current Price'] = low_recent_demand['price']
        low_recent_demand['Discounted Price'] = low_recent_demand['Current Price'] * (1 - low_recent_demand['Discount Percentage'])
        
        # Filter to show only products that should be discounted
        discount_suggestions = low_recent_demand[low_recent_demand['Suggested Discount'] != "No Discount"]
        
        if not discount_suggestions.empty:
            # Create visualization
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create a bar chart comparing prices
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=discount_suggestions['product_name'],
                    y=discount_suggestions['Current Price'],
                    name='Current Price',
                    marker_color='#3498db'
                ))
                
                fig.add_trace(go.Bar(
                    x=discount_suggestions['product_name'],
                    y=discount_suggestions['Discounted Price'],
                    name='Discounted Price',
                    marker_color='#e74c3c'
                ))
                
                fig.update_layout(
                    title='Current vs. Discounted Prices',
                    xaxis_title='Product',
                    yaxis_title='Price ($)',
                    barmode='group',
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(255, 255, 255, 0.9)',
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Summary metrics
                num_products = len(discount_suggestions)
                avg_discount = discount_suggestions['Discount Percentage'].mean() * 100
                
                st.markdown(
                    f"""
                    <div class="metric-container">
                        <div class="metric-value">{num_products}</div>
                        <div class="metric-label">Products Needing Discount</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    f"""
                    <div class="metric-container" style="margin-top: 1rem;">
                        <div class="metric-value">{avg_discount:.1f}%</div>
                        <div class="metric-label">Average Discount</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Display discount suggestions table
            st.dataframe(
                discount_suggestions[['product_id', 'product_name', 'product_category', 'Quantity', 'Current Price', 'Suggested Discount', 'Discounted Price']],
                hide_index=True,
                column_config={
                    "product_id": "Product ID",
                    "product_name": "Product Name",
                    "product_category": "Category",
                    "Quantity": "Units Sold (Last 3 Months)",
                    "Current Price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
                    "Suggested Discount": "Recommended Discount",
                    "Discounted Price": st.column_config.NumberColumn("Discounted Price", format="$%.2f")
                },
                use_container_width=True
            )
            
            # Create downloadable report
            st.markdown(get_download_link(discount_suggestions, "discount_recommendations", "📥 Download Discount Recommendations"), unsafe_allow_html=True)
        else:
            st.info("No low-demand products requiring discounts were found.")

def inventory_management():
    """Display inventory management section"""
    st.markdown('<div class="sub-header">📦 Inventory Management</div>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h4>Inventory by Category</h4>", unsafe_allow_html=True)
            
            # Create a hypothetical inventory based on product catalog
            inventory_df = products_df.copy()
            # Simulate inventory levels based on price (just for demo purposes)
            inventory_df['inventory_level'] = (inventory_df['price'] * 0.5).apply(lambda x: max(5, min(50, int(x))))
            
            # Group by category
            category_inventory = inventory_df.groupby('product_category')['inventory_level'].sum().reset_index()
            
            # Create visualization
            fig = px.bar(
                category_inventory,
                x='product_category',
                y='inventory_level',
                title='Current Inventory by Category',
                color='inventory_level',
                labels={'inventory_level': 'Units in Stock', 'product_category': 'Category'},
                color_continuous_scale='Viridis',
                text='inventory_level'
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(255, 255, 255, 0.9)',
                margin=dict(l=20, r=20, t=50, b=20),
                coloraxis_showscale=False
            )
            
            fig.update_traces(textposition='outside')
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h4>Inventory Status</h4>", unsafe_allow_html=True)
            
            # Create a simulated inventory status
            def get_status(level):
                if level < 10:
                    return "Low Stock"
                elif level < 25:
                    return "Adequate" 
                else:
                    return "Well Stocked"
            
            inventory_df['status'] = inventory_df['inventory_level'].apply(get_status)
            status_counts = inventory_df['status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            
            # Create status pie chart
            fig = px.pie(
                status_counts,
                values='Count',
                names='Status',
                title='Inventory Status Overview',
                color='Status',
                color_discrete_map={
                    'Low Stock': '#e74c3c',
                    'Adequate': '#f39c12',
                    'Well Stocked': '#2ecc71'
                }
            )
            
            fig.update_layout(
                legend_title='Status',
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Low stock alerts
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h4>Low Stock Alerts</h4>", unsafe_allow_html=True)
    
    low_stock = inventory_df[inventory_df['status'] == 'Low Stock'].sort_values('inventory_level')
    
    if not low_stock.empty:
        st.dataframe(
            low_stock[['product_id', 'product_name', 'product_category', 'inventory_level']],
            hide_index=True,
            column_config={
                "product_id": "Product ID",
                "product_name": "Product Name",
                "product_category": "Category",
                "inventory_level": st.column_config.ProgressColumn(
                    "Current Stock",
                    help="Inventory level",
                    format="%d units",
                    min_value=0,
                    max_value=50
                )
            },
            use_container_width=True
        )
    else:
        st.info("No low stock items found.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def sales_performance():
    """Display sales performance section"""
    st.markdown('<div class="sub-header">💰 Sales Performance</div>', unsafe_allow_html=True)
    
    with st.container():
        # Time series of sales
        sales_ts = reviews_df.copy()
        sales_ts['Month'] = sales_ts['Purchase Date'].dt.to_period('M').astype(str)
        
        monthly_sales = sales_ts.groupby('Month')['Quantity'].sum().reset_index()
        
        # Create time series chart
        fig = px.line(
            monthly_sales,
            x='Month',
            y='Quantity',
            title='Monthly Sales Trend',
            markers=True,
            labels={'Quantity': 'Units Sold', 'Month': ''},
            color_discrete_sequence=['#d62728']  # High-contrast red
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            margin=dict(l=20, r=20, t=50, b=20),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Payment method distribution
            payment_dist = sales_ts['Payment Method'].value_counts().reset_index()
            payment_dist.columns = ['Payment Method', 'Count']
            
            fig = px.pie(
                payment_dist,
                values='Count',
                names='Payment Method',
                title='Payment Methods',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            fig.update_layout(
                legend_title='Method',
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Gender distribution
            gender_dist = sales_ts['Gender'].value_counts().reset_index()
            gender_dist.columns = ['Gender', 'Count']
            
            fig = px.pie(
                gender_dist,
                values='Count',
                names='Gender',
                title='Customer Gender Distribution',
                color_discrete_map={
                    'Male': '#3498db',
                    'Female': '#e84393',
                    'Other': '#f39c12'
                }
            )
            
            fig.update_layout(
                legend_title='Gender',
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Sales summary
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h4>Sales Summary</h4>", unsafe_allow_html=True)
        
        # Create key metrics
        total_orders = len(reviews_df['Customer ID'].unique())
        total_units = reviews_df['Quantity'].sum()
        avg_order_size = total_units / total_orders if total_orders > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-value">{total_orders:,}</div>
                    <div class="metric-label">Total Orders</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-value">{total_units:,}</div>
                    <div class="metric-label">Total Units Sold</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-value">{avg_order_size:.2f}</div>
                    <div class="metric-label">Avg. Units Per Order</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

def product_search_analysis():
    
    st.markdown('<div class="sub-header">🔍 Product Lookup & Forecast</div>', unsafe_allow_html=True)

    search_term = st.text_input("Search for a product by name", placeholder="e.g., Summer T-Shirt")

    if search_term:
        matches = products_df[products_df['product_name'].str.contains(search_term, case=False)]
        
        if not matches.empty:
            for _, product in matches.iterrows():
                pid = product['product_id']
                st.markdown(f"""
                    <div class="card" style="margin-top: 1rem;">
                        <h4>{product['product_name']}</h4>
                        <p><strong>Category:</strong> {product['product_category']}<br>
                        <strong>Price:</strong> ${product['price']:.2f}<br>
                        <strong>Product ID:</strong> {pid}</p>
                    </div>
                """, unsafe_allow_html=True)

                # Historical monthly sales
                sales_history = reviews_df[reviews_df['product_id'] == pid]
                if sales_history.empty:
                    st.info("No sales history available for this product.")
                    continue

                sales_history.set_index('Purchase Date', inplace=True)
                monthly_sales = sales_history['Quantity'].resample('M').sum()
                monthly_sales = monthly_sales.interpolate()

                # Historical chart
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(
                    x=monthly_sales.index,
                    y=monthly_sales.values,
                    mode='lines+markers',
                    name='Historical Sales',
                    line=dict(color='#2c3e50', width=2),
                    marker=dict(size=6)
                ))
                fig_hist.update_layout(
                    title='Historical Monthly Sales',
                    xaxis_title='Month',
                    yaxis_title='Units Sold',
                    xaxis=dict(
                        title=dict(font=dict(color='black', size=13)),
                        tickfont=dict(color='black')
                    ),
                    yaxis=dict(
                        title=dict(font=dict(color='black', size=13)),
                        tickfont=dict(color='black')
                    ),
                    font=dict(color='black', size=12),
                    legend=dict(
                        title='',
                        font=dict(color='black'),
                        bgcolor='rgba(255,255,255,0.7)',
                        bordercolor='black',
                        borderwidth=1
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # Forecast
                try:
                    model = ExponentialSmoothing(monthly_sales, trend='add', seasonal='add', seasonal_periods=12).fit()
                    forecast = model.forecast(6)

                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(
                        x=monthly_sales.index,
                        y=monthly_sales.values,
                        mode='lines',
                        name='Historical',
                        line=dict(color='#3498db')
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast.index,
                        y=forecast.values,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#e74c3c', dash='dash'),
                        marker=dict(size=6)
                    ))
                    fig_forecast.update_layout(
                        title='Forecasted Sales (Next 6 Months)',
                        xaxis_title='Month',
                        yaxis_title='Forecasted Units',
                        xaxis=dict(
                            title=dict(font=dict(color='black', size=13)),
                            tickfont=dict(color='black')
                        ),
                        yaxis=dict(
                            title=dict(font=dict(color='black', size=13)),
                            tickfont=dict(color='black')
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='black', size=12),
                        legend=dict(
                            title='',
                            font=dict(color='black'),
                            bgcolor='rgba(255,255,255,0.7)',
                            bordercolor='black',
                            borderwidth=1
                        ),
                        margin=dict(l=20, r=20, t=50, b=20),
                        height=400
                    )

                    st.plotly_chart(fig_forecast, use_container_width=True)
                except:
                    st.warning("Forecasting failed for this product.")
        else:
            st.info("No matching products found.")


# -----------------------------------------------
# Main Application
# -----------------------------------------------

def main():
    # Main header
    st.markdown('<h1 class="main-header">Fashion Retail Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Create tabs for customer and admin views
    tab1, tab2 = st.tabs(["👥 Customer View", "🔐 Admin Dashboard"])
    
    with tab1:
        trending_products_section()
        product_category_analysis()
        customer_review_insights()
    
    with tab2:
        demand_forecast_section()
        high_demand_products()
        low_demand_products()
        inventory_management()
        sales_performance()
        product_search_analysis() 

if __name__ == "__main__":
    main()