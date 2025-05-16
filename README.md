# Recommendation System with Sentiment and Demand Analysis

## Overview

This project builds a recommendation system that goes beyond simple collaborative or content-based filtering. It incorporates sentiment analysis of user reviews and demand analysis of items to provide more nuanced and relevant recommendations.

The system follows these key steps:

1.  **Sentiment Analysis:** Analyzes user reviews to determine the overall sentiment (positive, negative, neutral) towards items. This helps in understanding user opinions and preferences more deeply than just ratings.
2.  **Demand Analysis:** Estimates the current demand for different items. This can be based on factors like purchase frequency, view counts, or other relevant metrics. Incorporating demand ensures recommendations include popular and trending items.
3.  **Combined Recommendation:** Integrates sentiment and demand information with traditional recommendation techniques:
    * **Collaborative Filtering:** Recommends items based on the preferences of users with similar tastes.
    * **Content-Based Filtering:** Recommends items similar to those a user has liked in the past.
    * **Hybrid Approach:** Combines collaborative and content-based filtering, potentially weighted by sentiment and demand scores, to provide more robust and diverse recommendations.

## Tech Stack

* **Python:** The primary programming language used for the entire project.

## Libraries Used

* **pandas:** For data manipulation and analysis.
* **scikit-learn (sklearn):**
    * `NearestNeighbors`: For implementing k-nearest neighbors algorithm used in collaborative filtering.
    * `cosine_similarity`: For calculating the similarity between items or users.
* **scipy:**
    * `csr_matrix`: For creating sparse matrices, which are efficient for collaborative filtering on large datasets.
* **streamlit:** For creating an interactive web interface to showcase the recommendation system.
* **base64:** Likely used for encoding and decoding data, possibly for handling image display or other media within the Streamlit application.

  ![image](https://github.com/user-attachments/assets/eb8066bc-e95c-412a-9bd8-9dfeeb8cfb9c)
  ![image](https://github.com/user-attachments/assets/0670810b-0866-41ac-928e-c16892805652)


  ## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **Install the required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to create a `requirements.txt` file if it doesn't exist. You can generate it using `pip freeze > requirements.txt` after installing the necessary libraries.)*

3.  **Prepare the data:**
    * Ensure you have the necessary data files (e.g., `reviews.csv`, `items.csv`, `user_interactions.csv`) in the `data/` directory or adjust the file paths in the code accordingly.
    * The data should be structured in a way that allows for sentiment analysis (e.g., user ID, item ID, review text), demand analysis (e.g., item ID, purchase count/view count), and recommendation algorithms (e.g., user ID, item ID, rating/interaction).

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run src/app.py
    ```
    This command will typically open the application in your web browser.

2.  **Interact with the recommendation system:**
    * You should be able to browse items.
    * Potentially, you can input user IDs or select items to get personalized recommendations.
    * The interface might display the sentiment scores and demand metrics alongside the recommendations.

## Key Components and Functionality

* **Sentiment Analysis (`sentiment_analysis.py`):**
    * Reads user reviews.
    * Implements a sentiment analysis technique (e.g., using libraries like NLTK, spaCy, or pre-trained models).
    * Calculates a sentiment score (e.g., F-score or a numerical value representing positivity/negativity) for each item based on the reviews.
    * The F-score calculation from reviews likely involves treating positive/negative sentiment as classes and evaluating the performance of the sentiment analysis model.

* **Demand Analysis (`demand_analysis.py`):**
    * Analyzes user interaction data (e.g., purchases, views).
    * Calculates a demand score for each item, reflecting its popularity or current trend.

* **Collaborative Filtering (`collaborative_filtering.py`):**
    * Implements user-based or item-based collaborative filtering using `NearestNeighbors` and `cosine_similarity`.
    * Utilizes user-item interaction data (e.g., ratings).
    * Generates recommendations based on the similarity of user preferences or item characteristics.
    * Likely uses `csr_matrix` for efficient computation on sparse interaction data.

* **Content-Based Filtering (`content_based_filtering.py`):**
    * Analyzes item features (from `items.csv`).
    * Calculates the similarity between items based on their content (e.g., using TF-IDF and cosine similarity).
    * Recommends items similar to those a user has interacted with positively.

* **Hybrid Recommendation (`hybrid_recommendation.py`):**
    * Combines the results from collaborative filtering, content-based filtering, sentiment analysis, and demand analysis.
    * Implements a strategy to weigh and integrate these different factors to generate the final recommendations. This could involve averaging scores, using weighted sums, or more complex techniques.

* **Streamlit Application (`app.py`):**
    * Creates a user-friendly web interface using the `streamlit` library.
    * Allows users to interact with the recommendation system.
    * Visualizes recommendations and potentially displays sentiment and demand information.
    * Uses `base64` for encoding/decoding, possibly for displaying images or other static assets.

## Potential Improvements and Future Work

* **Advanced Sentiment Analysis:** Explore more sophisticated sentiment analysis techniques, including aspect-based sentiment analysis to understand which features users like or dislike.
* **Dynamic Demand Analysis:** Implement real-time demand tracking and incorporate temporal factors.
* **User Feedback Integration:** Allow users to provide feedback on recommendations to improve the system's accuracy over time.
* **Scalability:** Consider techniques for handling larger datasets and user bases.
* **Evaluation Metrics:** Implement rigorous evaluation metrics to quantify the performance of the recommendation system and the impact of incorporating sentiment and demand.
* **More Hybrid Approaches:** Experiment with different hybrid recommendation strategies (e.g., switching, mixed, feature augmentation).
* **Explainability:** Add features to explain why certain items are recommended.

## Contributing

*(If this is an open-source project, include guidelines for contributions.)*

## License

*(Include the project's license information.)*



