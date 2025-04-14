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

  ![image](https://github.com/user-attachments/assets/41ac5dcf-b31f-4fcb-879b-954d62885bc8)



