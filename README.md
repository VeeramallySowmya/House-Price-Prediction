 ABSTRACT

This project aims to predict house prices using both structured data (such as location, square footage, and amenities) and unstructured textual data (property descriptions). The project combines machine learning models and Natural Language Processing (NLP) techniques, resulting in a hybrid model that improves prediction accuracy. Using Python libraries such as Pandas, Scikit-learn, NLTK, and Matplotlib, the project includes data preprocessing, exploratory data analysis (EDA), model building, and evaluation to provide valuable insights for real estate stakeholders, potential buyers, and anyone interested in understanding property value dynamics more comprehensively.


A. INTRODUCTION

Problem Statement: This project addresses the challenge of accurately predicting housing prices by leveraging both structured and unstructured data, combining the traditional attributes of real estate with descriptions that can capture intangible property characteristics.
Objectives:
•	To analyze and preprocess both structured and unstructured data for model input.
•	To implement regression models and NLP techniques to achieve accurate price prediction.
•	To provide real estate insights based on the impact of amenities and property descriptions.
Significance: This model's insights can benefit stakeholders like real estate investors, agents, and potential buyers by providing a more comprehensive understanding of property value.

B. DATA DESCRIPTION AND PREPROCESSING

Dataset Overview
The dataset used for the house price prediction project consists of real estate listings from various locations in Pennsylvania, USA. Each entry represents a property with structured and unstructured features that provide essential details about the property and its surroundings. The dataset contains the following features:
•	Location: The city where the property is located (e.g., Pittsburgh, Philadelphia, Allentown, Harrisburg, Lancaster).
•	Sub-Area: More specific neighborhood or area within the city (e.g., Pittsburgh, Chestnut Hill Village).
•	Property Type: Type of property, such as 1 BHK, 2 BHK, 3 BHK, or 3 BHK Grand.
•	Property Area in Sq. Ft.: The size of the property, given in square feet.
•	Price in $ (lakhs): The price of the property in lakhs (1 lakh = 100,000 dollars).
•	Price in $ (millions): The price of the property in millions of dollars.
•	Company Name: The real estate company that is listing the property (e.g., Lennar Corporation, Toll Brothers).
•	Township Name / Society Name: The name of the township or society where the property is located (e.g., The Heights at Falling Water, Chestnut Hill Village).
•	Total Township Area in Acres: The area of the entire township, in acres.
•	Clubhouse, School/University, Hospital, Mall, Park/Jogging Track, Swimming Pool, Gym: Binary features indicating whether these amenities are available in the township.
•	Description: A textual field containing the description of the property, including key features, layout, and nearby amenities.

Data Cleaning and Preparation
The preprocessing pipeline focuses on cleaning and preparing both structured and unstructured data to make it suitable for models.
        Categorical and Continuous Data Cleaning
•	Categorical Variables (e.g., Location, Property Type):
o	Parsed data to create specific columns, such as splitting Location into State and Country.
o	Encoded binary amenities (e.g., Clubhouse, Swimming Pool, Gym) as 1/0 for model compatibility.
•	Continuous Variables (e.g., Property Area, Price):
o	Managed missing values in continuous features.
o	Standardized data formats for uniformity.
o	Addressed outliers in Property Area and Price by clipping extreme values to prevent skewing the model.
           Text Data Processing
•	Regex Cleaning: Applied regex to remove special characters and digits in text fields, particularly in property descriptions, to clean the text.
•	Stop Word Removal and Lowercasing: Converted text to lowercase and removed common stop words, reducing noise and improving consistency.
    Statistical Analysis
•	Univariate Analysis: Visualized individual feature distributions through histograms to identify patterns and detect skewness.
•	Multivariate Analysis: Used correlation and scatter plots to explore relationships between key features, such as Price and Property Area, to inform feature selection.
Outlier Treatment
•	Applied outlier treatment to reduce the impact of extreme values in Price and Property Area, enhancing data stability for model training.



C. FEATURE ENGINEERING

To improve predictive accuracy, several new features were created using both structured and unstructured data:
•	Price by Sub-Area: A new feature was generated based on the average price within each Sub-Area, capturing local pricing patterns. This was achieved by grouping the data by Sub-Area and calculating the mean price.
•	Amenities Score: A composite score was created by summing binary values for various amenities (e.g., Gym, Clubhouse, Swimming Pool). This score provided a measure of property richness based on available facilities.
•	Text Feature Engineering:
o	Parts of Speech (POS) Tagging: Property descriptions were processed to count nouns, adjectives, and verbs, emphasizing significant descriptive words that characterize property features.
o	N-grams: Extracted common bigrams (two-word combinations) from descriptions, such as "spacious bhk," to capture context-relevant phrases that add meaning to textual descriptions.
•	Average Price by Amenities Score: This feature was calculated by grouping data by Amenities Score and averaging the price, allowing the model to consider the pricing impact of amenities.


D. ML MODEL BUILDING

After preprocessing the data, several machine learning models were implemented to predict house prices, each offering unique strengths:

Data Preparation
The dataset was prepared by selecting relevant features for predicting house prices and renaming them for clarity.
The data was split into:
•	Training Set: 80% for training.
•	Testing Set: 20% for evaluation.
This split ensures models are trained on one subset and evaluated on unseen data.
Linear Regression
•	Purpose: Used as the baseline model to establish a relationship between features and house prices.
•	Outcome: Achieved an R² score of 0.76 on the training set and 0.68 on the test set. Confidence intervals were calculated to assess the reliability of predictions.
Regularization
To prevent overfitting and improve generalization, regularized models were applied:
•	Ridge Regression: Introduces L2 regularization to penalize large coefficients, improving stability without sacrificing much performance.
•	Lasso Regression: Applies L1 regularization, which performs automatic feature selection by shrinking coefficients to zero, simplifying the model.
Both models showed similar performance to Linear Regression while improving stability and interpretability.
Ensemble Method
•	Purpose: Combines the predictions from Linear, Ridge, and Lasso models to improve accuracy.
•	Outcome: The ensemble model achieved an R² score of 0.69 on the test set, providing more robust predictions by leveraging the strengths of each individual model.
Prediction Intervals
•	Purpose: Provides a range of possible values for each prediction, offering insights into the model's uncertainty.
•	Outcome: Prediction intervals were calculated and visualized for each model, helping to assess the reliability of predictions.
Model Deployment
The final ensemble model was serialized using Joblib and Pickle for easy deployment, enabling real-time predictions of house prices in future applications.


E. MODEL DEPLOYMENT
After training the model, it was deployed as a real-time prediction service through a user-friendly web interface:
•	APIs (RESTful): A RESTful API was developed using FastAPI, enabling users to send property details (e.g., location, area, etc.) and receive predictions for house prices. This allows the model to be accessed via HTTP requests for seamless integration into other systems.
•	Web Application Development: A web application was created using FastAPI to serve the trained model, allowing users to input property data and receive house price predictions. The application provides fast and efficient responses, ensuring users get real-time predictions.
•	Model Loading and Prediction: The trained model is loaded into the FastAPI application using Joblib. Predictions are made through the /predict endpoint, where input data is processed, passed to the model, and the predicted price is returned as a JSON response.
•	Prediction Adjustment: In the event of negative predictions, the model output is adjusted using numpy’s clip function to ensure predictions are always non-negative.


F. IMPLEMENTATION
Libraries and Tools
This project uses the following libraries and tools:
•	Pandas: For data manipulation, including loading, cleaning, and transforming data.
•	Scikit-learn: For implementing regression models (Linear Regression, Ridge, Lasso, and Voting Regressor) and evaluating model performance with metrics such as RMSE and MAE.
•	NLTK: Used for text preprocessing, including tokenization and feature extraction from property descriptions.
•	Matplotlib: For visualizing model performance, including prediction accuracy and error metrics.
•	Joblib: For loading the trained model into the FastAPI app for real-time predictions.

G. DISCUSSION
Challenges Faced
•	Data Collection: Gathering consistent property descriptions and managing missing data were initial challenges.
•	Preprocessing: Ensuring that all features were appropriately cleaned, encoded, and standardized was critical for effective model training.
•	Combining Structured and Unstructured Data: Balancing the contribution of both types of data (numerical and text) presented challenges, as each requires different preprocessing techniques.
Limitations
•	Dataset Size: The dataset used was relatively small, which limited its representativeness and the model's ability to generalize to new data or regions. A larger dataset would improve the model’s robustness and predictive accuracy.
•	Location Bias: The model is likely to perform best in the regions represented within the dataset. Predictions for properties in other areas may not be as accurate due to location-specific factors not captured in the data.
•	Text Complexity: Despite using NLP techniques, property descriptions might not fully capture all the relevant information that could influence house prices.
Future Work
•	Incorporating Additional Features: Adding more features, such as historical price trends, macroeconomic factors, and market sentiment, could help build a more comprehensive model. A larger and more diverse dataset would also improve predictive performance.

H. CONCLUSION
This project demonstrates the integration of structured and unstructured data to predict house prices using machine learning. By combining numerical features with property descriptions processed through NLP techniques, we achieved improved model accuracy. This project provided insights into which factors influence house prices and highlighted the potential for more sophisticated models to further refine predictions. The deployment of this model through FastAPI makes it accessible for real-time predictions, and future improvements can enhance its effectiveness.



