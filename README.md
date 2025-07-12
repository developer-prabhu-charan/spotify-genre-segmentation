# 🎵 Spotify Songs' Genre Segmentation

An automated system that uses a machine learning model to analyze a song's audio features and segment it into its most likely music genre. The final deployed application allows users to explore these genres, view sample songs, and discover sonically related music based on data-driven clustering.

-----

## 🚀 Live Demo

You can access and interact with the live application deployed on Hugging Face Spaces:

**[➡️ Live Application Link](https://prabhu-spotify-genre-segmentation.streamlit.app/)**


-----

## ✨ Key Features

  * **Live Prediction Mode:** Predicts a song's genre in real-time based on manually adjusted audio features.
  * **Demo Mode:** Loads pre-calculated average features for a selected genre to demonstrate typical sonic profiles.
  * **Dynamic UI:** Intelligently switches between Demo and Live modes based on user interaction with the sliders.
  * **Song Recommendations:** Displays a list of sample songs from the dataset that belong to the predicted genre.
  * **Related Genre Discovery:** Suggests other related genres based on the results of K-Means clustering, providing deeper insight into sonic similarity.

-----

## 🛠️ Project Methodology

This project followed a comprehensive end-to-end data science workflow:

#### 1\. Data Analysis & Pre-processing

  * The initial dataset containing Spotify song features was loaded and cleaned to handle missing values and duplicates.
  * Exploratory Data Analysis (EDA) was performed to understand the data distribution, including visualizing the count of songs per genre and the statistical properties of audio features.
  * A correlation matrix was generated to analyze the relationships between different audio features like `energy`, `danceability`, and `loudness`.

#### 2\. Clustering for Segmentation

  * To fulfill the core "segmentation" task, the K-Means clustering algorithm was applied to the scaled audio features.
  * The "Elbow Method" was used to determine that **k=6** was the optimal number of natural clusters in the data.
  * These clusters formed the basis for the "Related Genres" recommendation feature.

#### 3\. Modeling & Evaluation

  * Several classification models were trained to predict a song's genre from its audio features.
  * Models tested included `RandomForestClassifier` (from both scikit-learn and NVIDIA's cuML for GPU acceleration) and `LightGBM`.
  * After experimentation and hyperparameter tuning, the **LightGBM Classifier** was identified as the best-performing model, achieving an accuracy of **55.30%**. This accuracy reflects the significant sonic overlap between certain genres (like pop, r\&b, and latin) in the dataset.

#### 4\. Deployment

  * The final, trained LightGBM model, along with the corresponding scaler and label encoder, were saved as `.pkl` files.
  * An interactive web application was built using the **Streamlit** framework.
  * The application was deployed for public access on **Hugging Face Spaces**.

-----

## 💻 Technologies Used

  * **Programming Language:** Python
  * **Data Manipulation & Analysis:** pandas, numpy
  * **Machine Learning:** scikit-learn, LightGBM, RAPIDS cuML
  * **Data Visualization:** matplotlib, seaborn
  * **Web Framework:** Streamlit
  * **Deployment:** Hugging Face Spaces, Git

-----

## ⚙️ How to Run Locally

To run this application on your own machine, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://huggingface.co/spaces/your-username/your-space-name
    ```

2.  **Navigate to the project directory:**

    ```bash
    cd your-space-name
    ```

3.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```
