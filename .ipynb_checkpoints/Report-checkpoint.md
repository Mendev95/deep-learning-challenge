# Alphabet Soup Charity Funding Success Prediction Model

## Overview of the Analysis

The Alphabet Soup Charity, a prestigious philanthropic organization, sought to enhance its funding allocation process through data-driven decision-making. The goal was to develop a predictive model capable of identifying the applications most likely to succeed if granted funding. Leveraging a dataset with over 34,000 past applications, this analysis aimed to create a binary classifier using deep learning techniques to forecast the effective use of the granted funds.

## Results

### Data Preprocessing

- **Target Variable:** The model targets the `IS_SUCCESSFUL` column, aiming to predict the likelihood of a successful funding outcome.
- **Feature Variables:** Critical features for the model encompass `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, and `ASK_AMT`. These variables provide a comprehensive overview of each application's context and intent.
- **Variables Removed:** The `EIN` and `NAME` columns were excluded from the analysis. As identifiers, they do not offer predictive value or insights into the success of applications.

### Compiling, Training, and Evaluating the Model

The neural network's architecture was meticulously designed to balance complexity and performance, informed by the nature of the data and the predictive task:

- **Neural Network Architecture:** The model featured an input layer sized to match the number of features, several hidden layers with varying numbers of neurons, and an output layer with a single neuron using a sigmoid activation function for binary classification.
- **Activation Functions:** The ReLU activation function was chosen for hidden layers due to its effectiveness in avoiding vanishing gradient problems and speeding up training. The sigmoid function in the output layer facilitates binary outcome prediction.
- **Model Performance:** Initial models hovered around the 72-73% accuracy mark. Through optimization, including architectural adjustments and hyperparameter tuning, improvements were observed. However, consistently surpassing the 75% accuracy benchmark proved challenging, highlighting the complexity of the problem and the dataset's inherent limitations.

<img width="592" alt="Screenshot 2024-03-11 at 8 48 46 PM" src="https://github.com/NidaB-C/deep-learning-challenge/assets/147389952/9d74d994-da60-4d1d-b0e8-4abd57c76039">
  
<img width="623" alt="Screenshot 2024-03-11 at 8 48 59 PM" src="https://github.com/NidaB-C/deep-learning-challenge/assets/147389952/15157e3a-2c87-4d52-9760-5e17d087bc05">


### Model Optimization Strategies

Efforts to enhance model efficacy included:

- **Data Preprocessing Adjustments:** Additional binning of categorical variables and removal of less informative features aimed to simplify the model's learning task.
- **Architectural Tweaks:** Increasing the number of neurons and layers aimed to capture more complex relationships in the data. However, care was taken to avoid overfitting, which could diminish model performance on unseen data.
- **Training Adjustments:** Experimentation with the number of epochs and batch sizes sought to find an optimal balance between sufficient learning and computational efficiency.

<img width="588" alt="Screenshot 2024-03-11 at 8 45 54 PM" src="https://github.com/NidaB-C/deep-learning-challenge/assets/147389952/d1c4640d-1320-41af-8104-d7d1e16a1bf1">
<img width="633" alt="Screenshot 2024-03-11 at 8 47 14 PM" src="https://github.com/NidaB-C/deep-learning-challenge/assets/147389952/2cdaa59d-faa4-4f8f-938d-c4f372746f33">

## Summary and Recommendations

The development of the Alphabet Soup Charity funding success prediction model illustrates the potential of deep learning in enhancing philanthropic decision-making. While the model achieved moderate success, it underscores the challenges inherent in predictive modelling - particularly when aiming for high accuracy in complex, real-world scenarios.

### Alternative Model Recommendation

Considering the challenges faced, an exploration of **Random Forest Classifier** could be beneficial for several reasons:

- **Robustness to Overfitting:** Random Forest can manage a high-dimensional feature space without severe overfitting, making it suitable for complex datasets.
- **Feature Importance Insights:** It provides clarity on which features most significantly impact predictions, offering strategic insights for Alphabet Soup.
- **Versatility and Performance:** Known for its high accuracy and ability to handle both categorical and continuous data, Random Forest might capture the nuances of the dataset more effectively than a single deep learning model.

### Further Steps

- **Feature Engineering:** Creating new features that encapsulate more nuanced information could enhance model accuracy.
- **Model Ensembles:** Combining predictions from multiple models could leverage their respective strengths, potentially surpassing the accuracy achievable by any single model.
- **Exploration of Advanced Techniques:** Investigating other algorithms like XGBoost could offer performance improvements and faster training times.

The journey to refine Alphabet Soup's predictive capabilities is ongoing. Continued exploration, experimentation, and the integration of machine learning insights into strategic decision-making processes promise to significantly impact the organization's ability to fund projects with the highest potential for positive change.

