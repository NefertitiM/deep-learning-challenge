# deep-learning-challenge

Machine learning and neural networks were used to build a tool for the nonprofit foundation, Alphabet Soup, that can help predict the success of organizations applying for their funding. A CSV with more than 34,000 organizations funded by Alphabet Soup, over several years, was provided. Key features in the dataet was used to create a binary classifier that correlate with successful funded organization and can predict successful applicants.
 
---

## Results

### Data Preprocessing

- **Target Variable**: 
  - `IS_SUCCESSFUL`: This variable indicates if the funding was used effectively (1 for successful, 0 for unsuccessful) and was chosen as the target variable.

- **Feature Variables**: 
  - `APPLICATION_TYPE`: Alphabet Soup application type.
  - `AFFILIATION`: Affiliated sector of industry.
  - `CLASSIFICATION`: Government organization classification.
  - `USE_CASE`: Use case for funding.
  - `ORGANIZATION`: Organization type.
  - `STATUS`: Active status.
  - `INCOME_AMT`: Income classification.
  - `SPECIAL_CONSIDERATIONS`: Special considerations for application.
  - `ASK_AMT`: Funding amount requested.

    All of these variables were chosen as features because they have impact the success of the organizations.
    
- **Variables to Remove**: 
  - `EIN`: Identification number of the organization.
  - `NAME`: Name of the organization.

     These variables were dropped because they don't tell us anything about the potential the success of the organizations.
---

### Compiling, Training, and Evaluating the Model

- **Neurons, Layers, and Activation Functions**:
  - **Neurons**: 
    - First hidden layer: 64 neurons
    - Second hidden layer: 32 neurons
  - **Layers**: 
    - Input layer
    - Two hidden layers
    - Output layer
  - **Activation Functions**: 
    - ReLU for hidden layers (to introduce non-linearity)
    - Sigmoid for the output layer (to predict probabilities)

- **Achieved Target Model Performance**: 
  - Initially achieved approximately 72.65% accuracy.
  - Target performance of over 75% was not achieved in the first attempt.

- **Steps Taken to Increase Model Performance**:
  - Experimented with different architectures (adding layers and neurons).
  - Adjusted activation functions and learning rates.
  - Increased the number of epochs to allow for more training time.
  - Utilized dropout layers to reduce overfitting.
  - Conducted data preprocessing to handle categorical variables effectively.

---

## Summary

The deep learning model developed in this analysis demonstrated a solid foundation for predicting the success of funding applications, achieving an initial accuracy of approximately 72.65%. However, this was below the desired threshold of 75%. 

### Recommendation for Different Models
To potentially enhance predictive performance, I recommend exploring the following alternatives:

1. **Random Forest Classifier**: 
   - **Pros**: Handles categorical variables well, reduces overfitting through ensemble methods, and provides feature importance metrics.
   - **Explanation**: Random Forest can capture complex interactions between features without extensive tuning, making it a robust option for this classification problem.

2. **Gradient Boosting Machines (GBM)**:
   - **Pros**: Often outperforms other algorithms in classification tasks by focusing on misclassified instances.
   - **Explanation**: GBM's iterative approach could help refine the predictions further, especially if the dataset has strong feature interactions.

By implementing these models alongside the deep learning approach, we could achieve more reliable predictions for the success of funding applications.
