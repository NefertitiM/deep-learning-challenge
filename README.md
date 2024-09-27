# Deep Learning Challenge

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

#### AlphabetSoupCharity Original Model

- **Neurons**:
  - **First Hidden Layer**: 80 neurons
  - **Second Hidden Layer**: 30 neurons

- **Layers**:
  - Input layer
  - Two hidden layers
  - Output layer

- **Activation Functions**:
  - **Hidden Layers**: ReLU (Rectified Linear Unit) to introduce non-linearity and help with learning complex patterns.
  - **Output Layer**: Sigmoid activation function to predict probabilities for binary classification (success or failure).

##### Model Performance

- **Achieved Accuracy**: 
  - The original model achieved an accuracy of approximately 72.55%, which is below the target performance of over 75%.

- **Model Loss**: 
  - The loss was approximately 0.5623, indicating room for improvement in the model's predictive capabilities.



#### AlphabetSoupCharity Optimization 1

- **Neurons**:
  - **First Hidden Layer**: 100 neurons
  - **Second Hidden Layer**: 100 neurons

- **Layers**:
  - Input layer
  - Two hidden layers with batch normalization and dropout
  - Output layer

- **Activation Functions**:
  - **Hidden Layers**: ReLU (Rectified Linear Unit) for both layers to capture complex patterns.
  - **Output Layer**: Sigmoid activation function for binary classification.

##### Model Performance

- **Achieved Accuracy**: 
  - The optimized model achieved an accuracy of approximately 72.93%, still below the target performance of over 75%.

- **Model Loss**: 
  - The loss was approximately 0.5554, indicating slight improvement compared to the original model.

##### Steps Taken to Increase Model Performance

1. **Increased Neurons**:
   - Both hidden layers were increased to 100 neurons to allow the model to learn more complex representations.

2. **Batch Normalization**:
   - Added batch normalization layers after each hidden layer to stabilize and accelerate training.

3. **Dropout Regularization**:
   - Implemented dropout layers (20%) to reduce the risk of overfitting by randomly disabling neurons during training.

4. **Learning Rate Adjustment**:
   - Used a learning rate of 0.009 with the Adam optimizer to improve convergence.

5. **Increased Epochs**:
   - Increased the number of epochs to 200 to provide more training time for the model.

6. **Validation Split**:
   - Continued using a validation split of 20% to monitor performance and mitigate overfitting.

---

#### AlphabetSoupCharity Optimization 2

- **Neurons**:
  - **First Hidden Layer**: 80 neurons
  - **Second Hidden Layer**: 80 neurons
  - **Third Hidden Layer**: 80 neurons
  - **Fourth Hidden Layer**: 80 neurons

- **Layers**:
  - Input layer
  - Four hidden layers with batch normalization and dropout
  - Output layer

- **Activation Functions**:
  - **Hidden Layers**: Sigmoid activation function to capture non-linear relationships.
  - **Output Layer**: Sigmoid activation function for binary classification.

##### Model Performance

- **Achieved Accuracy**: 
  - The optimized model achieved an accuracy of approximately 72.48%, still below the target performance of over 75%.

- **Model Loss**: 
  - The loss was approximately 0.5610, indicating minimal improvement compared to the previous optimization.

##### Steps Taken to Increase Model Performance

1. **Increased Layers**:
   - Added a total of four hidden layers with 80 neurons each to enhance the model's capacity to learn complex patterns.

2. **Activation Function Change**:
   - Changed the activation function from ReLU to Sigmoid for hidden layers, which can help when dealing with smaller datasets.

3. **Batch Normalization**:
   - Continued using batch normalization after each hidden layer to stabilize training.

4. **Dropout Regularization**:
   - Retained dropout layers (20%) to mitigate overfitting.

5. **Learning Rate Adjustment**:
   - Adjusted the learning rate to 0.001 to improve convergence speed.

6. **Increased Batch Size**:
   - Increased the batch size to 200 for potentially more stable gradient estimates.

7. **Reduced Epochs**:
   - Limited training to 15 epochs to quickly assess model performance without overfitting.

---

#### AlphabetSoupCharity Optimization 3

- **Neurons**:
  - **First Hidden Layer**: 80 neurons
  - **Second Hidden Layer**: 100 neurons
  - **Third Hidden Layer**: 80 neurons
  - **Fourth Hidden Layer**: 100 neurons

- **Layers**:
  - Input layer
  - Four hidden layers with batch normalization and dropout
  - Output layer

- **Activation Functions**:
  - **Hidden Layers**: Sigmoid activation function to capture non-linear relationships.
  - **Output Layer**: Sigmoid activation function for binary classification.

##### Model Performance

- **Achieved Accuracy**: 
  - The optimized model achieved an accuracy of approximately 72.54%, still below the target performance of over 75%.

- **Model Loss**: 
  - The loss was approximately 0.5585, indicating minimal improvement compared to previous optimizations.

##### Steps Taken to Increase Model Performance

1. **Adjusted Neurons**:
   - Used a mix of 80 and 100 neurons across hidden layers to better capture the data's complexity.

2. **Activation Function**:
   - Continued using the sigmoid activation function for all hidden layers to maintain consistency.

3. **Batch Normalization**:
   - Implemented batch normalization after each hidden layer to stabilize learning and improve convergence speed.

4. **Dropout Regularization**:
   - Maintained dropout layers (20%) to reduce overfitting.

5. **Learning Rate**:
   - Kept the learning rate at 0.001 for steady convergence.

6. **Batch Size**:
   - Increased the batch size to 50 for potentially more stable gradient estimates.

7. **Limited Epochs**:
   - Limited training to 15 epochs to quickly evaluate performance without overfitting.

---






