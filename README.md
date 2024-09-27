# Deep Learning Challenge

Machine learning and neural networks were used to build a tool for the nonprofit foundation, Alphabet Soup, that can help predict the success of organizations applying for their funding. A CSV with more than 34,000 organizations funded by Alphabet Soup over several years was provided. Key features in the dataset were used to create a binary classifier that correlates with successful funded organizations and can predict successful applicants.

---

## Data Preprocessing

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

  All of these variables were chosen as features because they impact the success of the organizations.
    
- **Variables to Remove**: 
  - `EIN`: Identification number of the 
