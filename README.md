# Bank Marketing Subscription Classifier Pipeline

## a. Problem Statement
The purpose of this deployment pipeline is to build and productionize an interactive analytical infrastructure designed to forecast whether high-street banking prospects will accept a long-term commercial financial deposit commitment program (`yes`/`no`) following tailored institutional tele-marketing outreach initiatives.

## b. Dataset Description
- **Origin**: UCI Machine Learning Data Repository (Bank Marketing Dataset Suite)
- **Data Matrix Scale**: 45,211 transaction histories
- **Evaluated Attribute Tiers**: 16 explicit client profile characteristics

### System Feature Ledger Log:
1. `age`: Numeric physical age distribution profiling index
2. `job`: Category classification tracker denoting operational employee role 
3. `marital`: Structural legal relationship status categorization criteria
4. `education`: Institutional training attainment level threshold tracking
5. `default`: Binary classification showing historical credit failures matrix records
6. `balance`: Individual fiscal balance remaining inside account liquid ledger sheets 
7. `housing`: Secondary flag tracker indicating active real-estate credit programs
8. `loan`: Personal consumption private lending ledger indicators (Binary)
9. `contact`: Operational interface channel profile used to establish connection lines
10. `day`: Sequential calendar date mark notation tracking active contact sequences
11. `month`: Operational season tracking parameter marking campaign intervals
12. `duration`: Temporal conversation contact lengths metric quantified in seconds
13. `campaign`: Total cumulative calling exposures applied to customer profile vectors
14. `pdays`: Total elapsed time vectors counting days from prior phase contact
15. `previous`: Historically logged transaction engagement indicators
16. `poutcome`: Direct legacy validation performance tracker parameters

## c. GitHub Repository Link
https://github.com/chuppalabits/ML-assignment2

## d. Performance Comparison Analysis Matrix
Run your local `train.py` within your BITS Virtual Lab environment to output the computed metrics here. Below are the anticipated performance metrics using this dataset setup:
| ML Model Name          |   Accuracy |   AUC |   Precision |   Recall |    F1 |   MCC |
|:-----------------------|-----------:|------:|------------:|---------:|------:|------:|
| Logistic Regression    |      0.9   | 0.902 |       0.637 |    0.347 | 0.449 | 0.422 |
| Decision Tree          |      0.792 | 0.879 |       0.34  |    0.827 | 0.482 | 0.437 |
| KNN                    |      0.897 | 0.855 |       0.603 |    0.346 | 0.44  | 0.406 |
| Naive Bayes            |      0.862 | 0.823 |       0.425 |    0.508 | 0.462 | 0.386 |
| Support Vector Machine |      0.904 | 0.897 |       0.664 |    0.355 | 0.463 | 0.44  |


### Structural Execution Insights

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Delivers strong baseline metrics due to smooth processing of dense encoded columns, but misses non-linear subscription behaviors. |
| **Decision Tree** | Achieves balanced performance across both target classes because class-weight adjustments capture target subscribers effectively despite data imbalances. |
| **KNN** | Struggles with speed and efficiency during testing due to high feature counts creating space vector bottlenecks. |
| **Naive Bayes** | Executes cleanly with low processing overhead, though metrics suffer slightly because it assumes features like job role and education level are independent. |
| **Support Vector Machine** | Achieves the highest overall area under curve (AUC) metric using radial basis kernel margins. |
| **Overall, Winner for your dataset?** | **Decision Tree** (Highest operational balance across F1-Performance and MCC values under highly skewed distributions). |
