# # Zillow Regression Project
Mack McGlenn, O'neil Cohort

### Project Overview
_____________________________________________________________________________________
This project will identify key drivers in determining tax value from the 2017 Zillow property dataset. Those drivers will then be taken to create a regression model which will best predict a home's future tax value.

**Objectives**

- Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter notebook final report.

- Create modules (wrangle.py) that make your process repeateable and your report (notebook) easier to read and follow.

- Ask exploratory questions of your data that will help you understand more about the attributes and drivers of home value. Answer questions through charts and statistical tests.

- Construct a model to predict assessed home value for single family properties using regression techniques.

- Make recommendations to a data science team about how to improve predictions.

**Business Goals**

- Construct an ML Regression model that predicts propery tax assessed values ('taxvaluedollarcnt') of Single Family Properties using attributes of the properties.

- Find the key drivers of property value for single family properties.

**Deliverables**

- a complete readme.md
- a final report (.ipynb)
- wrangle.py
- exploratory workbook


### Project Planning

_____________________________________________________________________________________

#### Outline:
1. Acquire & Prepare Data
2. Exploration/ Correlation Analysis
3. Statistical Analysis
4. Preprocessing
5. Modeling
6. Conclusion

_____________________________________________________________________________________


## 1. Acquire & Prepare

#### Acquire
- Data acquired from CodeUp Database using get_zillow() function 
-  1500000 rows × 9 columns
- metrics called: bedroomcnt, bathroomcnt,
calculatedfinishedsquarefeet, fips, lotsizesquarefeet,
 taxvaluedollarcnt, yearbuilt, assessmentyear, taxamount
#### Prepare Actions:
- Drop null values from columns
- Rename columns for ease of delivery
- Feature engineered new metric called 'yard_sqft' 
- Reassign values in 'fips' column to new variable called 'county', which returns the names of the county where the homes are being evaluated
- Created dummy variables for 'county' column
- Remove outliers from dataset
- Split data into train/val/test datasets

#### Functions called from wrangle.py:
1. get_zillow()
2. prep_zillow(df)
3. remove_outliers(df, columns)
4. split_zillow(df)


## 2. Exploration/ Correlation Analysis

Here are the steps I took during the exploration phase:

- Measure the distribution of my target variable (tax_val) in my train dataset
- Use spearmans's correlation to determine how my features relate to and impact my target variable
- Visualize that correlation via a heatmap and barplot

#### Functions called from wrangle.py:
1. show_corr(train)



## 3. Statistical Analysis

**Questions Addressed:**

1. Do homes with more bathrooms have higher tax values?
2. How does yard size effect tax value?
3. Does county location effect tax value?

**Methodology:**
1. Do homes with more bathrooms have higher tax values?
    - Null Hypothesis: Homes with more bathrooms will not have higher tax values than homes with less bathrooms
    - Hypothesis: Homes with more bathrooms will have higher tax values than homes with less values
    Test: T-test
    Results:
    - p < a
    - Reject the null hypothesis
    - Returns the average price of homes with 2 or less/ 2 or more bathrooms
    
2. How does yard size effect tax value?
    - H^O: Larger yard size decreases tax value.
    - Hypothesis: Larger yard size increases tax value.
    Test: ANOVA
    Results:
    - p < a, F-statistic: 2485.98
    - Reject the null hypothesis
    
3. Does county location effect tax value?
    - H^O: Tax value means will be equal across all three counties.
    - Hypothesis: Tax values will be very different across all three counties.
   Test: f_oneway
   Results:
    - p< a
    - Reject the null hypothesis
    - Returns the mean and median tax value of homes in all three counties


## 4. Modeling

The goal of my modeling is to reduce RMSE for train and validate data. I'll calculate my RMSE for mean and median data, and I'll finish by comparing my models to that data. I'm only dropping my non-numeric features for modeling.

#### Features kept:
- beds
- baths
- tax_val
- sq_ft
- Los Angeles
- Orange County
- Ventura
- yard_sqft

#### Baseline

![image](https://user-images.githubusercontent.com/122935207/233749846-bd74cdfe-d6ae-40bf-9c43-2646e877a8b2.png)

**Models Used** 

- Polynomial Features
- OLS 
- LASSO LARS

**Outcome**
![image](https://user-images.githubusercontent.com/122935207/233749937-22e63c2c-c1cc-4682-ad9f-63224aad2a94.png)

My best model is polynomial features with a degree of 5. It outperforms my baseline mean validate RMSE by $17,530, or 9%, and has r2 difference in the train and validate sets that are nearly identical

### CONCLUSION

- During exploration, I found that bathroom count was a really great indicator of predicting tax value within a home. Feature engineering the different yard sizes to predicts home tax value also provided valuable insights. However, the best indicator of tax value was property county location. When exploring this data in the future, I would start by separating the data by county and seeing how the features I chose shift in correlation across the counties individually.

#### Recommendations

As someone who has worked with residential property data before, I understand that there is a differnce in tax value and assessed property (or market) value. Tax value is calculated by the county where the property is located and is calculated by an assessor from that county. The assessor looks at comps (comperable properties) that have recently been sold, the value of recent structural improvements, and the cost of replacing the home (or the property it's on) if it were destroyed. Tax value usually represents 80% of market value. Market value also relies on comps, but also on appraisals, which are usually based on square footage. In the future, I would base this project around predicting market value.
