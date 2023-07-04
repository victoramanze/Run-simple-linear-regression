# Run-simple-linear-regression
Introduction
As you’re learning, simple linear regression is a way to model the relationship between two variables. By assessing the direction and magnitude of a relationship, data professionals are able to uncover patterns and transform large amounts of data into valuable knowledge. This enables them to make better predictions and decisions.

In this lab, you are part of an analytics team that provides insights about your company’s sales and marketing practices. You have been assigned to a project that focuses on the use of influencer marketing. For this task, you will explore the relationship between your radio promotion budget and your sales.

The dataset provided includes information about marketing campaigns across TV, radio, and social media, as well as how much revenue in sales was generated from these campaigns. Based on this information, company leaders will make decisions about where to focus future marketing resources. Therefore, it is critical to provide them with a clear understanding of the relationship between types of marketing campaigns and the revenue generated as a result of this investment.

Imports In this section, first import relevant Python libraries and modules.

Import relevant Python libraries and modules

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
Now load the dataset into a DataFrame. The dataset provided is a csv file (named marketing_sales_data.csv) containing information about promotional marketing conducted in collaboration with influencers, along with the corresponding sales. This is a fictional dataset that was created for educational purposes and modified for this lab. Assume that the numerical variables in the data are expressed in millions of dollars.

Load the dataset into a DataFrame and save in a variable

pd.read_csv("marketing sales data.csv")

data = pd.read_csv("marketing sales data.csv")
Data Exploration
To get a sense of what the data includes, display the first 10 rows of the data.

Display the first 10 rows of the data

data.head(10)

The data includes the following information: TV promotion budget (expressed as “Low”, “Medium”, or “High”) Radio promotion budget Social media promotion budget Type of influencer that the promotion is in collaboration with (expressed as “Mega”, “Macro”, or “Micro”, or “Nano”) Note: Mega-influencers have over 1 million followers, macro-influencers have 100,000 to 1 million followers, micro-influencers have 10,000 to 100,000 followers, and nano-influencers have fewer than 10,000 followers. Sales accrued from the promotion

To get a sense of how large the data is, identify the number of rows and the number of columns in the data.

Display number of rows, number of columns

data.shape

There are 572 rows and 5 columns in the data. One way to interpret this is that 572 companies are represented in the data, along with 5 aspects about each company that reveals how they promote their products/services and the sales accrued from their promotion.

Now, check for missing values in the rows of the data. This is important because missing values are not that meaningful when modeling the relationship between two variables. To do so, begin by getting Booleans that indicate whether each value in the data is missing. Then, check both columns and rows for missing values

Step 1.
Start with .isna() to get booleans indicating whether each value in the data is missing

data.isna()

Step 2.
Use .any(axis=1) to get booleans indicating whether there are any missing values along the columns in each row

data.isna().any(axis=1)

Step 3.
Use .sum() to get the number of rows that contain missing values
data.isna().any(axis=1).sum()

There are 3 rows containing missing values, which is not that many, considering the total number of rows. It would be appropriate to drop these rows that contain missing values to proceed with preparing the data for modeling.

Drop the rows that contain missing values. This is an important step in data cleaning, as it makes the data more usable for the analysis and regression that you will conduct next.

Step 1. Use .dropna(axis=0) to indicate that you want rows which contain missing values to be dropped
data = data.dropna(axis=0)
Check to make sure that the data does not contain any rows with missing values now
Start 1
with .isna() to get booleans indicating whether each value in the data is missing

Step 2.
Use .any(axis=1) to get booleans indicating whether there are any missing values along the columns in each row

Step 3.
Use .sum() to get the number of rows that contain missing values Use .dropna(axis=0) to indicate that you want rows which contain missing values to be dropped


data.isna().any(axis=1).sum()

Check model assumptions.
You would like to explore the relationship between radio promotion budget and sales. You could model the relationship using linear regression. To do this, you want to check if the model assumptions for linear regression can be made in this context. Some of the assumptions can be addressed before the model is built — — you will address those in this section. After the model is built, you can finish checking the assumptions.

Start by creating a plot of pairwise relationships in the data. This will help you visualize the relationships between variables in the data and help you check model assumptions.

sns.pairplot(data)

Is the assumption of linearity met?
In the scatter plot of Sales over Radio, the points appear to cluster around a line that indicates a positive association between the two variables. Since the points cluster around a line, it seems the assumption of linearity is met.

Model Building Start by selecting only the columns that are needed for the model that you will build from the data.

Select relevant columns
Save resulting DataFrame in a separate variable to prepare for regression

ols_data = data[["Radio", "Sales"]]
Display the first 10 rows of the new DataFrame to ensure it is accurate.

Display first 10 rows of the new DataFrame
ols_data.head(10)

Write the linear regression formula for modeling the relationship between the two variables of interest.

Write the linear regression formula
Save it in a variable

ols_formula = "Sales ~ Radio"
Implement the Ordinary Least Squares (OLS) approach for linear regression.

Implement OLS

OLS = ols(formula = ols_formula, data = ols_data)
Create a linear regression model for the data and fit the model to the data.

Fit the model to the data
Save the fitted model in a variable

model = OLS.fit()
Results and Evaluation Get a summary of the results from the model.

Get summary of results

model.summary()

Analyze the bottom table from the results summary. Based on that table, identify the coefficients that the model determined would generate the line of best fit, the coefficients here being the y-intercept and the slope.

Question
What is the y-intercept? The y-intercept is 41.5326.

Question
What is the slope?

The slope is 8.1733.

Question
What is the linear equation you would write to express the relationship between sales and radio promotion budget in the form of y = slope * x + y-intercept? sales = 8.1733 * radio promotion budget + 41.5326

Question
What do you think the slope in this context means?

One interpretation: If a company has a budget of 1 million dollars more for promoting their products/services on the radio, the company’s sales would increase by 8.1733 million dollars on average. Another interpretation: Companies with 1 million dollars more in their radio promotion budget accrue 8.1733 million dollars more in sales on average.

Finish checking model assumptions. Now that you’ve built the linear regression model and fit it to the data, you can finish checking the model assumptions. This will help confirm your findings.

Plot the OLS data with the best fit regression line.

Plot the OLS data with the best fit regression line

sns.regplot(x = "Radio", y = "Sales", data = ols_data)

What do you observe from the regression plot above? The regression plot above shows an approximately linear relationship between the two variables along with the best fit line. This confirms the assumption of linearity.

Check the normality assumption. To get started, get the residuals from the model.

Get the residuals from the model

residuals = model.resid
Visualize the distribution of the residuals.

Visualize the distribution of the residuals

fig = sns.histplot(residuals)
fig.set_xlabel("Residual Value")
fig.set_title("Histogram of Residuals")
plt.show()

Based on the visualization above, what do you observe about the distribution of the residuals? Based on the visualization above, the distribution of the residuals is approximately normal. This indicates that the assumption of normality is likely met.

Create a Q-Q plot to confirm the assumption of normality.

Create a Q-Q plot

sm.qqplot(residuals, line='s')
plt.title("Q-Q plot of Residuals")
plt.show()

Is the assumption of normality met?
In the Q-Q plot above, the points closely follow a straight diagonal line trending upward. This confirms that the normality assumption is met.

Check the assumptions of independent observation and homoscedasticity. Start by getting the fitted values from the model.

Get fitted values

fitted_values = model.predict(ols_data["Radio"])
Create a scatterplot of the residuals against the fitted values.

Create a scatterplot of residuals against fitted values

fig = sns.scatterplot(x=fitted_values, y=residuals)
fig.axhline(0)
fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")
plt.show()

Are the assumptions of independent observation and homoscedasticity met? In the scatterplot above, the data points have a cloud-like resemblance and do not follow an explicit pattern. So it appears that the independent observation assumption has not been violated. Given that the residuals appear to be randomly spaced, the homoscedasticity assumption seems to be met.

Conclusion
What are the key takeaways from this lab?

Data visualizations and exploratory data analysis can be used to check if linear regression is a well suited approach for modeling the relationship between two variables. The results of a linear regression model can be used to express the relationship between two variables. What results can be presented from this lab?

In the simple linear regression model, the y-intercept is 41.5326 and the slope is 8.1733. One interpretation: If a company has a budget of 1 million dollars more for promoting their products/services on the radio, the company’s sales would increase by 8.1733 million dollars on average. Another interpretation: Companies with 1 million dollars more in their radio promotion budget accrue 8.1733 million dollars more in sales on average.

The results are statistically significant with a p-value of 0.000, which is a very small value (and smaller than the common significance level of 0.05). This indicates that there is a very low probability of observing data as extreme or more extreme than this dataset when the null hypothesis is true. In this context, the null hypothesis is that there is no relationship between radio promotion budget and sales i.e. the slope is zero, and the alternative hypothesis is that there is a relationship between radio promotion budget and sales i.e. the slope is not zero. So, you could reject the null hypothesis and state that there is a relationship between radio promotion budget and sales for companies in this data.

The slope of the line of best fit that resulted from the regression model is approximate and subject to uncertainty (not the exact value). The 95% confidence interval for the slope is from 7.791 to 8.555. This indicates that there is a 95% probability that the interval [7.791, 8.555] contains the true value for the slope.

How would you frame your findings to external stakeholders?
Based on the dataset at hand and the regression analysis conducted here, there is a notable relationship between radio promotion budget and sales for companies in this data, with a p-value of 0.000 and standard error of 0.194. For companies represented by this data, a 1 million dollar increase in radio promotion budget could be associated with a 8.1733 million dollar increase in sales. It would be worth continuing to promote products/services on the radio. Also, it is recommended to consider further examining the relationship between the two variables (radio promotion budget and sales) in different contexts. For example, it would help to gather more data to understand whether this relationship is different in certain industries or when promoting certain types of products/services.

DATA SCIENCE : REGRESSION ANALYSIS : MODEL ASSUMPTIONS.
