# NUS EE5907 Pattern Recognition
## Assignment 1
### for assignment deadline: 11.59pm, Monday, Sep 30th, 2019

## *How to start*
Make sure utility file `DataUtil.py` and data file `spamData.mat` exist.

Make sure you have all the required packages installed, or simply use Anaconda.

Run q1 to q4 by executing the corresponding Python scripts.
For example:
```
python q1.py
```

## *Use Jupyter Notebook*
To convert Python file into Jupyter Notebook, use `ipynb-py-convert`

* Installation: 
```
pip install ipynb-py-convert
```

or

```
conda install -c defaults -c conda-forge ipynb-py-convert
```

* Convert from Python to Jupyter Notebook
```
ipynb-py-convert q1.py q1.ipynb
```

* Convert from Jupyter Notebook to Python
```
ipynb-py-convert q1.ipynb q1.py 
```



# Q1. Beta-binomial Naïve Bayes

## Question Interpretation:
This question asks to use generative module, naïve bayes classifier. The equation is:

![p(\tilde{y} = c | \tilde{x}, D) \propto  log p(\tilde{y} = c | \lambda ^{ML}) + \sum_{j=1}^{D}log p(\tilde{x}_{j} |x_{i\in cj}, \lambda ^{ML})](https://latex.codecogs.com/svg.latex?p(\tilde{y}%20=%20c%20|%20\tilde{x},%20D)%20\propto%20%20log%20p(\tilde{y}%20=%20c%20|%20\lambda%20^{ML})%20+%20\sum_{j=1}^{D}log%20p(\tilde{x}_{j}%20|x_{i\in%20cj},%20\lambda%20^{ML}))

## Approach:

Since the email can only be classified as spam or non-spam, y can only be 0 or 1. In order to get ![p(\tilde{y} = 0 | \tilde{x}, D)](https://latex.codecogs.com/svg.latex?p(\tilde{y}%20=%200%20|%20\tilde{x},%20D)) and ![p(\tilde{y} = 1 | \tilde{x}, D)](https://latex.codecogs.com/svg.latex?p(\tilde{y}%20=%201%20|%20\tilde{x},%20D)), calculate first term and the sum of the following terms respectively.

## Results:

# Q2. Gaussian Naïve Bayes

## Question Interpretation:

## Approach:

## Results:
 
# Q3. Logistic Regression

## Question Interpretation:

## Approach:

## Results:
 
# Q4. K-Nearest Neighbors

## Question Interpretation:

## Approach:

## Results:

