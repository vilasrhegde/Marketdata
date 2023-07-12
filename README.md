# Market data Analysis using K-means Clustering
> An Unsupervised machine learning algorithm to create a model with Python

## What it is all about? 
- Taking a company's dataset about all purchased made and details of customers
- Depending on these data we create Clusters to understand it in a more better way
- Here we have done 5 *clusters* and given 5 different colours namely,
    1. Red
    2. Green
    3. Blue
    4. Black
    5. Violet

- All the co-ordinates are labelled accordingly. Here I took 3 labels
    1. Customer groups
    2. Spending scores (1-100)
    3. Annual income

- Here we can get the clear picture of customer-sales data.
- Now it can be used to analyse and take a correct decision to increase profit and also user needs.

## Packages used
- Numpy
- Pandas 
- Seaborn
- Matplotlib
- Sklearn

![image](https://github.com/vilasrhegde/Marketdata/assets/85540091/3fcca89f-9362-41e7-950b-562b42b3fd74)

## The Process

### The Dataset that I have used for this project is from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)


![image](https://user-images.githubusercontent.com/85540091/171766090-9e99abe7-0ac4-4d11-a446-fe83b320736c.png)

A little peak into dataset


![image](https://user-images.githubusercontent.com/85540091/171766395-2e5a12d1-b2f0-4934-b348-318f3db48d39.png)

Checked for any missing data in the csv file, these fill feed false data into our model and we will loose accuracy 

- Slicing of multiple columns
```
x=customer_data.iloc[:,[3,4]].values 
```

- Finding WCSS value for each clusters and store it for a list
> WCSS -> Within Clusters Sum Of Squares Distance b/w each clusters and centroid 

we get,

![image](https://user-images.githubusercontent.com/85540091/171767456-0e8ec549-98e6-4060-b304-c8b15e3e6127.png)

> Observe sharp cuttings suggests significant drop

- Training the KMeans model
```kmeans = KMeans(n_clusters=5,init='k-means++',random_state=0)```

- Doing prediction from the trained model, it'll give in ununderstandable format which is list of numbers
- So we scatter all the clusters and their centroids
- Based on x,y coordinate different colours have given to distinguish the clusters easily
- Then using *matplotlb* we plot the graph like this

![image](https://user-images.githubusercontent.com/85540091/171768040-45164669-bfaa-4501-a225-061e9a3a13ae.png)


## Conclusion

- By visualising the data we can understand these like,
    - Blue = less income and less purchase
    - Purple = less income and more purchase
    - Green = more income and less purchase
    - Black = more income more purchase
- Market can attract **Blue** group people providing some discounts
- Market can attract **Green** region people who have money but not buying more things


# Applications
- Netflix suggesting group of people who are watching some genre more
- Google ads personalisation
