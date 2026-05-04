# Customer Segmentation using K-Means Clustering

## Project Overview
This project performs customer segmentation using the K-Means clustering algorithm.  
Customers are grouped based on their age, annual income, and spending score to understand purchasing behavior.


## Objective
- Identify different customer segments  
- Analyze customer behavior  
- Help businesses improve marketing strategies  
- Target customers effectively  


## Technologies Used
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Visual Studio Code  


## Dataset
- Mall Customers Dataset  
- File: `Mall_Customers.csv`

## Features Used:
- Age  
- Annual Income (k$)  
- Spending Score (1-100)  


## Methodology
1. Data Collection  
2. Data Preprocessing (Handling missing values & scaling)  
3. Feature Selection  
4. Finding optimal clusters using Elbow Method  
5. Applying K-Means clustering  
6. Visualization of clusters  
7. Cluster analysis  


## Output
- Customer clusters  
- Elbow method graph  
- Cluster visualization graph  
- Cluster summary CSV file  


## Project Structure
```
customer-segmentation-ml/
│
├── data/
├── graphs/
├── outputs/
├── src/
├── README.md
```

## How to Run

1. Install dependencies:
   ```pip install -r requirements.txt```

2. Run the project:
   ```python src/main.py```


## Results
- Customers are grouped into 4 clusters  
- Each cluster represents a unique customer segment  
- Helps identify high-value and low-value customers  


## Conclusion
K-Means clustering helps businesses understand customer behavior and enables better decision-making through targeted marketing strategies.
