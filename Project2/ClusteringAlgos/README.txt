Steps to execute the clustering algoirhtms -

1. KMeans

2. HAC using Min approach
Open terminal and execute the following command:
> python HAC.py
Enter the input values for number of clusters (e.g. number of clusters = 2)
*Note - The data file name has been hard coded into the code in order to save execution time during demo.

3. DBSCAN
Open terminal and execute the following command:
> python DBSCAN.py
Enter the input values for eps and minPts (e.g. eps = 0.2 and minPts = 10)
*Note - The data file name has been hard coded into the code in order to save execution time during demo.

4. GMM
Open terminal and execute the following command:
> python gmm.py
    This will ask if you wanna input the parameters manually or not.
> Read Input (Y or N)? : N
    GMM will initialize it's required parameters at run time if selected "No".
> Read Input (Y or N)? : Y
    Enter mean: [[0, 0], [0, 4], [4, 4]]
    Enter cov: [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]]
    Enter pi: [0.3333, 0.3333, 0.3333]
    Enter no. of cluster: 3
    Enter max iteration: 100
    Enter convergence threshold: 0.000000001
    Enter smoothing value: 0.000000001
Note:
Please change the filename in the code in order to manually type parameters.
By default cho.txt is default filename. Please change it to run on different dataset.

5. Spectral