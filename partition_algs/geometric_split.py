import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class GeometricSplit:
    def __init__(self, file_name, df, test_size):
        self.file_name = file_name
        self.X = df.iloc[:, :-1]
        self.y = df.iloc[:, -1]
        self.test_size = test_size
        self.train_size = 1 - test_size
        self.SEEDS = [0, 42, 19, 2, 23, 11, 37, 29, 57, 5] # 10 random SEEDs to construct CI

    """
    Supportive functions:
            _largest_distance(self, center)
            _data_within_ball(center, radius)
            _ball_selection(self, center)
            _random_sums(self, total_size, num)
            _construct_hyperplane(self, seed)
            _data_within_slab(self, normal_vec, b, delta)
            _data_one_side(self, normal_vec, b)
            _compute_proportion(self, X_test)
            _modified_FPS(self, SEED)
    """
    
    def _largest_distance(self, center):
      return np.max(np.linalg.norm(self.X - center, axis=1))

    
    def _data_within_ball(center, radius):
      # np.linalg.norm -> L2-norm
      return self.X[np.linalg.norm(self.X - center, axis=1) <= radius]

    
    def _ball_selection(self, center):
        # binary search
        # CAUTION: the highest possible radius must be the distance from the center to the 
        #          farthest point in the feature space, not the dataset's diameter
        lo = 0
        epsilon = 0.01
        high = self._largest_distance(self.X, center)
        while lo < high:
            mid = lo + (high - lo) / 2
            
            inclusive_data = self._data_within_ball(self.X, center, mid)
            if len(inclusive_data) / total < self.train_size:
              lo = mid + 1
            else:
              high = mid - 1
        
        return inclusive_data


    def _union(self, a, b):
        """
        This return a list of indices that are in either a or b.
    
        Assume:
          len(a) > len(b)
    
        Optimize time complexity using Small-to-Large Merging technique
        """
        if len(a) < len(b):
          b, a = a, b
        return a.union(b)


    def _random_sums(self, total_size, num):
        cuts = np.sort(np.random.uniform(high = total_size, size = num - 1))
        parts = np.diff(np.concatenate(([0], cuts, [total_size])))
        return parts


    def _construct_hyperplane(self, seed):
        """
        This function randomly choose a point in the given set of inputs, 
        choose a normal vector ~N(0, 1), i.e. Normal distribution with mean = 0 and std = 1
        then return the corresponding hyperplane.
        
        Arguments:
        X         : all points in feature space
        
        return:
        normal_vec: the normal vector of the hyperplane
        b         : the bias of the hyperplane
        point     : the chosen point on the hyperplane
        
        """
        np.random.seed(seed)
        point = self.X.iloc[np.random.randint(0, len(self.X))]
        normal_vec = np.random.normal(size = self.X.shape[1]) # normal_vec ~ N(0, 1)
        b = np.dot(normal_vec, point)
        
        return normal_vec, b, point


    def _data_within_slab(self, normal_vec, b, delta):
        # | X.dot(normal_vector) - b | < delta
        return self.X[np.abs(self.X @ normal_vec - b) < delta]


    def _data_one_side(self, normal_vec, b):
        return self.X[self.X @ normal_vec - b < 0]


    def _compute_proportion(self, X_test):
        return len(X_test) / len(self.X)
        

    def _modified_FPS(self, SEED):
        """
        This function returns two boundaries w.r.t the given normal vector.
        
        The lowerbound (lo) represents the point that would sample the testing set whose size is < 40%,
        whereas the upperbound (hi) represents the point that would sample the testing set whose size is > 40%,
        """
        np.random.seed(SEED)
        
        normal_vec, b, point = self._construct_hyperplane(SEED)
        
        lo_found = False
        hi_found = False
        
        while not lo_found or not hi_found:
            b = np.dot(normal_vec, point)
            X_test = self._data_one_side(normal_vec, b)
            
            if compute_proportion(X_test, self.X) < test_size:
              lo = point
              lo_found = True
            else:
              hi = point
              hi_found = True
            
            dist = np.linalg.norm(self.X - point, axis = 1)
            farthes_idx = np.argmax(dist)
            point = self.X.iloc[farthes_idx] # Next point
        
        return normal_vec, lo, hi

        
    def single_hyperball(self):
        """
            Construct and save
        """

        for idx, SEED in enumerate(self.SEEDS):
            np.random.seed(SEED)
            
            center = self.X.iloc[np.random.randint(0, len(self.X))]
            inclusive_data = self._ball_selection(self.X, self.X.shape[0], 0.6, center)
            
            index_list = inclusive_data.index.to_list()
            X_train = self.X.iloc[index_list]
            y_train = self.y.iloc[index_list]
            X_test = self.X.drop(index_list)
            y_test = self.y.drop(index_list)

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)

            # Save files using the idx
            path = f"../data/splitted/{self.file_name}/Single_Hyperball/train_{idx}.csv"
            df_train.to_csv(path, index = False)
            path = f"../data/splitted/{self.file_name}/Single_Hyperball/test_{idx}.csv"
            df_test.to_csv(path, index = False)


    def multiple_hyperballs(self, num_balls):
        """
        This function randomly select balls such that it can sample and obtain the test set, whose size is less than test_size.
        
        To modify the size of multiple balls, this function will assign different test_size requirement for each ball,
        such that they all sum up to the desired test_size for the entire dataset.
        
        This encounter an issue, that the balls possibly intersect each other, this lead to fewer data points than expected.
        To address this, we omit the selected data points to guarantee the number of data obtained by each ball.
        """

        for idx, SEED in enumerate(self.SEEDS):
            # Avoid accidental modification to the original dataset
            X = self.X.copy(deep = True) 
            y = self.y.copy(deep = True)
            
            total = X.shape[0]
            
            inclusive_data = set()
    
            np.random.seed(SEED)
            sub_test_sizes = self._random_sums(self.test_size, num_balls)
            
            for sub_test_size in sub_test_sizes:
                np.random.seed(SEED)
                center = X.iloc[np.random.randint(0, len(X))]
                sub_inclusive_data = self._ball_selection(center)
                sub_inclusive_data = sub_inclusive_data.index.to_list()
                sub_inclusive_data = set(sub_inclusive_data)
                
                # Omit the selected data points to guarantee the number of data obtained by each ball
                sub_inclusive_data = sub_inclusive_data - inclusive_data
                X = X.drop(sub_inclusive_data)
                
                inclusive_data = self._union(inclusive_data, sub_inclusive_data)

            index_list = list(inclusive_data)
            X_test = self.X.iloc[index_list]
            y_test = self.y.iloc[index_list]
            X_train = self.X.drop(index_list)
            y_train = y.drop(index_list)

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)

            # Save files using the idx
            path = f"../data/splitted/{self.file_name}/Multiple_Hyperballs/train_{idx}.csv"
            df_train.to_csv(path, index = False)
            path = f"../data/splitted/{self.file_name}/Multiple_Hyperballs/test_{idx}.csv"
            df_test.to_csv(path, index = False)


    def single_slab(self):
        """
        This function construct a slab and randomly select data within the slab as training data,
        and those beyond the slab as testing data.
        
        Arguments:
        X:        the matrix of features
        train_size:     the size of training set
        
        Return:
        X_train:  the selected training samples
        """

        for idx, SEED in enumerate(self.SEEDS):
            normal_vec, b, point = self._construct_hyperplane(SEED)
            
            # binary search
            lo = 0
            high = np.max(np.linalg.norm(self.X - point, axis = 1))
            
            while lo < high:
                delta = lo + (high - lo) / 2
                inclusive_data = self._data_within_slab(normal_vec, b, delta)
                if len(inclusive_data) / self.X.shape[0] < self.train_size:
                  lo = delta
                else:
                  high = delta - 1
            
            X_train = self._data_within_slab(normal_vec, b, delta)
            y_train = self.y.iloc[X_train.index]
            
            X_test = self.X.drop(X_train.index)
            y_test = self.y.drop(X_train.index)
    
            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)
    
            # Save files using the idx
            path = f"../data/splitted/{self.file_name}/Single_Slab/train_{idx}.csv"
            df_train.to_csv(path, index = False)
            path = f"../data/splitted/{self.file_name}/Single_Slab/test_{idx}.csv"
            df_test.to_csv(path, index = False)


    def semi_infinite_slab(self):
        """
        This function construct a semi-infinite slab
        
        This function utilises the middle point between two bounds, instead of the given points between two bounds.
        
        According to the lower and upperbounds, this function leverages "binary search" to find a point which lies in the hyperplane that
        can split the dataset into two sets that satisfies the given size requirements. The termination condition is when the testing set's size
        belongs to [test_size, test_size + epsilon].
        
        Arguments:
        X:        the matrix of features
        test_size: the size of testing set
        
        Return:
        X_test   : the testing set
        """

        for idx, SEED in enumerate(self.SEEDS):
            normal_vec, lo, high = self._modified_FPS(SEED)
            epsilon = 0.01
            cur_proportion = 0
            
            while cur_proportion < self.test_size or cur_proportion > self.test_size + epsilon:
                X_lo = self._data_one_side(normal_vec, np.dot(normal_vec, lo))
                X_high = self._data_one_side(normal_vec, np.dot(normal_vec, high))
                X_mid = X_high.drop(X_lo.index, errors = "ignore")
                
                np.random.seed(SEED)
                point = X_mid.iloc[np.random.randint(len(X_mid))]
                
                b = np.dot(normal_vec, point)
                X_test = self._data_one_side(normal_vec, b)
                
                cur_proportion = self._compute_proportion(X_test)
                if cur_proportion < self.test_size:
                  lo = point
                else:
                  high = point
            
                b = np.dot(normal_vec, point)
                X_test = self._data_one_side(normal_vec, b)
                
            y_test = self.y.iloc[X_test.index]
            X_train = self.X.drop(X_test.index)
            y_train = self.y.iloc[X_train.index]

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)
    
            # Save files using the idx
            path = f"../data/splitted/{self.file_name}/Semi_Infinite_Slab/train_{idx}.csv"
            df_train.to_csv(path, index = False)
            path = f"../data/splitted/{self.file_name}/Semi_Infinite_Slab/test_{idx}.csv"
            df_test.to_csv(path, index = False)
            

    def kmeans_hyperballs(self, n_clusters):
        for idx, SEED in enumerate(self.SEEDS):
            kmeans = KMeans(n_clusters = n_clusters, random_state = SEED)
            kmeans.fit(self.X)
            
            centroids = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            total = self.X.shape[0]
            clusters = [i for i in range(n_clusters)]
            
            inclusive_data = set()
            
            while len(inclusive_data) / total < self.test_size and len(clusters):
                np.random.seed(SEED)
                cl = np.random.choice(clusters)
                labels_indices = np.where(labels == cl)[0]
                
                if (len(inclusive_data) + len(labels_indices)) / total < self.test_size:
                  inclusive_data = self._union(inclusive_data, set(labels_indices))
                else:
                  num = int(test_size * total - len(inclusive_data))
                  inclusive_data = self._union(inclusive_data, set(labels_indices[:num]))
                
                clusters.remove(cl)
            
            X_train = self.X.drop(inclusive_data)
            y_train = self.y.drop(inclusive_data)
            X_test = self.X.iloc[inclusive_data]
            y_test = self.y.iloc[inclusive_data]

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)
    
            # Save files using the idx
            path = f"../data/splitted/{self.file_name}/KMeans_Hyperballs/train_{idx}.csv"
            df_train.to_csv(path, index = False)
            path = f"../data/splitted/{self.file_name}/KMeans_Hyperballs/test_{idx}.csv"
            df_test.to_csv(path, index = False)
            
    
        