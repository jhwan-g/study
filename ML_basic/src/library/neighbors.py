import numpy as np

class KNeighbors:
    AVAILABLE_DISTANCE_TYPES = ["mincowski"]

    def __init__(self):
        self.n_neighbors = None
        self.distance_type = None
        self.data = None
        self.target = None

    @staticmethod
    def _mincowski_dist(input1, input2):
        dist = 0
        for x1, y1 in zip(input1, input2):
            dist += (x1 - y1)**2
        return np.sqrt(dist)
    
    def _dist(self, input1, input2):
        if self.distance_type == self.AVAILABLE_DISTANCE_TYPES[0]:
            return self._mincowski_dist(input1, input2)

    def _distances_with_metadata(self, input):
        return [(self._dist(input, self.data[i]), self.target[i], self.data[i])
                 for i in range(len(self.data))]
    
    def __str__(self):
        return f'<KNeighborsInstance n_neighbors:{self.n_neighbors} matrix:{self.distance_type}>'

class KNeighborsRegressor(KNeighbors):
    def __init__(self, n_neighbors = 5, distane_type = "mincowski"):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.distance_type = distane_type

        assert(distane_type in self.AVAILABLE_DISTANCE_TYPES)
        
        self.data = None
        self.target = None

    def _predict_one_value(self, input):
        assert(len(self.data) == len(self.target))

        distances_with_metadata = self._distances_with_metadata(input)

        distances_with_metadata = sorted(distances_with_metadata, key = (lambda x: x[0]))
        distances_with_metadata = distances_with_metadata[:self.n_neighbors]

        yhat = np.zeros(input.shape)
        for distance_with_metadata in distances_with_metadata:
            yhat += distance_with_metadata[1]
        
        yhat = yhat / self.n_neighbors
        
        return yhat

    def kneighbors(self, input):
        distances_with_metadata = self._distances_with_metadata(input)

        distances_with_metadata = sorted(distances_with_metadata, key = (lambda x: x[0]))
        distances_with_metadata = distances_with_metadata[:self.n_neighbors]

        return (np.array([i[2] for i in distances_with_metadata]), np.array([i[1] for i in distances_with_metadata]))

    def predict(self, input):
        ''' 
        KNeighborsRegressor predict  
        input: input(array like object)  
        output: predict result  

        '''
        input = np.array(input)

        if len(input.shape) == len(self.data.shape) - 1:
            input = input.reshape((1,-1))
        
        return np.array(
            [self._predict_one_value(i) for i in input]
        )

    def fit(self, input, target):
        '''
        KNeighborsRegressor fit  
        input: input(array like object), target(array like obejct)  
        output: new KNeighborsRegressor  
        '''
        new_classifier = KNeighborsRegressor(self.n_neighbors, self.distance_type)
        new_classifier.data = np.array(input)
        new_classifier.target = np.array(target)
        return new_classifier

    def score(self, input, target):
        '''
        KNeightborsRegressor score  
        input: input(array like object), target(array like obejct)  
        output: float(0~1 range)  
        '''
        input = np.array(input)
        target = np.array(target)

        SSR = 0
        SST = target.std(axis = 1) ** 2

        for i in range(len(input)):
            SSR += (self.predict(i) - target[i]) ** 2
        
        return 1 - SSR / SST


class KNeighborsClassifier(KNeighbors):
    def __init__(self, n_neighbors = 5, distane_type = "mincowski"):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.distance_type = distane_type

        assert(distane_type in self.AVAILABLE_DISTANCE_TYPES)
        
        self.data = None
        self.target = None
    
    def _predict_one_value(self, input):
        assert(len(self.data) == len(self.target))

        distances_with_metadata = self._distances_with_metadata(input)

        distances_with_metadata = sorted(distances_with_metadata, key = (lambda x: x[0]))
        distances_with_metadata = distances_with_metadata[:self.n_neighbors]

        class_count = {}
        for distance_with_metadata in distances_with_metadata:
            curr_class = distance_with_metadata[1]
            if curr_class not in class_count.keys():
                class_count[curr_class] = 1
            else:
                class_count[curr_class] += 1
        
        max_index = list(class_count.keys())[0]
        for index in class_count.keys():
            if class_count[index] > class_count[max_index]:
                max_index = index
        
        return max_index

    def kneighbors(self, input):
        distances_with_metadata = self._distances_with_metadata(input)

        distances_with_metadata = sorted(distances_with_metadata, key = (lambda x: x[0]))
        distances_with_metadata = distances_with_metadata[:self.n_neighbors]

        return (np.array([i[2] for i in distances_with_metadata]), np.array([i[1] for i in distances_with_metadata]))

    def predict(self, input):
        ''' 
        KNeighborsClassifier predict  
        input: input(array like object)  
        output: predict result  
        '''
        input = np.array(input)

        if len(input.shape) == len(self.data.shape) - 1:
            input = input.reshape((1,-1))
        
        return np.array(
            [self._predict_one_value(i) for i in input]
        )

    def fit(self, input, target):
        '''
        KNeighborsClassifier fit  
        input: input(array like object), target(array like obejct)  
        output: new KNeighborsClassifier  
        '''
        new_classifier = KNeighborsClassifier(self.n_neighbors, self.distance_type)
        new_classifier.data = np.array(input)
        new_classifier.target = np.array(target)
        return new_classifier

    def score(self, input, target):
        '''
        KNeightborsClassifier score  
        input: input(array like object), target(array like obejct)  
        output: float(0~1 range)  
        '''
        input = np.array(input)
        target = np.array(target)

        sample_num = len(input)
        right_sample_num = 0

        for i in range(len(input)):
            if self.predict(input[i]) == np.array(target[i]):
                right_sample_num += 1
        
        return right_sample_num / sample_num
