import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
class NSO():
    """
    Nyström Subspace Override Transfer Service Class
    
    Functions
    ----------
    nys_subspace_override: Transfer Basis from Target to Source Domain.
    data_augmentation: Augmentation of data by removing or upsampling of source data

    Example:
    --------
    >>> import os
    >>> import scipy.io as sio
    >>> from sklearn.svm import SVC
    >>> 
    >>> os.chdir("../../../Database/domain_adaptation/OfficeCaltech/features/surf")
    >>> 
    >>> # Load and preprocessing of data. Note normalization to N(0,1) is necessary.
    >>> dslr = sio.loadmat("dslr_SURF_L10.mat")
    >>> Xs = preprocessing.scale(np.asarray(dslr["fts"]))
    >>> Ys = np.asarray(dslr["labels"])
    >>> 
    >>> amazon = sio.loadmat("amazon_SURF_L10.mat")
    >>> Xt = preprocessing.scale(np.asarray(amazon["fts"]))
    >>> Yt = np.asarray(amazon["labels"])
    >>> 
    >>> # Applying SVM without transfer learning. Accuracy should be about 10%
    >>> clf = SVC(gamma=1,C=10)
    >>> clf.fit(Xs,Ys)
    >>> print("SVM without transfer "+str(clf.score(Xt,Yt.ravel())))
    >>> 
    >>> # Initialization of NSO. Accuracy of SVM + NSO should be about 90%
    >>> nso = NSO(landmarks=100)
    >>> # Compute domain invariant subspace data directly by 
    >>> Xt,Xs,Ys = nso.fit_transform(Xt,Xs,Ys)
    >>> 
    >>> # Or use two steps
    >>> # nso.fit(Xt)
    >>> # Xt,Xs,Ys = nso.transform(Xs,Ys)
    >>> 
    >>> clf = SVC(gamma=1,C=10)
    >>> clf.fit(Xs,Ys)
    >>> print("SVM + NSO: "+str(clf.score(Xt,Yt.ravel())))
    >>> 
    >>> model = KNeighborsClassifier(n_neighbors=1)
    >>> model.fit(Xs, Ys.ravel())
    >>> 
    >>> score = model.score(Xt, Yt)
    >>> print("KNN + NSO: "+str(score))
    >>> """

    def __init__(self,landmarks=10):
        self.n_landmarks = landmarks
        pass

    def fit_transform(self,Xt,Xs,Ys=None):
        """
        Nyström Subspace Override
        Transfers Basis of X to Xs obtained by Nyström SVD
        Implicit dimensionality reduction
        Applications in domain adaptation or transfer learning
        Parameters.
        Note target,source are order sensitiv.

        ----------
        Parameters 
        X   : Target Matrix, where classifier is trained on
        Xs  : Source Matrix, where classifier is trained on
        Ys  : Source data label, if none, classwise sampling is not applied.
        landmarks : Positive integer as number of landmarks
        
        ----------
        Returns
        Xt : Reduced Target Matrix
        Xs : Reduced approximated Source Matrix
        Ys : Augmented source label matrix
        """
        Ys,Xs = self.data_augmentation(Xs,Xt.shape[0],Ys)
        self.n_xt = Xt.shape[0]
        if type(Xt) is not np.ndarray or type(Xs) is not np.ndarray:
            raise ValueError("Numpy Arrays must be given!")
        if type(self.n_landmarks ) is not int or self.n_landmarks  < 1:
            raise ValueError("Positive integer number must given!")

        # Correct landmarks if user enters impossible value
        self.n_landmarks = int(np.min(list(Xt.shape)+list(Xs.shape)+[self.n_landmarks ]))
        max_idx = np.min(list(Xt.shape)+list(Xs.shape))
        idx = np.random.randint(0,max_idx-1,self.n_landmarks)
        A = Xt[np.ix_(idx,idx)]
        # B = X[0:landmarks,landmarks:]
        F = Xt[self.n_landmarks:,0:self.n_landmarks]
        #C = X[landmarks:,landmarks:]
        U, S, H = np.linalg.svd(A, full_matrices=True)
        S = np.diag(S)

        U_k = np.concatenate([U,(F @H )@np.linalg.pinv(S)])
        self.subspace = U_k
        #V_k = np.concatenate([H, np.matmul(np.matmul(B.T,U),np.linalg.pinv(S))])
        Xt = U_k @S

        if type(Ys) is np.ndarray:
            A = self.classwise_sampling(Xs,Ys)
        else:
            A = Xs[np.ix_(idx,idx)]

        D = np.linalg.svd(A, full_matrices=True,compute_uv=False)
        Xs = U_k @ np.diag(D)
        self.Xt =preprocessing.scale(Xt)
        self.Xs =preprocessing.scale(Xs)
        self.Ys = Ys
        return self.Xt,self.Xs,self.Ys

    def classwise_sampling(self,Xs,Y):

        A = []
        classes = np.unique(Y)
        c_classes = classes.size
        samples_per_class = int(self.n_landmarks / c_classes)
        for c in classes:
            class_data = Xs[np.where(c == Y),:][0]

            if samples_per_class > class_data.shape[0]:
                A.list(class_data)
            else:
                A = A+list(class_data[np.random.randint(0,class_data.shape[0],samples_per_class),:self.n_landmarks])

        return np.array(A)


    def fit(self,Xt):
        '''
        Applies nyström approximation to  Xt. 
        Projects Xt into the subspace with normalization afterward.
        ----------
        Parameters
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        '''
        if type(Xt) is not np.ndarray:
            raise ValueError("Numpy Arrays must be given!")
        if type(self.n_landmarks ) is not int or self.n_landmarks  < 1:
            raise ValueError("Positive integer number must given!")
        self.n_landmarks = int(np.min(list(Xt.shape)+list(Xs.shape)+[self.n_landmarks ]))
        max_idx = np.min(list(Xt.shape)+list(Xs.shape))
        idx = np.random.randint(0,max_idx-1,self.n_landmarks )
        A = Xt[np.ix_(idx,idx)]
        # B = X[0:landmarks,landmarks:]
        F = Xt[self.n_landmarks :,0:self.n_landmarks ]
        #C = X[landmarks:,landmarks:]
        U, S, H = np.linalg.svd(A, full_matrices=True)
        S = np.diag(S)

        U_k = np.concatenate([U,(F @H )@np.linalg.pinv(S)])
        #V_k = np.concatenate([H, np.matmul(np.matmul(B.T,U),np.linalg.pinv(S))])
        Xt = U_k @S
        self.Xt = preprocessing.scale(Xt)
        self.subspace = U_k
        self.n_xt = Xt.shape[0]

    def transform(self,Xs,Ys):
        '''
        Augments Xs and Ys to fit sample sizes. Projects Xs into the subspace of Xt
       
        ----------
        Parameters
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        
        ----------
        Returns
        Xt : Augment and Projected
        Xs : augmented and projected
        Ys : Augmented source labels 

        '''
        if type(Xs) is not np.ndarray:
            raise ValueError("Numpy Arrays must be given!")
        if type(self.n_landmarks ) is not int or self.n_landmarks  < 1:
            raise ValueError("Positive integer number must given!")
        
        Ys,Xs = self.data_augmentation(Xs,self.n_xt,Ys)
        max_idx = np.min(list(Xt.shape)+list(Xs.shape))
        idx = np.random.randint(0,max_idx-1,self.n_landmarks)

        if type(Ys) is np.ndarray:
            A = self.classwise_sampling(Xs,Ys)
        else:
            A = Xs[np.ix_(idx,idx)]

        D = np.linalg.svd(A, full_matrices=True,compute_uv=False)
        Xs = self.subspace @ np.diag(D)
        self.Xs = preprocessing.scale(Xs) 
        self.Ys = Ys
        return self.Xt,self.Xs,self.Ys

    def data_augmentation(self,Xs,required_size,Y):
        """
        Data Augmentation
        Upsampling if Xs smaller as required_size via multivariate gaussian mixture
        Downsampling if Xs greater as required_size via uniform removal

        Note both are class-wise with goal to harmonize class counts
        
        ----------
        Parameters
        Xs : Matrix, where classifier is trained on
        required_size : Size to which Xs is reduced or extended
        Y : Label vector, which is reduced or extended like Xs

        ----------
        Returns
        Ys : Augmented 
        Xs : Augmented

        """
        if type(Xs) is not np.ndarray or type(required_size) is not int or type(Y) is not np.ndarray:
            raise ValueError("Numpy Arrays must be given!")
        if Xs.shape[0] == required_size:
            return Y,Xs
        
        _, idx = np.unique(Y, return_index=True)
        C = Y[np.sort(idx)].flatten().tolist()
        size_c = len(C)
        if Xs.shape[0] < required_size:
            print("Source smaller target")
            data = np.empty((0,Xs.shape[1]))
            label = np.empty((0,1))
            diff = required_size - Xs.shape[0]
            sample_size = int(np.floor(diff/size_c))
            for c in C:
                #indexes = np.where(Y[Y==c])
                indexes =  np.where(Y==c)
                class_data = Xs[indexes,:][0]
                m = np.mean(class_data,0) 
                sd = np.var(class_data,0)
                sample_size = sample_size if c !=C[-1] else sample_size+np.mod(diff,size_c)
                augmentation_data =np.vstack([np.random.normal(m, sd, size=len(m)) for i in range(sample_size)])
                data =np.concatenate([data,class_data,augmentation_data])
                label = np.concatenate([label,np.ones((class_data.shape[0]+sample_size,1))*c])
            
        if Xs.shape[0] > required_size:
            print("Source greater target")
            data = np.empty((0,Xs.shape[1]))
            label = np.empty((0,1))
            sample_size = int(np.floor(required_size/size_c))
            for c in C:
                indexes = np.where(Y[Y==c])[0]
                class_data = Xs[indexes,:]
                if len(indexes) > sample_size:
                    sample_size = sample_size if c !=C[-1] else np.abs(data.shape[0]-required_size)
                    y = np.random.choice(class_data.shape[0],sample_size)
                    class_data = class_data[y,:]
                data =np.concatenate([data,class_data])
                label = np.concatenate([label,np.ones((class_data.shape[0],1))*c])
        self.Xs = data
        self.Ys = label
        return self.Ys,self.Xs

if __name__ == "__main__":

    import os
    import scipy.io as sio
    from sklearn.svm import SVC


    os.chdir("../data/OfficeCaltech/surf")

    # Load and preprocessing of data. Note normalization to N(0,1) is necessary.
    dslr = sio.loadmat("dslr_SURF_L10.mat")
    Xs = preprocessing.scale(np.asarray(dslr["fts"]))
    Ys = np.asarray(dslr["labels"])

    amazon = sio.loadmat("amazon_SURF_L10.mat")
    Xt = preprocessing.scale(np.asarray(amazon["fts"]))
    Yt = np.asarray(amazon["labels"])

    # Applying SVM without transfer learning. Accuracy should be about 10%
    clf = SVC(gamma=1,C=10)
    clf.fit(Xs,Ys)
    print("SVM without transfer "+str(clf.score(Xt,Yt.ravel())))

    # Initialization of NSO. Accuracy of SVM + NSO should be about 90%
    nso = NSO(landmarks=100)
    # Compute domain invariant subspace data directly by 
    Xt,Xs,Ys = nso.fit_transform(Xt,Xs,Ys)
    
    # Or use two steps
    # nso.fit(Xt)
    # Xt,Xs,Ys = nso.transform(Xs,Ys)

    clf = SVC(gamma=1,C=10)
    clf.fit(Xs,Ys)
    print("SVM + NSO: "+str(clf.score(Xt,Yt.ravel())))

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(Xs, Ys.ravel())

    score = model.score(Xt, Yt)
    print("KNN + NSO: "+str(score))