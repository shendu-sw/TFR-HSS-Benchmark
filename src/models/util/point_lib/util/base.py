import os
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


TOL = 1e-14


class Base:
    """
    The observations are in matrix format
    """
    def __init__(self,u_obs,u,constant=298):
        self.u_obs = np.array(u_obs)
        self.u = np.array(u)
        self.constant = constant
        self.u_pred = None
        self.rows = None
        self.cols = None
        self.obser()

    def obser(self):
        [self.rows,self.cols] = np.where(self.u_obs>self.constant)

    def pred_init(self):
        self.u_pred = np.zeros_like(self.u)

    def train_samples(self):
        X_train = np.transpose(np.array([self.rows,self.cols])) / max(self.u.shape)
        y_train = np.transpose(self.u[self.rows,self.cols])
        return X_train, y_train

    def test_samples(self):
        samples = [[row,col] for row in range(self.u.shape[0]) for col in range(self.u.shape[1])]
        samples = np.array(samples) / max(self.u.shape)
        return samples

    def predict(self):
        pass


class BaseVec:
    """
    The observations are in matrix format
    """
    def __init__(self, root, train_list, constant=298):
        self.root = root
        self.train_list = train_list

        self.u_pred = None
        self.train_file = self.make_dataset(root, train_list)
        self.layout = None

        self.constant = constant

    def _loader(self, path, mode='train'):
        
        input = []
        
        output = []
        if mode == 'train':
            for _ in range(4*4):
                output.append([])
        else:
            pass

        #print((path[3]))
        num = 0
        for i in range(len(path)):
            num = num + 1
            #print(len(path))
            #print(i)
            source = np.array(sio.loadmat(path[i])['u_obs'])
            target = np.array(sio.loadmat(path[i])['u'])
            if self.layout is None:
                self.layout = np.array(sio.loadmat(path[i])['F'])
            else:
                pass

            indata = source[np.where(source>TOL)]
            input.append(indata)
            if mode == 'train':
                for k in range(4):
                    for kk in range(4):
                        sep = target[0+k:target.shape[0]:4, 0+kk:target.shape[1]:4].flatten()
                        output[k*4+kk].append(sep)
            elif mode == 'test':
                output.append(target)
            else:
                pass
            
            if num % 1000 == 0 :
                print("num:", num)

        return input, output

    def make_dataset(self, root_dir, list_path):
        files = []
        # root_dir = os.path.expanduser(root_dir)

        # root_path = os.path.dirname(list_path)
        base = os.path.dirname(list_path)
        print(base)
        test_name = os.path.splitext(os.path.basename(list_path))[0]
        subdir = os.path.join("train", "train") \
            if base=='train' else os.path.join("test", test_name)
        file_dir = os.path.join(root_dir, subdir)
        list_file = os.path.join(root_dir, list_path)
        print(file_dir)
        print(list_file)
        assert os.path.isdir(file_dir)
        with open(list_file, 'r') as rf:
            for line in rf.readlines():
                data_path = line.strip()
                path = os.path.join(file_dir, data_path)
                files.append(path)
        return files

    def train_samples(self):
        #print(self.train_file)
        X_train, y_train = self._loader(self.train_file)
        return X_train, y_train

    def test_samples(self, test_path):
        test_file = self.make_dataset(self.root, test_path)
        X_test, y_test = self._loader(test_file, mode='test')

        return X_test, np.array(y_test)

    def predict(self):
        pass



if __name__ == '__main__':
    #m=sio.loadmat('Example0.mat')
    #u_obs=m['u_obs']
    #u=m['u']
    #sample = Base(u_obs,u)
    #sample.vec()
    root = 'g:/gong/recon_project/TFRD/HSink/'
    train_list = 'g:/gong/recon_project/TFRD/HSink/train/train_val.txt'
    test_list = 'g:/gong/recon_project/TFRD/HSink/train/test_0.txt'

    sample = BaseVec(root, train_list)

    a, b = sample.train_samples()
    print(a)
    print(b[0])


