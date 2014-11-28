
import os
import struct
import numpy as np

class Mfcc():

    def __init__(self,mfcc_dir1, mfcc_dir2):
        self.mfcc_dir1 = mfcc_dir1
        self.mfcc_dir2 = mfcc_dir2

    def load_mfcc(self,mfcc_file, m):

        mfcc = []
        fp = open(mfcc_file, "rb")
        while True:
            b = fp.read(4)
            if b == "": break
            val = struct.unpack("f", b)[0]
            mfcc.append(val)
        fp.close()

        mfcc = np.array(mfcc)
        num_frame = len(mfcc) / m
        mfcc = mfcc.reshape(num_frame, m)

        return mfcc

    def create_mfcc(self,mfccDir):

        all_mfcc_1d = []

        for i,file in enumerate(os.listdir(mfccDir)):
            if not file.endswith(".mfc"): continue
            mfccFile = os.path.join(mfccDir, file)

            mfcc = self.load_mfcc(mfccFile, 20)
            mfcc_1d = []
            for i in xrange(len(mfcc)):
                mfcc_1d.extend(mfcc[i])

            all_mfcc_1d.append(mfcc_1d)

        return all_mfcc_1d

    def get_data(self):

        mfccs1 = self.create_mfcc(self.mfcc_dir1)
        num_mfccs1 = len(mfccs1)
        classes1 = [ 0 for _ in xrange(num_mfccs1)]

        mfccs2 = self.create_mfcc(self.mfcc_dir2)
        num_mfccs2 = len(mfccs2)
        classes2 = [ 1 for _ in xrange(num_mfccs2)]

        data_x=[]
        data_x.extend(mfccs1)
        data_x.extend(mfccs2)
        data_y=[]
        data_y.extend(classes1)
        data_y.extend(classes2)

        return data_x, data_y

