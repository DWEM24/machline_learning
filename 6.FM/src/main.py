import sys
sys.path.append("..")

from FM import FMLayer
import numpy as np
from keras.layers import Dense, Input, Dropout, Embedding, Conv1D, GlobalMaxPooling1D
from keras.models import Model
from scipy.sparse import csr_matrix


def read_data(path):
    with open(path) as f:
        lines = f.readlines()[0:4]
    lines = [line.strip().split("\t") for line in lines]
    y_data = []
    data = []
    col = []
    row = []
    f = -1
    for i, line in enumerate(lines):
        y_data.append(int(line[0]))
        for ss in line[1:]:
            c, num = ss.split(":")
            f = max(f, int(c))
            data.append(float(num))
            row.append(i)
            col.append(int(c))
    x_data = csr_matrix((data, (row, col)), shape=(len(lines), f + 1))
    y_data = np.array(y_data)
    return x_data, y_data, f + 1


x_data, y_data, dim = read_data("../data/raw_data")
input1 = Input(shape=(dim,))
out = FMLayer(4)(input1)
# out=DeepWideLayer()(input1)
# out=FMDeepLayer()(input1)
model = Model(inputs=input1, outputs=out)
model.compile(loss="binary_crossentropy", optimizer="Adam")
model.fit(x_data, y_data, epochs=100, batch_size=128, shuffle=True)
model.save("../model/model_FM.h5")