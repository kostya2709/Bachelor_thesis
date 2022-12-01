import numpy as np
from random import uniform

def to_str( elem):
    return str(round(elem, 3))

def generate_data( point_range: list, function):
    return [to_str(function(elem)) for elem in point_range]

def get_range( left: int, right: int, points_num: int):
    step = (right - left) / points_num
    return np.arange( left, right, step)

def write_answer( data_name: str, pred_name: str, point_range: list, sample_num: int, given_num: int):
    file_data = open( data_name, "w")
    file_pred = open( pred_name, "w")
    data_str = ""
    pred_str = ""
    for i in range(sample_num):
        data = generate_data( point_range, lambda x: np.sin(x) + uniform(0, 0.05))
        data_str += " ".join(data[:given_num])
        pred_str += " ".join(data[given_num:])
        if i != sample_num - 1:
            data_str += "\n"
            pred_str += "\n"
    file_data.write( data_str)
    file_pred.write( pred_str)
    file_data.close()
    file_pred.close()

LEFT = 0
RIGHT = 15
SAMPLE_NUM = 10
POINTS_NUM = 20
KNOWN_NUM = 12

if __name__ == "__main__":

    point_range = get_range( LEFT, RIGHT, POINTS_NUM)
    write_answer( "data", "pred", point_range, SAMPLE_NUM, KNOWN_NUM)
