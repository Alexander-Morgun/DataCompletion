import numpy as np
import pandas as pd


def project(vector, direction):
    return vector.dot(direction) / direction.dot(direction)


def error(x, y):
    d = x - y
    return d.dot(d)


class Filler:
    def __init__(self):
        self._center = np.array([6506.07284097,
                                2237.70351374,
                                1373.64749769,
                                45781.33820179,
                                9371.19896322])
        self._directions = [np.array([1705616.13070135,
                                     21780065.57442008,
                                     30426.32993618,
                                     120009.52858987,
                                     362198.89296095]),
                            np.array([2.87761285e+07,
                                     6.29836123e+05,
                                     -1.85603967e+03,
                                     3.44091942e+03,
                                     -9.36307422e+03])]

    def predict_point(self, point):
        mask = ~np.isnan(point)
        filled_point = np.nan_to_num(point)
        projections = [project((filled_point - self._center) * mask,
                               direction * mask)
                       for direction in self._directions]
        predictions = [(projection * direction + self._center).astype(int)
                       for projection, direction in zip(projections,
                                                        self._directions)]
        errors = [error(prediction * mask, filled_point)
                  for prediction in predictions]
        prediction = predictions[np.argmin(errors)]
        prediction[prediction < 0] = 0
        return filled_point * mask + prediction * ~mask

    def predict(self, X):
        answer = np.zeros(X.shape)
        for idx, point in enumerate(X):
            answer[idx] = self.predict_point(point)
        return answer


data = pd.read_csv('in.csv', index_col=0)
filler = Filler()
answer = pd.DataFrame(filler.predict(data.values),
                      columns=data.columns,
                      index=data.index)
answer.to_csv("out.csv")
