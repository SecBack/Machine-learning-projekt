import random


class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.values = []

        for i in range(0, self.rows):
            self.values.append([])
            for j in range(0, self.cols):
                self.values[i].append(0)

    @staticmethod
    def tomatrix(array):
        if isinstance(array[0], list):
            result = Matrix(len(array), len(array[0]))
            for i in range(0, len(array)):
                for j in range(0, len(array[0])):
                    result.values[i][j] = array[i][j]
            return result
        elif isinstance(array[0], int):
            result = Matrix(1, len(array))
            for i in range(0, len(array)):
                for j in range(0, len(array)):
                    result.values[0][i] = array[i]
            return result

    def randomize(self):
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                self.values[i][j] = random.randint(-1, 1)

    def add(self, n):
        if isinstance(n, Matrix):
            for i in range(0, self.rows):
                for j in range(0, self.cols):
                    self.values[i][j] += n.values[i][j]
            # print("add result: ", self.values)
        else:
            for i in range(0, self.rows):
                for j in range(0, self.cols):
                    self.values[i][j] += n

    @staticmethod
    def subtract(m1, m2):
        if m1.rows != m2.rows and m1.cols != m2.cols:
            raise Exception(
                "To subtract two arrays element-wise, the matricies must be the same size - m1 rows: {} m2 rows: {} and m1 cols: {} m2 cols: {}".format(
                    m1.rows, m2.rows, m1.cols, m2.cols
                )
            )
        else:
            result = Matrix(m1.rows, m2.cols)

            for i in range(0, result.rows):
                for j in range(0, result.cols):
                    result.values[i][j] = m1.values[i][j] - m2.values[i][j]
            return result

    @staticmethod
    def multiply(m1, m2):
        if m1.cols != m2.rows:
            raise Exception(
                "number of m1 cols was {} and number of m2 rows was {}, must be equal".format(
                    m1.cols, m2.rows
                )
            )
        elif m1.cols == m2.rows:
            result = Matrix(m1.rows, m2.cols)
            print("mul m1 rows: ", m1.rows)
            print("mul m1 cols: ", m1.cols)
            print("mul m1 vals: ", m1.values)
            print("mul m2 rows: ", m2.rows)
            print("mul m2 cols: ", m2.cols)
            print("mul m2 vals: ", m2.values)

            for i in range(0, result.rows):
                for j in range(0, result.cols):
                    sum = 0
                    for k in range(0, m1.cols):
                        sum += m1.values[i][k] * m2.values[k][j]
                    result.values[i][j] = sum
            print("mul result: ", result.values, "\n")
            return result

    @staticmethod
    def elemulti(m1, m2):
        if m1.rows != m2.rows and m1.cols != m2.cols:
            raise Exception(
                "To multiply two arrays element-wise, the matricies must be the same size - m1 rows: {} m2 rows: {} and m1 cols: {} m2 cols: {}".format(
                    m1.rows, m2.rows, m1.cols, m2.cols
                )
            )
        else:
            result = Matrix(m1.rows, m2.cols)

            for i in range(0, result.rows):
                for j in range(0, result.cols):
                    result.values[i][j] = m1.values[i][j] * m2.values[i][j]
            return result

    def scale(self, n):
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                self.values[i][j] *= n

    def transpose(self):
        result = Matrix(self.cols, self.rows)
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                result.values[j][i] = self.values[i][j]
        return result

    @staticmethod
    def statranse(m):
        result = Matrix(m.cols, m.rows)

        for i in range(0, m.rows):
            for j in range(0, m.cols):
                result.values[j][i] = m.values[i][j]
        return result

    def map(self, m, func, *args):
        func(m, *args)

    @staticmethod
    def stamap(m, func):
        result = Matrix(m.rows, m.cols)

        for i in range(0, m.rows):
            for j in range(0, m.cols):
                val = func(m.values[i][j])
                result.values[i][j] = val
        return result
