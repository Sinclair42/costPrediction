import numpy as np
import plotly.express as px


class Coefficient:
    """
    선형회귀에서 기울기와 절편의 정보를 담고 있는 클래스
    """
    def __init__(self, W, b):
        """
        객체 선언 시 실행되는 메서드
        :param W: 선형회귀에서의 기울기
        :param b: 선형회귀에서의 절편
        """
        self.W = W
        self.b = b

    def __repr__(self):
        """
        객체를 출력 시 실행되는 메서드
        :return: 객체 형식으로 출력. 각각 기울기와 절편을 나타냄
        """
        return f'Coefficient({self.W}, {self.b})'


class LinearRegression:
    """
    단순선형회귀를 위한 클래스
    """
    def __init__(self, input_data: np.ndarray, output_data: np.ndarray):
        """
        객체 선언 시 실행되는 메서드
        :param input_data: 입력 데이터
        :param output_data: 출력 정답 데이터
        """
        self.input_data = input_data
        self.output_data = output_data

    def prediction(self, coefficient: Coefficient):
        """
        예상되는 값을 반환하는 메서드
        :param coefficient: 예측 시 사용될 기울기와 절편이 포함된 Coefficient 객체
        :return: 예상 값을 numpy 배열로 반환
        """
        y_hat = coefficient.W * self.input_data + coefficient.b
        return y_hat

    def rss(self, coefficient: Coefficient):
        """
        계수(Coefficient 객체)에 따라 오차제곱합(RSS)를 반환하는 메서드
        :param coefficient: 사용될 Coefficient 객체
        :return: 계수에 따라 RSS 값을 반환
        """
        y_hat = self.prediction(coefficient)
        rss = np.sum((y_hat - self.output_data) ** 2)
        return rss

    def find_min(self):
        """
        절편을 첫째 값으로 고정하고 RSS가 최소가 되게 하는 기울기를 찾는 메서드
        :return: 기울기가 최소가 되게 하는 Coefficient 객체
        """
        result = Coefficient(0, self.output_data[0])
        b = result.b
        result.W = (np.sum(self.input_data * self.output_data) - b * np.sum(self.input_data)) / np.sum(
            self.input_data ** 2)

        return result

    def draw_scatter(self):
        """
        객체의 입력 데이터와 출력 데이터에 따라 산점도를 그리는 메서드
        :return: 반환하지 않음
        """
        fig = px.scatter(x=self.input_data, y=self.output_data)
        fig.show()

    def draw_line(self, coefficient: Coefficient):
        """
        산점도와 선형 회귀시 만들어지는 직선을 그리는 메서드
        :param coefficient: 사용된 Coefficient 객체
        :return: 반환하지 않음
        """
        fig = px.scatter(x=self.input_data, y=self.output_data)
        fig.add_scatter(x=self.input_data, y=self.prediction(coefficient))
        fig.show()
