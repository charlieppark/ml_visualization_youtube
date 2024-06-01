from manim import *
import numpy as np

class SteepAndFlatGraph(ThreeDScene):
    def construct(self):
        # 기본 설정
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-1, 9, 1],
            x_length=6,
            y_length=6,
            z_length=6,
        ).add_coordinates()

        # 첫 번째 함수 f(x, y) = x^2 + y^4
        def func(x, y):
            return x**2 + y**4

        graph = Surface(
            lambda u, v: axes.c2p(u, v, func(u, v)),
            u_range=[-3, 3],
            v_range=[-3, 3],
            fill_opacity=0.75,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )

        # 두 번째 함수 g(x, y) = x^2 + y^4 + lambda * (x^2 + y^2)
        lambda_value = 0.5

        def func_with_regularization(x, y, lambda_value):
            return x**2 + y**4 + lambda_value * (x**2 + y**2)

        graph_with_regularization = Surface(
            lambda u, v: axes.c2p(u, v, func_with_regularization(u, v, lambda_value)),
            u_range=[-3, 3],
            v_range=[-3, 3],
            fill_opacity=0.75,
            checkerboard_colors=[GREEN_D, GREEN_E],
        )

        # 그래디언트 디센트 함수
        def gradient_descent(f_grad, start, learning_rate, steps):
            points = [start]
            current_point = np.array(start)
            for _ in range(steps):
                grad = np.array(f_grad(*current_point))
                next_point = current_point - learning_rate * grad
                points.append(next_point)
                current_point = next_point
            return points

        # 첫 번째 함수의 그래디언트
        def grad_func(x, y):
            return np.array([2*x, 4*y**3])

        # 두 번째 함수의 그래디언트
        def grad_func_with_regularization(x, y, lambda_value):
            return np.array([2*x * (1 + lambda_value), 4*y**3 + 2*lambda_value*y])

        # 시작점, 학습률, 스텝 수
        start_point = np.array([2.5, 2.5])
        learning_rate = 0.01
        steps = 50

        # 그래디언트 디센트를 사용하여 점들의 궤적 계산
        points = gradient_descent(grad_func, start_point, learning_rate, steps)
        points_with_regularization = gradient_descent(
            lambda x, y: grad_func_with_regularization(x, y, lambda_value),
            start_point, learning_rate, steps
        )

        # 점 생성
        dot = Dot3D(point=axes.c2p(*start_point, func(*start_point)), color=YELLOW)
        dot_with_regularization = Dot3D(point=axes.c2p(*start_point, func_with_regularization(*start_point, lambda_value)), color=RED)

        # 등고선 축 설정
        contour_axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            axis_config={"include_numbers": True}
        ).to_edge(RIGHT)

        # 첫 번째 함수의 등고선
        contour_graphs = VGroup(*[
            contour_axes.plot_implicit_curve(lambda x, y: func(x, y) - c, color=BLUE)
            for c in np.arange(1, 6)
        ])

        # 두 번째 함수의 등고선
        contour_graphs_with_regularization = VGroup(*[
            contour_axes.plot_implicit_curve(
                lambda x, y: func_with_regularization(x, y, lambda_value) - c, color=GREEN
            )
            for c in np.arange(1, 6)
        ])

        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        self.play(Create(axes))
        self.play(Create(contour_axes), Create(contour_graphs), Create(graph))
        self.play(Create(dot))

        # 그래디언트 디센트를 통한 이동 애니메이션 (첫 번째 함수)
        for point in points:
            self.play(
                dot.animate.move_to(axes.c2p(point[0], point[1], func(*point))),
                Create(Dot(contour_axes.c2p(point[0], point[1]), color=YELLOW)),
                run_time=0.1
            )

        self.wait(1)

        # 곡률이 변한 후의 그래프와 점의 이동
        self.play(Transform(contour_graphs, contour_graphs_with_regularization), Transform(graph, graph_with_regularization))
        self.play(Transform(dot, dot_with_regularization))
        
        for point in points_with_regularization:
            self.play(
                dot_with_regularization.animate.move_to(axes.c2p(point[0], point[1], func_with_regularization(*point, lambda_value))),
                Create(Dot(contour_axes.c2p(point[0], point[1]), color=RED)),
                run_time=0.1
            )

        self.wait(2)

# To run this code, ensure you have Manim installed and set up properly.
# Save this code in a Python file and run it using the command:
# manim -pql your_script.py SteepAndFlatGraph
