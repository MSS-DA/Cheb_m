import numpy as np

# import logging
# logging for better quality
# logging.basicConfig(filename="Computations.log", level=logging.INFO, filemode='w')


class ChebyshevCollocaton:
    """
    Class providing Chebyshev derivative matrices and mesh points.
    Creates chebyshev derivative matrices in computational domain [-1, 1]
    Has useful scaling methods for various domains.
    """

    def __init__(self, n):
        """
        :arg N - number of mesh nodes
        :return nothing
        """
        assert type(n) == int
        assert n > 1
        self.n = n
        (self.mesh, self.D) = ChebyshevCollocaton.create_mesh_and_derivative(n)

    @staticmethod
    def create_mesh_and_derivative(n):
        """
        For given number of points N on [-1, 1] returns Gauss-Lobatto mesh and D-matrix
        :arg n - number of mesh nodes
        :return (mesh[N], D-matrix[NxN])
        """

        @np.vectorize
        def f(i, j):

            def c(k):
                return 2 if k == 0 or k == n - 1 else 1

            if i != j:
                return -c(i) * ((-1) ** (i + j)) / (2 * c(j) * np.sin((i + j) * np.pi / (2 * (n - 1))) * np.sin(
                    (i - j) * np.pi / (2 * (n - 1))))
            else:
                if i == 0 and j == 0:
                    return (2 * (n - 1) ** 2 + 1) / 6
                elif i == n - 1 and j == n - 1:
                    return -(2 * (n - 1) ** 2 + 1) / 6
                else:
                    return -np.cos(np.pi * i / (n - 1)) / (2 * np.sin(j * np.pi / (n - 1)) ** 2)

        return np.cos(np.pi * np.arange(0, n) / (n - 1)), np.fromfunction(f, (n, n), dtype=int)

    def scale_for_semi_infinite(self, **kwargs):
        """
        Scales mesh, D-matrix and D^2-matrix for domain using f: x -> l (x + a) / (b - x).
        [y_min, y_max] -> [-1, 1]
        l - tuning parameter, l > 0
        :return (mesh[N], D-matrix[NxN], D^2-matrix[NxN])
        """
        y_max = kwargs.get("y_max")
        assert y_max is not None
        y_min = kwargs.get('y_min', 0.0)
        l = kwargs.get('l', 1.0)

        assert l > 0
        assert y_min < y_max

        a = (y_max + y_min) / (y_max - y_min) + (2 * y_min * y_max) / (y_max - y_min) / l
        b = (y_max + y_min) / (y_max - y_min) + 2 * l / (y_max - y_min)
        y_mesh_flow = l * (a + self.mesh) / (b - self.mesh)
        dz_dy = (b + a) * l / ((l + y_mesh_flow) ** 2)
        d2z_dy2 = -2 * dz_dy / (l + y_mesh_flow)

        dz_dy = np.diag(dz_dy)
        d2z_dy2 = np.diag(d2z_dy2)
        d_2_flow = dz_dy @ dz_dy @ (self.D @ self.D) + d2z_dy2 @ self.D
        d_1_flow = dz_dy @ self.D
        return y_mesh_flow, d_1_flow, d_2_flow

    def scale_for_interval(self, **kwargs):
        """
        Scales mesh, D-matrix and D^2-matrix for domain using f: x -> (r - l) x / 2 + (r + l) / 2.
        l - left end
        r - right end
        :return: (mesh[N], D-matrix[NxN], D^2-matrix[NxN])
        """

        r = kwargs.get("r")
        l = kwargs.get("l")
        assert r is not None
        assert l is not None
        assert r > l

        dz_dy = 2.0 / (r - l)
        y_mesh_coat = (r - l) / 2 * self.mesh + (l + r) / 2
        d_2_coat = (dz_dy * dz_dy) * self.D @ self.D
        d_1_coat = dz_dy * self.D
        return y_mesh_coat, d_1_coat, d_2_coat


if __name__ == '__main__':
    print('Don\'t use as exe')
