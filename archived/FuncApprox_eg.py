import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import solve

class LeastSquaresApproximation:
    """
    A class to compute least-squares approximations of continuous functions
    using different basis functions.
    """
    
    def __init__(self, function, interval, weight_function = None):
        """
        Initialize the approximation.
        
        Parameters:
        - function: The continuous function to approximate
        - interval: Tuple (a, b) defining the approximation interval
        - weight_function: Weight function for weighted least squares (default: 1)
        """
        self.function = function
        self.a, self.b = interval
        self.weight_function = weight_function if weight_function else lambda x: 1.0
        
    def polynomial_approximation(self, degree):
        """
        Compute polynomial least-squares approximation using monomial basis.
        
        Parameters:
        - degree: Degree of the polynomial approximation
        
        Returns:
        - coefficients: Coefficients of the polynomial [a0, a1, ..., an]
        - polynomial: Function that evaluates the approximation
        """
        # Build the Gram matrix and right-hand side vector
        n = degree + 1
        G = np.zeros((n, n))
        b = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                # Compute inner product <x^i, x^j>
                integrand = lambda x: (x**i) * (x**j) * self.weight_function(x)
                G[i, j], _ = quad(integrand, self.a, self.b)
            
            # Compute inner product <f, x^i>
            integrand_f = lambda x: self.function(x) * (x**i) * self.weight_function(x)
            b[i], _ = quad(integrand_f, self.a, self.b)
        
        # Solve the linear system G * c = b
        coefficients = solve(G, b)
        
        # Create the polynomial function
        def polynomial(x):
            result = 0
            for i, coeff in enumerate(coefficients):
                result += coeff * (x**i)
            return result
        
        return coefficients, polynomial
    
    def legendre_approximation(self, degree):
        """
        Compute least-squares approximation using Legendre polynomials.
        
        Parameters:
        - degree: Degree of the approximation
        
        Returns:
        - coefficients: Coefficients for Legendre polynomials
        - approximation: Function that evaluates the approximation
        """
        # Generate Legendre polynomials using recurrence relation
        def legendre_poly(n):
            if n == 0:
                return lambda x: np.ones_like(x)
            elif n == 1:
                return lambda x: x
            else:
                def poly(x):
                    P_prev_prev = np.ones_like(x)  # P0
                    P_prev = x                     # P1
                    for k in range(2, n + 1):
                        P_current = ((2 * k - 1) * x * P_prev - (k - 1) * P_prev_prev) / k
                        P_prev_prev, P_prev = P_prev, P_current
                    return P_prev
                return poly
        
        coefficients = []
        legendre_polys = []
        
        for n in range(degree + 1):
            P_n = legendre_poly(n)
            legendre_polys.append(P_n)
            
            # Compute coefficient: c_n = <f, P_n> / <P_n, P_n>
            numerator_integrand = lambda x: self.function(x) * P_n(x) * self.weight_function(x)
            denominator_integrand = lambda x: P_n(x) * P_n(x) * self.weight_function(x)
            
            numerator, _ = quad(numerator_integrand, self.a, self.b)
            denominator, _ = quad(denominator_integrand, self.a, self.b)
            
            coefficients.append(numerator / denominator)
        
        def approximation(x):
            result = 0
            for i, coeff in enumerate(coefficients):
                result += coeff * legendre_polys[i](x)
            return result
        
        return coefficients, approximation
    
    def fourier_approximation(self, num_terms):
        """
        Compute Fourier series approximation.
        
        Parameters:
        - num_terms: Number of terms in the Fourier series
        
        Returns:
        - coefficients: Dictionary with a0, an, bn coefficients
        - approximation: Function that evaluates the approximation
        """
        L = (self.b - self.a) / 2
        
        # Compute coefficients
        a0_integrand = lambda x: self.function(x) * self.weight_function(x)
        a0, _ = quad(a0_integrand, self.a, self.b)
        a0 /= (self.b - self.a)
        
        an_coeffs = []
        bn_coeffs = []
        
        for n in range(1, num_terms + 1):
            # Compute an
            an_integrand = lambda x: self.function(x) * np.cos(n * np.pi * x / L) * self.weight_function(x)
            an, _ = quad(an_integrand, self.a, self.b)
            an /= L
            an_coeffs.append(an)
            
            # Compute bn
            bn_integrand = lambda x: self.function(x) * np.sin(n * np.pi * x / L) * self.weight_function(x)
            bn, _ = quad(bn_integrand, self.a, self.b)
            bn /= L
            bn_coeffs.append(bn)
        
        coefficients = {'a0': a0, 'an': an_coeffs, 'bn': bn_coeffs}
        
        def approximation(x):
            result = a0
            for n in range(1, num_terms + 1):
                result += an_coeffs[n-1] * np.cos(n * np.pi * x / L)
                result += bn_coeffs[n-1] * np.sin(n * np.pi * x / L)
            return result
        
        return coefficients, approximation
    
    def compute_error(self, approximation_func, num_points=1000):
        """
        Compute the L2 error of the approximation.
        
        Parameters:
        - approximation_func: The approximation function
        - num_points: Number of points for discrete error computation
        
        Returns:
        - error: L2 error
        """
        x_vals = np.linspace(self.a, self.b, num_points)
        f_vals = self.function(x_vals)
        approx_vals = approximation_func(x_vals)
        
        # Discrete L2 error
        error = np.sqrt(np.sum((f_vals - approx_vals)**2) / num_points)
        return error

def example_usage():
    """
    Example demonstrating the usage of the LeastSquaresApproximation class.
    """
    # Define the function to approximate
    def target_function(x):
        return np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)
    
    # Define the interval
    interval = (-1, 1)
    
    # Create approximation object
    approximator = LeastSquaresApproximation(target_function, interval)
    
    # Compute different approximations
    print("Computing least-squares approximations...")
    
    # Polynomial approximation
    poly_degree = 5
    poly_coeffs, poly_approx = approximator.polynomial_approximation(poly_degree)
    poly_error = approximator.compute_error(poly_approx)
    print(f"Polynomial approximation (degree {poly_degree}) error: {poly_error:.6f}")
    
    # Legendre polynomial approximation
    legendre_degree = 5
    legendre_coeffs, legendre_approx = approximator.legendre_approximation(legendre_degree)
    legendre_error = approximator.compute_error(legendre_approx)
    print(f"Legendre approximation (degree {legendre_degree}) error: {legendre_error:.6f}")
    
    # Fourier approximation
    fourier_terms = 6
    fourier_coeffs, fourier_approx = approximator.fourier_approximation(fourier_terms)
    fourier_error = approximator.compute_error(fourier_approx)
    print(f"Fourier approximation ({fourier_terms} terms) error: {fourier_error:.6f}")
    
    # Plot the results
    x_plot = np.linspace(interval[0], interval[1], 1000)
    y_true = target_function(x_plot)
    y_poly = poly_approx(x_plot)
    y_legendre = legendre_approx(x_plot)
    y_fourier = fourier_approx(x_plot)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x_plot, y_true, 'k-', linewidth=2, label='True function')
    plt.plot(x_plot, y_poly, 'r--', linewidth=1.5, label=f'Polynomial (deg {poly_degree})')
    plt.title('Polynomial Approximation')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(x_plot, y_true, 'k-', linewidth=2, label='True function')
    plt.plot(x_plot, y_legendre, 'g--', linewidth=1.5, label=f'Legendre (deg {legendre_degree})')
    plt.title('Legendre Polynomial Approximation')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(x_plot, y_true, 'k-', linewidth=2, label='True function')
    plt.plot(x_plot, y_fourier, 'b--', linewidth=1.5, label=f'Fourier ({fourier_terms} terms)')
    plt.title('Fourier Series Approximation')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(x_plot, np.abs(y_true - y_poly), 'r-', label='Polynomial error')
    plt.plot(x_plot, np.abs(y_true - y_legendre), 'g-', label='Legendre error')
    plt.plot(x_plot, np.abs(y_true - y_fourier), 'b-', label='Fourier error')
    plt.title('Approximation Errors')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'polynomial': (poly_coeffs, poly_approx, poly_error),
        'legendre': (legendre_coeffs, legendre_approx, legendre_error),
        'fourier': (fourier_coeffs, fourier_approx, fourier_error)
    }

def weighted_example():
    """
    Example demonstrating weighted least-squares approximation.
    """
    def target_function(x):
        return np.exp(-x) * np.sin(2 * np.pi * x)
    
    # Use a weight function that emphasizes the center of the interval
    def weight_function(x):
        return np.exp(-x**2)
    
    interval = (-2, 2)
    approximator = LeastSquaresApproximation(target_function, interval, weight_function)
    
    # Compute weighted polynomial approximation
    degree = 4
    coeffs, approx_func = approximator.polynomial_approximation(degree)
    
    # Plot results
    x_plot = np.linspace(interval[0], interval[1], 1000)
    y_true = target_function(x_plot)
    y_approx = approx_func(x_plot)
    weight_vals = weight_function(x_plot)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_true, 'k-', linewidth=2, label='True function')
    plt.plot(x_plot, y_approx, 'r--', linewidth=1.5, label=f'Weighted approximation (deg {degree})')
    plt.plot(x_plot, weight_vals, 'g:', linewidth=1, label='Weight function')
    plt.title('Weighted Least-Squares Approximation')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("Least-Squares Function Approximation Demo")
    print("=" * 50)
    
    # Run the main example
    results = example_usage()
    
    print("\n" + "=" * 50)
    print("Weighted Least-Squares Example")
    print("=" * 50)
    
    # Run the weighted example
    weighted_example()
