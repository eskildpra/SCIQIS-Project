import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider, Button

def fock_state(n, dim):
    state = np.zeros((dim, 1), dtype=complex)
    if n < dim:
        state[n, 0] = 1.0
    return state
def coherent_state(alpha, dim):
    state = np.zeros((dim, 1), dtype=complex)
    for n in range(dim):
        state[n, 0] = (alpha**n / np.sqrt(factorial(n))) * np.exp(-0.5 * np.abs(alpha)**2)
    return state / np.linalg.norm(state)

def creation_op(dim):
    a_dag = np.zeros((dim, dim), dtype=complex)
    for n in range(dim - 1):
        a_dag[n + 1, n] = np.sqrt(n + 1)
    return a_dag

def raise_state(state, a_dag):
    new_state = a_dag @ state
    norm = np.linalg.norm(new_state)
    if norm < 1e-15:
        return state 
    return new_state / norm

def X_op(dim):
    a_dag = creation_op(dim)
    a = np.conjugate(a_dag.T)
    return (a + a_dag) / np.sqrt(2)

def P_op(dim):
    a_dag = creation_op(dim)
    a = np.conjugate(a_dag.T)
    return 1j * (a_dag - a) / (np.sqrt(2))

def number_op(dim):
    a_dag = creation_op(dim)
    a = np.conjugate(a_dag.T)
    return a_dag @ a

def uncertaintyX(state, dim):
    # Delta A = sqrt(<A^2> - <A>^2)
    # første <A^2> = <state|A^2|state>
    X = X_op(dim)
    term1 = (np.conjugate(state.T) @ (X @ X) @ state).item()
    # anden <A>^2 = (<state|A|state>)^2
    term2 = (np.conjugate(state.T) @ X @ state).item()**2
    var = float(np.real(term1 - term2))
    return np.sqrt(max(0.0, var))

def uncertaintyP(state, dim):
    P = P_op(dim)
    term1 = (np.conjugate(state.T) @ (P @ P) @ state).item()
    term2 = (np.conjugate(state.T) @ P @ state).item()**2
    var = float(np.real(term1 - term2))
    return np.sqrt(max(0.0, var))
    

def extract_uncertainty(state, dim):
    deltaX = uncertaintyX(state, dim)
    deltaP = uncertaintyP(state, dim)
    return deltaX, deltaP

def extract_angle_length_coherent(state, dim):
    a = np.conjugate(creation_op(dim).T)
    alpha_exp = (state.T.conj() @ a @ state).item()
    length = np.abs(alpha_exp)
    angle = np.angle(alpha_exp) 
    return angle, length


def plot_coherent_state(alpha, dim, max_x=10, max_p=10):
    alphaR_init = np.real(alpha)
    alphaI_init = np.imag(alpha)
    n_init = 5
    
    a_dag = creation_op(dim)
    
    def generate_current_state(r, i, n):
        current_alpha = r + 1j * i
        current_state = coherent_state(current_alpha, dim)
        for _ in range(n):
            current_state = raise_state(current_state, a_dag)
        return current_state, current_alpha

    state, current_alpha = generate_current_state(alphaR_init, alphaI_init, n_init)

    deltax_mat, deltap_mat = extract_uncertainty(state, dim)
    deltax = float(np.real(np.ravel(deltax_mat)[0]))
    deltap = float(np.real(np.ravel(deltap_mat)[0]))
    angle, length = extract_angle_length_coherent(state, dim)
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.35, left=0.25)
    xpos = np.sqrt(2) * length * np.cos(angle)
    ypos = np.sqrt(2) * length * np.sin(angle)

    scatter = ax.scatter(xpos, ypos, color='blue', label='State center')
    uncertainty_ellipse = Ellipse(
        xy=(xpos, ypos), 
        width=2 * deltax, 
        height=2 * deltap, 
        edgecolor='blue', 
        facecolor='blue', 
        alpha=0.3,
        label='Uncertainty'
    )
    ax.add_patch(uncertainty_ellipse)
    
    ax.set_xlim(-max_x, max_x)
    ax.set_ylim(-max_p, max_p)
    ax.set_xlabel('X')
    ax.set_ylabel('P')
    ax.set_title(f'State: alpha={current_alpha:.1f}, n={n_init}')
    ax.set_aspect('equal')
    ax.legend()
    
    ax_alphar = fig.add_axes([0.25, 0.20, 0.65, 0.03])
    re_slider = Slider(ax_alphar, 'Re(alpha)', -5, 5, valinit=alphaR_init)
    
    ax_alphai = fig.add_axes([0.25, 0.12, 0.65, 0.03])
    im_slider = Slider(ax_alphai, 'Im(alpha)', -5, 5, valinit=alphaI_init)
    
    ax_n = fig.add_axes([0.25, 0.04, 0.65, 0.03])
    n_slider = Slider(ax_n, 'Photons (n)', 0, 25, valinit=n_init, valstep=1)
    def update(val):
        r = re_slider.val
        i = im_slider.val
        n_val = int(n_slider.val)
        new_state, new_alpha = generate_current_state(r, i, n_val)
        n_deltax_mat, n_deltap_mat = extract_uncertainty(new_state, dim)
        n_deltax = float(np.real(np.ravel(n_deltax_mat)[0]))
        n_deltap = float(np.real(np.ravel(n_deltap_mat)[0]))
        n_angle, n_length = extract_angle_length_coherent(new_state, dim)
        n_xpos = np.sqrt(2) * n_length * np.cos(n_angle)
        n_ypos = np.sqrt(2) * n_length * np.sin(n_angle)
        scatter.set_offsets(np.c_[n_xpos, n_ypos])
        uncertainty_ellipse.center = (n_xpos, n_ypos)
        uncertainty_ellipse.width = 2 * n_deltax
        uncertainty_ellipse.height = 2 * n_deltap
        
        ax.set_title(f'State: alpha={new_alpha:.1f}, n={n_val}')
        fig.canvas.draw_idle()

    re_slider.on_changed(update)
    im_slider.on_changed(update)
    n_slider.on_changed(update)
    
    plt.show()
    return re_slider, im_slider, n_slider
