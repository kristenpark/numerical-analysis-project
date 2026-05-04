"""
PINN Final Project
EN 553.481/681 Numerical Analysis
"""
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_pinn(model, loss_fn, epochs, lr=1e-3, log_every=2000):
    """Train a PINN model.
    Returns: (loss_history, wall_clock_time_seconds)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn(model)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if epoch % log_every == 0:
            print(f"  Epoch {epoch}/{epochs}, Loss = {loss.item():.6e}")
    wall_time = time.time() - t_start
    print(f"  Training time: {wall_time:.1f}s")
    return loss_history, wall_time


def plot_loss_curve(loss_history, title="Training Loss"):
    plt.figure(figsize=(6, 4))
    plt.semilogy(loss_history)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(title); plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_ode_comparison(model, exact_fn, t_range=(0, 5), label="PINN"):
    t = torch.linspace(*t_range, 1000, device=device).unsqueeze(1)
    with torch.no_grad():
        u_pred = model(t).cpu().numpy().flatten()
    t_np = t.cpu().numpy().flatten()
    u_ex = exact_fn(t_np)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t_np, u_ex, 'k-', lw=2, label='Exact')
    axes[0].plot(t_np, u_pred, 'r--', lw=1.5, label=label)
    axes[0].set_xlabel('t'); axes[0].set_ylabel('u(t)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'{label} vs Exact')

    err = np.abs(u_pred - u_ex)
    axes[1].plot(t_np, err, 'b-')
    axes[1].set_xlabel('t'); axes[1].set_ylabel('|error|')
    axes[1].set_title(f'Pointwise Error (max = {err.max():.4e})')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    print(f"  Max absolute error: {err.max():.6e}")
    return err.max()


def plot_heat_comparison(model, exact_fn, label="PINN"):
    """Plot PINN vs exact for heat eq. Returns relative L2 error."""
    Ntest = 100
    x = np.linspace(0, 1, Ntest)
    t = np.linspace(0, 0.5, Ntest)
    X, T = np.meshgrid(x, t)
    xt = np.column_stack([X.ravel(), T.ravel()])
    xt_t = torch.tensor(xt, dtype=torch.float32, device=device)
    with torch.no_grad():
        u_pred = model(xt_t).cpu().numpy().reshape(Ntest, Ntest)
    u_ex = exact_fn(X, T)
    err = np.abs(u_pred - u_ex)
    rel_l2 = np.sqrt(np.sum((u_pred - u_ex)**2)) / np.sqrt(np.sum(u_ex**2))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    c0 = axes[0].pcolormesh(X, T, u_pred, shading='auto', cmap='viridis')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('t')
    axes[0].set_title(f'{label} Prediction'); plt.colorbar(c0, ax=axes[0])
    c1 = axes[1].pcolormesh(X, T, u_ex, shading='auto', cmap='viridis')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('t')
    axes[1].set_title('Exact Solution'); plt.colorbar(c1, ax=axes[1])
    c2 = axes[2].pcolormesh(X, T, err, shading='auto', cmap='hot')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('t')
    axes[2].set_title(f'|Error| (rel L2 = {rel_l2:.4e})'); plt.colorbar(c2, ax=axes[2])
    plt.tight_layout()
    print(f"  Relative L2 error: {rel_l2:.6e}")
    return rel_l2


def compute_loss_ode_ad(model, Nr = 500):
    """PINN loss for ODE using AUTOGRAD.
    ODE: du/dt = -5u + 5cos(t) - sin(t),  u(0) = 0
    """
    # collocation points
    t_r = torch.rand(Nr, 1, device=device) * 5.0
    t_r.requires_grad_(True)

    u = model(t_r)

    # du/dt via autograd
    du_dt = torch.autograd.grad(
        outputs=u,
        inputs=t_r,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # residual: du/dt = -5u + 5cos(t) - sin(t)
    residual = du_dt + 5.0 * u - 5.0 * torch.cos(t_r) + torch.sin(t_r)
    Lr = torch.mean(residual**2)

    # initial condition u(0)=0
    t0 = torch.zeros(1, 1, device=device)
    u0 = model(t0)
    Lic = torch.mean(u0**2)

    return Lr + 50.0 * Lic


def compute_loss_ode_fdm(model, epsilon=1e-3, Nr = 500):
    """PINN loss for ODE using FINITE DIFFERENCES.
    Same ODE as above. Instead of autograd, approximate du/dt
    using the central difference formula:
        du/dt(t) ≈ (u(t + epsilon) - u(t - epsilon)) / (2 * epsilon)
    """
    # collocation points (no autograd needed for FDM derivative)
    t_r = torch.rand(Nr, 1, device=device)

    # central difference points
    t_plus = t_r + epsilon
    t_minus = t_r - epsilon

    # evaluate model (only forward passes require gradients)
    u = model(t_r)
    u_plus = model(t_plus)
    u_minus = model(t_minus)

    du_dt = (u_plus - u_minus) / (2.0 * epsilon)

    # residual: same ODE
    residual = du_dt + 5.0 * u - 5.0 * torch.cos(t_r) + torch.sin(t_r)
    Lr = torch.mean(residual**2)

    # initial condition
    t0 = torch.zeros(1, 1, device=device)
    u0 = model(t0)
    Lic = torch.mean(u0**2)

    return Lr + 50.0 * Lic


def compute_loss_heat_ad(model, Nr=10000):
    """PINN loss for heat equation using AUTOGRAD.
    PDE: u_t = 0.01 * u_xx  on (0,1) x (0, 0.5]
    IC:  u(x, 0) = sin(pi*x) + 0.5*sin(3*pi*x)
    BC:  u(0, t) = u(1, t) = 0
    """
    Nic = 200
    Nbc = 200
    nu = 0.01

    # interior (residual) points
    x_r = torch.rand(Nr, 1, device=device)
    t_r = torch.rand(Nr, 1, device=device) * 0.5

    x_r.requires_grad_(True)
    t_r.requires_grad_(True)

    xt_r = torch.cat([x_r, t_r], dim=1)
    u = model(xt_r)

    # u_t
    u_t = torch.autograd.grad(
        u, t_r,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # u_x
    u_x = torch.autograd.grad(
        u, x_r,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # u_xx
    u_xx = torch.autograd.grad(
        u_x, x_r,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    # residual
    residual = u_t - nu * u_xx
    Lr = torch.mean(residual**2)

    # initial condition
    x_ic = torch.rand(Nic, 1, device=device)
    t_ic = torch.zeros(Nic, 1, device=device)

    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    u_ic_pred = model(xt_ic)

    u_ic_true = torch.sin(np.pi * x_ic) + 0.5 * torch.sin(3 * np.pi * x_ic)
    Lic = torch.mean((u_ic_pred - u_ic_true)**2)

    # boundary conditions
    t_bc = torch.rand(Nbc, 1, device=device) * 0.5

    x0 = torch.zeros(Nbc, 1, device=device)
    x1 = torch.ones(Nbc, 1, device=device)

    xt_bc0 = torch.cat([x0, t_bc], dim=1)
    xt_bc1 = torch.cat([x1, t_bc], dim=1)

    u_bc0 = model(xt_bc0)
    u_bc1 = model(xt_bc1)

    Lbc = torch.mean(u_bc0**2) + torch.mean(u_bc1**2)

    return Lr + 20.0 * Lic + 20.0 * Lbc


def compute_loss_heat_fdm(model, epsilon=1e-3, Nr=10000):
    """PINN loss for heat equation using FINITE DIFFERENCES.
    Same PDE, IC, BC as above. Approximate derivatives:
        u_t(x,t)  ≈ (u(x, t+eps) - u(x, t-eps)) / (2*eps)
        u_xx(x,t) ≈ (u(x+eps, t) - 2*u(x,t) + u(x-eps, t)) / eps^2
    """
    Nic = 200
    Nbc = 200
    nu = 0.01

    # interior points
    x_r = torch.rand(Nr, 1, device=device)
    t_r = torch.rand(Nr, 1, device=device) * 0.5

    # finite-difference shifts
    t_plus = t_r + epsilon
    t_minus = t_r - epsilon

    x_plus = x_r + epsilon
    x_minus = x_r - epsilon

    # PDE evaluations
    u = model(torch.cat([x_r, t_r], dim=1))
    u_t = (model(torch.cat([x_r, t_plus], dim=1)) -
           model(torch.cat([x_r, t_minus], dim=1))) / (2 * epsilon)

    u_xx = (model(torch.cat([x_plus, t_r], dim=1))
            - 2 * u
            + model(torch.cat([x_minus, t_r], dim=1))) / (epsilon**2)

    residual = u_t - nu * u_xx
    Lr = torch.mean(residual**2)

    # initial condition
    x_ic = torch.rand(Nic, 1, device=device)
    t_ic = torch.zeros(Nic, 1, device=device)

    u_ic_pred = model(torch.cat([x_ic, t_ic], dim=1))
    u_ic_true = torch.sin(np.pi * x_ic) + 0.5 * torch.sin(3 * np.pi * x_ic)

    Lic = torch.mean((u_ic_pred - u_ic_true)**2)

    # boundary conditions
    t_bc = torch.rand(Nbc, 1, device=device) * 0.5

    x0 = torch.zeros(Nbc, 1, device=device)
    x1 = torch.ones(Nbc, 1, device=device)

    u_bc0 = model(torch.cat([x0, t_bc], dim=1))
    u_bc1 = model(torch.cat([x1, t_bc], dim=1))

    Lbc = torch.mean(u_bc0**2) + torch.mean(u_bc1**2)

    return Lr + 20.0 * Lic + 20.0 * Lbc


if __name__ == "__main__":
    def ode_exact(t):
        return np.cos(t) - np.exp(-5.0 * t)
    nu = 0.01
    def heat_exact(X, T):
        return (
            np.exp(-0.01 * np.pi**2 * T) * np.sin(np.pi * X)
            + 0.5 * np.exp(-9 * 0.01 * np.pi**2 * T) * np.sin(3 * np.pi * X)
        )
        

    # --- Problem 1.1 Part A: Forward Euler ---
    print("=" * 50)
    print("Problem 1.1 Part A: Forward Euler")
    print("=" * 50)
    
    # step size and grid
    h = 0.01
    t = np.arange(0, 5 + h, h)

    # initialize solution
    u = np.zeros_like(t)
    u[0] = 0

    # start time
    t0 = time.time()
    
    # Forward Euler
    for n in range(len(t) - 1):
        u[n+1] = u[n] + h * (-5*u[n] + 5*np.cos(t[n]) - np.sin(t[n]))
    
    # end time
    euler_time = time.time() - t0

    # exact solution
    u_exact = ode_exact(t)
    
    # calculate error
    ode_euler_error = np.max(np.abs(u - u_exact))

    # plot
    plt.plot(t, u, label="Forward Euler", linestyle="--")
    plt.plot(t, u_exact, label="Exact solution")
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.legend()
    plt.title("Forward Euler vs Exact Solution")
    plt.show()
    
    
    
    # --- Problem 1.1 Part B: 4th-order Runge-Kutta ---
    print("=" * 50)
    print("Problem 1.1 Part B: 4th-order Runge-Kutta")
    print("=" * 50)
    
    # ODE definition
    def f(t, u):
        return -5*u + 5*np.cos(t) - np.sin(t)

    # step size and grid
    h = 0.01
    t = np.arange(0, 5 + h, h)

    # solution array
    u = np.zeros_like(t)
    u[0] = 0
    
    # start time
    t0 = time.time()

    # RK4 method
    for n in range(len(t) - 1):
        k1 = f(t[n], u[n])
        k2 = f(t[n] + h/2, u[n] + h*k1/2)
        k3 = f(t[n] + h/2, u[n] + h*k2/2)
        k4 = f(t[n] + h, u[n] + h*k3)
        
        u[n+1] = u[n] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    # end time
    rk4_time = time.time() - t0
    
    # exact solution
    u_exact = ode_exact(t)

    # calculate error
    ode_rk4_error = np.max(np.abs(u - u_exact))

    # plot
    plt.plot(t, u, "--", label="RK4")
    plt.plot(t, u_exact, label="Exact")
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.legend()
    plt.title("RK4 vs Exact Solution")
    plt.show()
    
    

    # --- Problem 1.1 Part C: Forward Euler vs RK4 ---
    print("=" * 50)
    print("Problem 1.1 Part C: Forward Euler vs RK4")
    print("=" * 50)
    
    def f(t, u):
        return -5*u + 5*np.cos(t) - np.sin(t)

    # Forward Euler method
    def forward_euler(h):
        t = np.arange(0, 5 + h, h)
        u = np.zeros_like(t)
        u[0] = 0
        
        for n in range(len(t) - 1):
            u[n+1] = u[n] + h * f(t[n], u[n])
        
        return t, u

    # RK4 method
    def rk4(h):
        t = np.arange(0, 5 + h, h)
        u = np.zeros_like(t)
        u[0] = 0
        
        for n in range(len(t) - 1):
            k1 = f(t[n], u[n])
            k2 = f(t[n] + h/2, u[n] + h*k1/2)
            k3 = f(t[n] + h/2, u[n] + h*k2/2)
            k4 = f(t[n] + h, u[n] + h*k3)
            u[n+1] = u[n] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        
        return t, u

    # compute max error
    def max_error(t, u):
        return np.max(np.abs(u - ode_exact(t)))

    hs = [0.01, 0.005, 0.001]

    fe_errors = []
    rk_errors = []

    # compute errors
    for h in hs:
        t_fe, u_fe = forward_euler(h)
        t_rk, u_rk = rk4(h)
        
        fe_errors.append(max_error(t_fe, u_fe))
        rk_errors.append(max_error(t_rk, u_rk))

    # compute observed order
    def order(e1, e2, h1, h2):
        return np.log(e1/e2) / np.log(h1/h2)

    def make_table(hs, errors):
        orders = ["-"] + [order(errors[i-1], errors[i], hs[i-1], hs[i]) for i in range(1, len(hs))]
        return pd.DataFrame({"h": hs, "Error": errors, "Order": orders})

    fe_df = make_table(hs, fe_errors)
    rk_df = make_table(hs, rk_errors)

    print("Forward Euler")
    print(fe_df.to_string(index=False))
    print("\nRK4")
    print(rk_df.to_string(index=False))

    
        
    # --- Problem 1.2: ODE with AD ---
    print("=" * 50)
    print("Problem 1.2: ODE with AD")
    print("=" * 50)
    model = PINN(input_dim=1, hidden_dim=64, num_layers=4).to(device)

    loss_history, train_time = train_pinn(
        model,
        compute_loss_ode_ad,
        epochs=10000,
        lr=1e-3,
        log_every=2000
    )

    plot_loss_curve(loss_history, title="ODE PINN (AD) Loss")

    # plot vs exact + error
    max_err = plot_ode_comparison(
        model,
        ode_exact,
        t_range=(0, 5),
        label="PINN (AD)"
    )

    # dense evaluation (1000 points)
    t_test = torch.linspace(0, 5, 1000, device=device).unsqueeze(1)

    with torch.no_grad():
        u_pred = model(t_test).cpu().numpy().flatten()

    t_np = t_test.cpu().numpy().flatten()
    u_true = ode_exact(t_np)

    ode_ad_error = np.max(np.abs(u_pred - u_true))
    ode_ad_time = train_time

    plt.show()
    
    

    # --- Problem 1.3: ODE with FDM ---
    print("\n" + "=" * 50)
    print("Problem 1.3: ODE with FDM")
    print("=" * 50)
    
    # model (same architecture as AD case)
    model_fdm = PINN(input_dim=1, hidden_dim=64, num_layers=4).to(device)

    # training
    loss_history_fdm, train_time_fdm = train_pinn(
        model_fdm,
        lambda m: compute_loss_ode_fdm(m, epsilon=1e-3),
        epochs=10000,
        lr=1e-3,
        log_every=2000
    )

    # loss curve
    plot_loss_curve(loss_history_fdm, title="ODE PINN (FDM) Loss")

    # comparison plot + error
    max_err_fdm = plot_ode_comparison(
        model_fdm,
        ode_exact,
        t_range=(0, 5),
        label="PINN (FDM)"
    )

    # dense evaluation (1000 test points)
    t_test = torch.linspace(0, 5, 1000, device=device).unsqueeze(1)

    with torch.no_grad():
        u_pred = model_fdm(t_test).cpu().numpy().flatten()

    t_np = t_test.cpu().numpy().flatten()
    u_true = ode_exact(t_np)

    ode_fdm_error = np.max(np.abs(u_pred - u_true))
    ode_fdm_time = train_time_fdm

    plt.show()

    

    # --- Problem 1.4 Part A: ODE with AD vs FDM ---
    print("\n" + "=" * 50)
    print("Problem 1.4 Part A: ODE with AD vs FDM")
    print("=" * 50)
    data = [
        {
            "Method": "AD-PINN",
            "Final Loss": loss_history[-1],
            "Max Error": ode_ad_error,
            "Time (s)": ode_ad_time
        },
        {
            "Method": "FDM-PINN",
            "Final Loss": loss_history_fdm[-1],
            "Max Error": ode_fdm_error,
            "Time (s)": ode_fdm_time
        }
    ]

    df = pd.DataFrame(data)
    print(df)

    

    # --- Problem 1.4 Part B: ODE FDM for various epsilon ---

    print("=" * 50)
    print("Problem 1.4 Part B: ODE FDM for various epsilon")
    print("=" * 50)
    
    eps_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    errors = []

    for eps in eps_list:
        print(f"Training FDM-PINN with epsilon = {eps}")

        model = PINN(input_dim=1, hidden_dim=64, num_layers=4).to(device)

        loss_history, _ = train_pinn(
            model,
            lambda m: compute_loss_ode_fdm(m, epsilon=eps),
            epochs=10000,
            lr=1e-3,
            log_every=2000
        )

        # evaluate max error on dense grid
        t_test = torch.linspace(0, 5, 1000, device=device).unsqueeze(1)

        with torch.no_grad():
            u_pred = model(t_test).cpu().numpy().flatten()

        t_np = t_test.cpu().numpy().flatten()
        u_true = ode_exact(t_np)

        max_err = np.max(np.abs(u_pred - u_true))
        errors.append(max_err)

    # plot error vs epsilon
    plt.figure()
    plt.loglog(eps_list, errors, marker='o')
    plt.xlabel("epsilon")
    plt.ylabel("max absolute error")
    plt.title("FDM-PINN Error vs epsilon")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

    
    
    
    # --- Problem 2.1: Finite-difference heatmap ---
    print("\n" + "=" * 50)
    print("Problem 2.1: Finite-difference heatmap")
    print("=" * 50)
    
    # parameters
    nu = 0.01
    dx = 1/64

    # stability condition: r <= 1/2
    dt = 0.5 * dx**2 / nu   # choose largest stable dt
    r = nu * dt / dx**2

    print(f"dx = {dx}")
    print(f"dt = {dt}")
    print(f"r  = {r}")

    # grid
    x = np.arange(0, 1 + dx, dx)
    Nx = len(x)

    T = 0.5
    Nt = int(T / dt)
    t = np.linspace(0, T, Nt + 1)

    # initial condition
    u_num = np.zeros((Nt + 1, Nx))
    u_num[0, :] = np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)

    # enforce boundary conditions
    u_num[:, 0] = 0
    u_num[:, -1] = 0
    
    # start time
    t0 = time.time()

    # Forward Euler scheme
    for n in range(Nt):
        for i in range(1, Nx - 1):
            u_num[n+1, i] = (
                u_num[n, i]
                + r * (u_num[n, i+1] - 2*u_num[n, i] + u_num[n, i-1])
            )

    # end time
    heat_fd_time = time.time() - t0

    # compute L2 error at t = 0.5
    u_ex_final = heat_exact(x, T)
    u_num_final = u_num[-1, :]

    L2_error = np.sqrt(np.sum((u_num_final - u_ex_final)**2) * dx)
    print(f"L2 error at t=0.5: {L2_error:.6e}")
    
    heat_fd_error = L2_error

    # heatmap
    X, T_grid = np.meshgrid(x, t)

    plt.figure(figsize=(6, 4))
    plt.pcolormesh(X, T_grid, u_num, shading='auto')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Numerical Solution Heatmap')
    plt.colorbar(label='u(x,t)')
    plt.tight_layout()
    plt.show()
    
    
    
    # --- Problem 2.2: Heat with AD ---
    print("\n" + "=" * 50)
    print("Problem 2.2: Heat with AD")
    print("=" * 50)
    
    model_heat_ad = PINN(input_dim=2, hidden_dim=64, num_layers=4).to(device)

    loss_history_heat_ad, train_time_heat_ad = train_pinn(
        model_heat_ad,
        compute_loss_heat_ad,
        epochs=20000,
        lr=1e-3,
        log_every=4000
    )

    # loss curve
    plot_loss_curve(loss_history_heat_ad, title="Heat PINN (AD) Loss")

    # heatmap + error + relative L2
    rel_l2_error = plot_heat_comparison(
        model_heat_ad,
        heat_exact,
        label="PINN (AD)"
    )
    
    heat_ad_time = train_time_heat_ad
    heat_ad_error = rel_l2_error

    plt.show()

    

    # --- Problem 2.3: Heat with FDM ---
    print("\n" + "=" * 50)
    print("Problem 2.3: Heat with FDM")
    print("=" * 50)
    
    model_heat_fdm = PINN(input_dim=2, hidden_dim=64, num_layers=4).to(device)

    loss_history_heat_fdm, train_time_heat_fdm = train_pinn(
        model_heat_fdm,
        lambda m: compute_loss_heat_fdm(m, epsilon=1e-3),
        epochs=20000,
        lr=1e-3,
        log_every=4000
    )

    # loss curve
    plot_loss_curve(loss_history_heat_fdm, title="Heat PINN (FDM) Loss")

    # evaluation
    rel_l2_error_fdm = plot_heat_comparison(
        model_heat_fdm,
        heat_exact,
        label="PINN (FDM)"
    )
    
    heat_fdm_time = train_time_heat_fdm
    heat_fdm_error = rel_l2_error_fdm
    
    plt.show()
    
    

    # --- Problem 2.4 Part A: Heat with AD vs FDM ---
    print("\n" + "=" * 50)
    print("Problem 2.4 Part A: Heat with AD vs FDM")
    print("=" * 50)
    heat_results = pd.DataFrame([
        {
            "Method": "AD-PINN",
            "Final Loss": loss_history_heat_ad[-1],
            "Rel L2 Error": rel_l2_error,
            "Time (s)": train_time_heat_ad
        },
        {
            "Method": "FDM-PINN",
            "Final Loss": loss_history_heat_fdm[-1],
            "Rel L2 Error": rel_l2_error_fdm,
            "Time (s)": train_time_heat_fdm
        }
    ])
    print(heat_results.to_string(index=False))
    
    

    # --- Problem 2.4 Part B: Heat with FDM for various epsilon ---
    print("\n" + "=" * 50)
    print("Problem 2.4 Part B: Heat with FDM for various epsilon")
    print("=" * 50)
    
    eps_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    rel_errors = []
    
    for eps in eps_list:
        print(f"Training FDM-PINN with epsilon = {eps}")

        model = PINN(input_dim=2, hidden_dim=64, num_layers=4).to(device)

        loss_history, _ = train_pinn(
            model,
            lambda m: compute_loss_heat_fdm(m, epsilon=eps),
            epochs=5000,
            lr=1e-3,
            log_every=2000
        )

        # evaluate
        N = 100
        x = np.linspace(0, 1, N)
        t = np.linspace(0, 0.5, N)
        X, T = np.meshgrid(x, t)

        xt = torch.tensor(np.column_stack([X.ravel(), T.ravel()]),
                        dtype=torch.float32, device=device)

        with torch.no_grad():
            u_pred = model(xt).cpu().numpy().reshape(N, N)

        u_true = heat_exact(X, T)

        rel_l2 = np.sqrt(np.sum((u_pred - u_true)**2)) / np.sqrt(np.sum(u_true**2))
        rel_errors.append(rel_l2)

    # plot
    plt.figure()
    plt.loglog(eps_list, rel_errors, marker='o')
    plt.xlabel("epsilon")
    plt.ylabel("relative L2 error")
    plt.title("Heat FDM-PINN: error vs epsilon")
    plt.grid(True, which="both")
    plt.show()
    
    

    # --- Problem 2.4 Part C: Unstable Forward Euler ---
    print("\n" + "=" * 50)
    print("Problem 2.4 Part C: Unstable Forward Euler")
    print("=" * 50)
    
    # unstable CFL choice
    nu = 0.01
    dx = 1/64
    r = 0.6  # violates stability condition r <= 1/2
    dt = r * dx**2 / nu

    print("r =", r, "dt =", dt)

    # grid
    x = np.linspace(0, 1, 65)
    t = np.arange(0, 0.5 + dt, dt)

    u = np.zeros((len(t), len(x)))

    # initial condition
    u[0] = np.sin(np.pi * x) + 0.5 * np.sin(3*np.pi*x)

    # explicit scheme (unstable case)
    for n in range(len(t)-1):
        for i in range(1, len(x)-1):
            u[n+1, i] = (
                u[n, i]
                + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
            )

    # plot heatmap
    X, T = np.meshgrid(x, t)
    plt.figure()
    plt.pcolormesh(X, T, u, shading='auto')
    plt.title("Unstable Forward Euler (r = 0.6)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.colorbar()
    plt.show()



    # --- Problem 3 Part A: Summary of all methods ---
    print("\n" + "=" * 50)
    print("Problem 3 Part A: Summary of all methods")
    print("=" * 50)

    summary = pd.DataFrame([
        # ODE methods
        {"Problem": "ODE", "Method": "Euler", "Error": ode_euler_error, "Time": euler_time},
        {"Problem": "ODE", "Method": "RK4", "Error": ode_rk4_error, "Time": rk4_time},
        {"Problem": "ODE", "Method": "AD-PINN", "Error": ode_ad_error, "Time": ode_ad_time},
        {"Problem": "ODE", "Method": "FDM-PINN", "Error": ode_fdm_error, "Time": ode_fdm_time},

        # Heat methods
        {"Problem": "Heat", "Method": "FD scheme", "Error": heat_fd_error, "Time": heat_fd_time},
        {"Problem": "Heat", "Method": "AD-PINN", "Error": heat_ad_error, "Time": heat_ad_time},
        {"Problem": "Heat", "Method": "FDM-PINN", "Error": heat_fdm_error, "Time": heat_fdm_time},
    ])
    print(summary.to_string(index=False))

    
    
    # --- Problem 3 Part B: PINNs with various Nr ---
    print("\n" + "=" * 50)
    print("Problem 3 Part B: PINNs with various Nr")
    print("=" * 50)
    
    ode_nr_list = [100, 500, 2000, 10000]

    ode_ad_errors = []
    ode_fdm_errors = []

    print("ODE PINNs:")
    for Nr in ode_nr_list:
        print(f"Nr = {Nr}")

        # AD
        model = PINN(1, 64, 4).to(device)
        train_pinn(model,
            lambda m: compute_loss_ode_ad(m, Nr=Nr),
            epochs=5000, lr=1e-3)

        t_test = torch.linspace(0,5,1000,device=device).unsqueeze(1)
        with torch.no_grad():
            pred = model(t_test).cpu().numpy().flatten()

        true = ode_exact(t_test.cpu().numpy().flatten())
        ode_ad_errors.append(np.max(np.abs(pred-true)))

        # FDM
        model = PINN(1, 64, 4).to(device)
        train_pinn(model,
            lambda m: compute_loss_ode_fdm(m, Nr=Nr),
            epochs=5000, lr=1e-3)

        with torch.no_grad():
            pred = model(t_test).cpu().numpy().flatten()

        ode_fdm_errors.append(np.max(np.abs(pred-true)))
    
    plt.figure()
    plt.loglog(ode_nr_list, ode_ad_errors, marker='o', label='AD-PINN')
    plt.loglog(ode_nr_list, ode_fdm_errors, marker='o', label='FDM-PINN')
    plt.xlabel('Nr')
    plt.ylabel('max error')
    plt.title('ODE: Error vs Nr')
    plt.legend()
    plt.grid(True, which='both')
    plt.show()
    
    heat_nr_list = [500, 2000, 10000, 50000]

    heat_ad_errors = []
    heat_fdm_errors = []

    print("\nHeat PINNs:")
    for Nr in heat_nr_list:
        print(f"Nr = {Nr}")

        # AD
        model = PINN(2, 64, 4).to(device)
        train_pinn(model,
            lambda m: compute_loss_heat_ad(m, Nr=Nr),
            epochs=10000, lr=1e-3)

        rel_l2 = plot_heat_comparison(model, heat_exact)
        heat_ad_errors.append(rel_l2)

        # FDM
        model = PINN(2, 64, 4).to(device)
        train_pinn(model,
            lambda m: compute_loss_heat_fdm(m, Nr=Nr),
            epochs=10000, lr=1e-3)

        rel_l2 = plot_heat_comparison(model, heat_exact)
        heat_fdm_errors.append(rel_l2)
    
    plt.figure()
    plt.loglog(heat_nr_list, heat_ad_errors, marker='o', label='AD-PINN')
    plt.loglog(heat_nr_list, heat_fdm_errors, marker='o', label='FDM-PINN')
    plt.xlabel('Nr')
    plt.ylabel('relative L2 error')
    plt.title('Heat Equation: Error vs Nr')
    plt.legend()
    plt.grid(True, which='both')
    plt.show()

    
    
    # --- Problem 3 Part C: PINNs with various network sizes ---
    print("\n" + "=" * 50)
    print("Problem 3 Part C: PINNs with various network sizes")
    print("=" * 50)
    
    configs = {
        "small": (2, 16),
        "base": (4, 64),
        "large": (5, 64)
    }
    
    ode_results = []

    t_test = torch.linspace(0, 5, 1000, device=device).unsqueeze(1)
    t_np = t_test.cpu().numpy().flatten()
    u_true = ode_exact(t_np)

    for name, (layers, width) in configs.items():
        print(f"ODE | {name}")

        # AD
        model = PINN(1, width, layers).to(device)
        loss_history, t_ad = train_pinn(model, compute_loss_ode_ad, epochs=10000)

        with torch.no_grad():
            pred = model(t_test).cpu().numpy().flatten()

        err_ad = np.max(np.abs(pred - u_true))

        # FDM
        model = PINN(1, width, layers).to(device)
        loss_history, t_fdm = train_pinn(
            model,
            lambda m: compute_loss_ode_fdm(m, epsilon=1e-3),
            epochs=10000
        )

        with torch.no_grad():
            pred = model(t_test).cpu().numpy().flatten()

        err_fdm = np.max(np.abs(pred - u_true))

        ode_results.append({
            "Config": name,
            "AD Error": err_ad,
            "AD Time": t_ad,
            "FDM Error": err_fdm,
            "FDM Time": t_fdm
        })
    
    heat_results = []

    for name, (layers, width) in configs.items():
        print(f"Heat | {name}")

        # AD
        model = PINN(2, width, layers).to(device)
        loss_history, t_ad = train_pinn(model, compute_loss_heat_ad, epochs=20000)

        err_ad = plot_heat_comparison(model, heat_exact)

        # FDM
        model = PINN(2, width, layers).to(device)
        loss_history, t_fdm = train_pinn(
            model,
            lambda m: compute_loss_heat_fdm(m, epsilon=1e-3),
            epochs=20000
        )

        err_fdm = plot_heat_comparison(model, heat_exact)

        heat_results.append({
            "Config": name,
            "AD Error": err_ad,
            "AD Time": t_ad,
            "FDM Error": err_fdm,
            "FDM Time": t_fdm
        })
    
    print("\nODE Results:")
    print(pd.DataFrame(ode_results).to_string(index=False))

    print("\nHeat Results:")
    print(pd.DataFrame(heat_results).to_string(index=False))
    
    # plots
    labels = list(configs.keys())

    # ODE
    plt.figure()
    plt.plot(labels, [r["AD Error"] for r in ode_results], marker='o', label='AD')
    plt.plot(labels, [r["FDM Error"] for r in ode_results], marker='o', label='FDM')
    plt.ylabel("Max Error")
    plt.title("ODE: Error vs Network Size")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure()
    plt.plot(labels, [r["AD Time"] for r in ode_results], marker='o', label='AD')
    plt.plot(labels, [r["FDM Time"] for r in ode_results], marker='o', label='FDM')
    plt.ylabel("Time (s)")
    plt.title("ODE: Training Time vs Network Size")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Heat
    plt.figure()
    plt.plot(labels, [r["AD Error"] for r in heat_results], marker='o', label='AD')
    plt.plot(labels, [r["FDM Error"] for r in heat_results], marker='o', label='FDM')
    plt.ylabel("Relative L2 Error")
    plt.title("Heat: Error vs Network Size")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure()
    plt.plot(labels, [r["AD Time"] for r in heat_results], marker='o', label='AD')
    plt.plot(labels, [r["FDM Time"] for r in heat_results], marker='o', label='FDM')
    plt.ylabel("Time (s)")
    plt.title("Heat: Training Time vs Network Size")
    plt.legend()
    plt.grid(True)
    plt.show()

    
    
    # --- Bonus: Inverse problem ---
    print("\n" + "=" * 50)
    print("Bonus: Inverse problem")
    print("=" * 50)
    
    # modified model that includes nu as a trainable parameter
    class PINN_with_nu(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers):
            super().__init__()
            self.net = PINN(input_dim, hidden_dim, num_layers).net
            
            # trainable diffusion coefficient
            self.nu = nn.Parameter(torch.tensor([0.02], dtype=torch.float32))  
            # initialize slightly wrong on purpose

        def forward(self, x):
            return self.net(x)
    
    # generate noisy data
    # exact solution with nu parameter
    def heat_exact(x, t, nu=0.01):
        return (
            np.exp(-nu*np.pi**2*t)*np.sin(np.pi*x)
            + 0.5*np.exp(-9*nu*np.pi**2*t)*np.sin(3*np.pi*x)
        )

    # sample noisy observations
    Ndata = 50
    sigma = 0.01

    x_data = np.random.rand(Ndata)
    t_data = np.random.rand(Ndata) * 0.5

    u_data = heat_exact(x_data, t_data) + sigma * np.random.randn(Ndata)

    # convert to torch
    xt_data = torch.tensor(np.column_stack([x_data, t_data]),
                        dtype=torch.float32, device=device)
    u_data_t = torch.tensor(u_data.reshape(-1,1),
                            dtype=torch.float32, device=device)
    
    # AD model
    def compute_loss_heat_inverse_ad(model):
        Nr = 10000
        Nic = 200
        Nbc = 200

        nu = model.nu

        # interior points
        x_r = torch.rand(Nr,1,device=device, requires_grad=True)
        t_r = torch.rand(Nr,1,device=device, requires_grad=True)*0.5

        xt = torch.cat([x_r, t_r], dim=1)
        u = model(xt)

        u_t = torch.autograd.grad(u, t_r,
                                grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]

        u_x = torch.autograd.grad(u, x_r,
                                grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]

        u_xx = torch.autograd.grad(u_x, x_r,
                                grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]

        Lr = torch.mean((u_t - nu * u_xx)**2)

        # IC
        x_ic = torch.rand(Nic,1,device=device)
        t_ic = torch.zeros(Nic,1,device=device)
        u_ic = model(torch.cat([x_ic,t_ic],dim=1))
        u_true = torch.sin(np.pi*x_ic) + 0.5*torch.sin(3*np.pi*x_ic)

        Lic = torch.mean((u_ic - u_true)**2)

        # BC
        t_bc = torch.rand(Nbc,1,device=device)*0.5
        u0 = model(torch.cat([torch.zeros_like(t_bc), t_bc], dim=1))
        u1 = model(torch.cat([torch.ones_like(t_bc), t_bc], dim=1))

        Lbc = torch.mean(u0**2) + torch.mean(u1**2)

        # DATA LOSS
        u_pred_data = model(xt_data)
        Ldata = torch.mean((u_pred_data - u_data_t)**2)

        return Lr + 20*Lic + 20*Lbc + 50*Ldata
    
    # FDM model
    def compute_loss_heat_inverse_fdm(model, epsilon=1e-3):
        Nr = 10000
        Nic = 200
        Nbc = 200

        nu = model.nu

        x_r = torch.rand(Nr,1,device=device)
        t_r = torch.rand(Nr,1,device=device)*0.5

        u = model(torch.cat([x_r, t_r], dim=1))

        u_t = (model(torch.cat([x_r, t_r+epsilon], dim=1)) -
            model(torch.cat([x_r, t_r-epsilon], dim=1))) / (2*epsilon)

        u_xx = (model(torch.cat([x_r+epsilon, t_r], dim=1))
                - 2*u
                + model(torch.cat([x_r-epsilon, t_r], dim=1))) / (epsilon**2)

        Lr = torch.mean((u_t - nu*u_xx)**2)

        # IC + BC same as above
        x_ic = torch.rand(Nic,1,device=device)
        t_ic = torch.zeros(Nic,1,device=device)
        u_ic = model(torch.cat([x_ic,t_ic],dim=1))
        u_true = torch.sin(np.pi*x_ic) + 0.5*torch.sin(3*np.pi*x_ic)

        Lic = torch.mean((u_ic - u_true)**2)

        t_bc = torch.rand(Nbc,1,device=device)*0.5
        u0 = model(torch.cat([torch.zeros_like(t_bc), t_bc], dim=1))
        u1 = model(torch.cat([torch.ones_like(t_bc), t_bc], dim=1))

        Lbc = torch.mean(u0**2) + torch.mean(u1**2)

        # DATA LOSS
        u_pred_data = model(xt_data)
        Ldata = torch.mean((u_pred_data - u_data_t)**2)

        return Lr + 20*Lic + 20*Lbc + 50*Ldata
    
    # training
    # AD
    print("AD-PINN:")
    model = PINN_with_nu(2, 64, 4).to(device)

    _, _ = train_pinn(model, compute_loss_heat_inverse_ad, epochs=20000)

    nu_recovered_ad = model.nu.item()

    # FDM
    print("FDM-PINN:")
    model_fdm = PINN_with_nu(2, 64, 4).to(device)

    _, _ = train_pinn(model_fdm,
                    lambda m: compute_loss_heat_inverse_fdm(m, 1e-3),
                    epochs=20000)

    nu_recovered_fdm = model_fdm.nu.item()

    nu_true = 0.01

    err_ad = abs(nu_recovered_ad - nu_true) / nu_true
    err_fdm = abs(nu_recovered_fdm - nu_true) / nu_true

    print("AD nu:", nu_recovered_ad, "rel error:", err_ad)
    print("FDM nu:", nu_recovered_fdm, "rel error:", err_fdm)
    
    

    print("\nDone! All plots saved.")
