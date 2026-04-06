import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

plt.close("all")

plt.rcParams.update({'text.usetex': False})
plt.rcParams.update({'image.cmap': 'viridis'})
plt.rcParams.update({
    'font.serif': [
        'Times New Roman', 'Times', 'DejaVu Serif',
        'Computer Modern Roman'
    ]
})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 9})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.close('all')


def train_svr():
    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    df = pd.read_excel("DataCoordinatesNoPackage.xlsx")

    # Inputs from cameras
    X = df[["CAM1_X", "CAM1_Y", "CAM2_X", "CAM2_Y"]].values

    # Outputs from OptiTrack
    Y = df[["opt_X", "opt_Y", "opt_Z"]].values

    # Original frame indices
    indices = np.arange(len(X))

    X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
        X, Y, indices, test_size=0.2, random_state=42
    )

    # --------------------------------------------------------
    # Hyperparameter grid
    # --------------------------------------------------------
    C_list = [1e3]
    gamma_list = [1]

    cv_mse = np.zeros((len(C_list), len(gamma_list)))
    test_mse_grid = np.zeros_like(cv_mse)

    best_cv_mse = np.inf
    best_cv_idx = None

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # --------------------------------------------------------
    # Grid search (select by CV)
    # --------------------------------------------------------
    for i, C in enumerate(C_list):
        for j, gamma in enumerate(gamma_list):

            base_model = SVR(
                kernel="rbf",
                C=C,
                gamma=gamma
            )

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("svr", MultiOutputRegressor(base_model))
            ])

            cv_errors = []
            for tr, va in kf.split(X_train):
                model.fit(X_train[tr], Y_train[tr])
                pred = model.predict(X_train[va])
                cv_errors.append(mean_squared_error(Y_train[va], pred))

            cv_mse[i, j] = np.mean(cv_errors)

            model.fit(X_train, Y_train)
            pred_test_tmp = model.predict(X_test)
            test_mse_grid[i, j] = mean_squared_error(Y_test, pred_test_tmp)

            if cv_mse[i, j] < best_cv_mse:
                best_cv_mse = cv_mse[i, j]
                best_cv_idx = (i, j)

            print(f"C={C}, gamma={gamma:.1e} | "
                  f"CV MSE={cv_mse[i,j]:.4f}, Test MSE={test_mse_grid[i,j]:.4f}")

    # --------------------------------------------------------
    # Train final model with best CV hyperparameters
    # --------------------------------------------------------
    best_C = C_list[best_cv_idx[0]]
    best_gamma = gamma_list[best_cv_idx[1]]

    base_model = SVR(kernel="rbf", C=best_C, gamma=best_gamma)
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", MultiOutputRegressor(base_model))
    ])

    final_model.fit(X_train, Y_train)

    # Test prediction
    Y_test_pred = final_model.predict(X_test)
    final_test_mse = mean_squared_error(Y_test, Y_test_pred)

    # Full-data prediction for Figure 1
    Y_all_pred = final_model.predict(X)

    print("\n================ FINAL MODEL (CV-SELECTED) ================")
    print(f"Best C (CV)        : {best_C}")
    print(f"Best Gamma (CV)    : {best_gamma}")
    print(f"Best CV MSE        : {best_cv_mse:.4f}")
    print(f"Test MSE           : {final_test_mse:.4f}")

    # --------------------------------------------------------
    # Sort test set back to original frame order
    # --------------------------------------------------------
    order = np.argsort(idx_test)
    frames_test = idx_test[order]

    Y_test_sorted = Y_test[order]
    Y_test_pred_sorted = Y_test_pred[order]

    # Absolute error for each coordinate
    errors_test = np.abs(Y_test_pred_sorted - Y_test_sorted)

    # --------------------------------------------------------
    # Manual tick options
    # --------------------------------------------------------
    coord_yticks_list = [None, None, None]
    err_yticks_list = [None, None, None]

    coord_ylim_list = [None, None, None]
    err_ylim_list   = [None, None, None]

    # --------------------------------------------------------
    # Figure 1: 3D plot of all data
    # ground truth = line
    # prediction   = scatter
    # --------------------------------------------------------
    fig1 = plt.figure(figsize=(6.5, 5))
    ax3d = fig1.add_subplot(111, projection='3d')

    ax3d.plot(
        Y[:, 0], Y[:, 1], Y[:, 2],
        linestyle='-',
        linewidth=1.5,
        label='OptiTrack (ground truth)'
    )

    ax3d.scatter(
        Y_all_pred[:, 0], Y_all_pred[:, 1], Y_all_pred[:, 2],
        s=8,
        alpha=0.7,
        color='orange',
        label='predicted coordinates'
    )

    ax3d.set_xlabel("x (mm)")
    ax3d.set_ylabel("y (mm)")
    ax3d.set_zlabel("z (mm)")
    ax3d.set_title("3D OptiTrack Ground Truth vs Predicted Coordinates")
    ax3d.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Figure 2: test-only in original time/frame order
    # --------------------------------------------------------
    fig2, axes = plt.subplots(3, 1, figsize=(6.5, 5), sharex=True)

    coord_labels = ["x (mm)", "y (mm)", "z (mm)"]
    error_labels = ["|error| (mm)", "|error| (mm)", "|error| (mm)"]

    for k, ax in enumerate(axes):
        line_ref, = ax.plot(
            frames_test, Y_test_sorted[:, k],
            linestyle='-',
            linewidth=0.8,
            label="reference coordinates"
        )
        line_pred, = ax.plot(
            frames_test, Y_test_pred_sorted[:, k],
            linestyle='--',
            linewidth=0.8,
            label="predicted coordinates"
        )

        ax.set_ylabel(coord_labels[k])

        if coord_yticks_list[k] is not None:
            ax.set_yticks(coord_yticks_list[k])
        if coord_ylim_list[k] is not None:
            ax.set_ylim(*coord_ylim_list[k])

        ax.grid(True, which='both', linestyle='-', linewidth=0.5)

        ax_err = ax.twinx()
        line_err, = ax_err.plot(
            frames_test, errors_test[:, k],
            color='k',
            linewidth=0.4,
            label="absolute error"
        )
        ax_err.set_ylabel(error_labels[k])

        if err_yticks_list[k] is not None:
            ax_err.set_yticks(err_yticks_list[k])
        if err_ylim_list[k] is not None:
            ax_err.set_ylim(*err_ylim_list[k])

        if k == 0:
            lines = [line_ref, line_pred, line_err]
            labels = [
                "reference coordinates",
                "predicted coordinates",
                "absolute error"
            ]
            ax.legend(lines, labels, loc="upper right", ncol=3, frameon=True)

    axes[-1].set_xlabel("frames")
    plt.tight_layout(pad=0)
    plt.show()


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    train_svr()