import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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


def train_svr_opt_y():
    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    df = pd.read_excel("DataCoordinatesNoPackage.xlsx")   # replace with your actual file name

    X = df[["CAM1_X", "CAM1_Y", "CAM2_X", "CAM2_Y"]].values
    y = df["opt_Y_adjusted"].values

    # Each row is a timestep
    timesteps = np.arange(len(df))

    # --------------------------------------------------------
    # 80/20 train-test split while keeping timestep info
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test, ts_train, ts_test = train_test_split(
        X, y, timesteps, test_size=0.2, random_state=42
    )

    # --------------------------------------------------------
    # SVR model
    # --------------------------------------------------------
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=1e5, gamma=1, epsilon=0.1))
    ])

    # Train on 80%
    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------
    # Predict on test set
    y_test_pred = model.predict(X_test)

    # Predict on full dataset for Figure 1
    y_all_pred = model.predict(X)

    # --------------------------------------------------------
    # Metrics on test set only
    # --------------------------------------------------------
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    print("================ TEST RESULTS ================")
    print(f"Test MSE  : {test_mse:.4f}")
    print(f"Test RMSE : {test_rmse:.4f}")
    print(f"Test R^2  : {test_r2:.4f}")

    # --------------------------------------------------------
    # Sort test data by original timestep
    # --------------------------------------------------------
    order = np.argsort(ts_test)

    ts_test_sorted = ts_test[order]
    y_test_sorted = y_test[order]
    y_test_pred_sorted = y_test_pred[order]

    abs_error = np.abs(y_test_pred_sorted - y_test_sorted)

    # --------------------------------------------------------
    # Figure 1: Entire dataset (train + test)
    # x-axis = timestep
    # --------------------------------------------------------
    plt.figure(figsize=(6.5, 3))
    plt.plot(
        timesteps, y,
        linestyle='-',
        linewidth=1.0,
        label='actual UAV height'
    )
    plt.plot(
        timesteps, y_all_pred,
        linestyle='--',
        linewidth=1.0,
        label='predicted UAV height'
    )
    plt.xlabel("time step")
    plt.ylabel("UAV height from ground (mm)")
    plt.title("SVR Prediction Over Entire Dataset")
    plt.grid(True, linestyle='-', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Figure 2: Test set only, with original timestep
    # --------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(6.5, 3))

    line1, = ax1.plot(
        ts_test_sorted, y_test_sorted,
        linestyle='-',
        linewidth=1.0,
        label='actual height from test data'
    )
    line2, = ax1.plot(
        ts_test_sorted, y_test_pred_sorted,
        linestyle='--',
        linewidth=1.0,
        label='predicted height from test data'
    )

    ax1.set_xlabel("time step")
    ax1.set_ylabel("UAV height from ground (mm)")
    ax1.grid(True, linestyle='-', linewidth=0.5)

    ax2 = ax1.twinx()
    line3, = ax2.plot(
        ts_test_sorted, abs_error,
        color='k',
        linewidth=0.8,
        label='absolute error'
    )
    ax2.set_ylabel("absolute error (mm)")

    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title("20% Test Dataset with Original Timesteps")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Save/print test results with timestep
    # --------------------------------------------------------
    results_df = pd.DataFrame({
        "timestep": ts_test_sorted,
        "actual_opt_Y_adjusted": y_test_sorted,
        "predicted_opt_Y_adjusted": y_test_pred_sorted,
        "absolute_error": abs_error
    })


    # Optional:
    # results_df.to_excel("svr_test_predictions.xlsx", index=False)


if __name__ == "__main__":
    train_svr_opt_y()