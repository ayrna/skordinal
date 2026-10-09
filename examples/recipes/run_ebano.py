from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
from skordinal.classifiers import EBANO
from skordinal.metrics import average_mean_absolute_error
from skordinal.datasets import download_tocuco, load_dataset


# Configuration parameters
dataset_name = "dr04_machine" # oc03_tae, 46014, balance_scale
tocuco_dataset = True
data_home = None
base_models = ["LogisticAT", "LogisticIT"]
results_base_path = Path("results/base_models")
ebano_preds_path = Path("results/EBANO") / dataset_name / "predictions_by_seed"
ebano_models_path = Path("results/EBANO") / dataset_name / "models"
n_resamples = 30
weights_cv_iters = 1000

ebano_results = []
ebano_models_path.mkdir(parents=True, exist_ok=True)

model_paths = [str(results_base_path / model) for model in base_models]

print(f"Loading '{dataset_name}' dataset...")

if tocuco_dataset:
    tocuco_root = download_tocuco(data_home=data_home)
    X, _ = load_dataset(
        name=dataset_name,
        data_home=tocuco_root,
        return_X_y=True,
    )
else:
    X, _ = load_dataset(
        name=dataset_name,
        data_home=data_home,
        return_X_y=True,
    )

print("Running EBANO...")

for seed in range(n_resamples):
    ref_model = base_models[0]
    seed_dir = results_base_path / ref_model / dataset_name / "predictions_by_seed" / f"seed_{seed}"
    
    train_df = pd.read_csv(seed_dir / "train_predictions.csv").sort_values("Pattern ID").reset_index(drop=True)
    test_df = pd.read_csv(seed_dir / "test_predictions.csv").sort_values("Pattern ID").reset_index(drop=True)
    
    y_train = train_df["Target"].values
    y_test = test_df["Target"].values
    pattern_id_train = train_df["Pattern ID"].values
    pattern_id_test = test_df["Pattern ID"].values

    X_train = X[pattern_id_train]
    X_test = X[pattern_id_test]
    
    ebano = EBANO(
        models_saved_results_paths=model_paths,
        dataset_name=dataset_name,
        weights_cv_n_iters=weights_cv_iters,
        random_state=seed
    )
    
    ebano.fit(X_train, y_train)

    y_pred_train = ebano.predict(X_train)
    y_proba_train = ebano.predict_proba(X_train)
    
    y_pred_test = ebano.predict(X_test)
    y_proba_test = ebano.predict_proba(X_test)
    
    amae_test_ebano = average_mean_absolute_error(y_test, y_pred_test)
    
    base_models_amae = {}
    for model_name in base_models:
        model_seed_dir = results_base_path / model_name / dataset_name / "predictions_by_seed" / f"seed_{seed}"
        model_test_df = pd.read_csv(model_seed_dir / "test_predictions.csv").sort_values("Pattern ID").reset_index(drop=True)
        model_y_pred_test = model_test_df["Prediction"].values

        model_amae = average_mean_absolute_error(y_test, model_y_pred_test)
        base_models_amae[f"amae_test_{model_name}"] = model_amae

    seed_result = {
        "resample_id": seed,
        "amae_test_EBANO": amae_test_ebano,
        "optimal_weights": ebano.estimator_weights_.tolist()
    }
    seed_result.update(base_models_amae)
    
    ebano_results.append(seed_result)
    
    print(f"  Resample {seed}: AMAE EBANO = {amae_test_ebano:.4f} | Weights = {ebano.estimator_weights_.round(3)}")
    
    seed_out_dir = ebano_preds_path / f"seed_{seed}"
    seed_out_dir.mkdir(parents=True, exist_ok=True)
    
    train_out_df = pd.DataFrame({
        "Pattern ID": pattern_id_train,
        "Target": y_train,
        "Prediction probabilities": [p.tolist() for p in y_proba_train],
        "Prediction": y_pred_train
    })
    test_out_df = pd.DataFrame({
        "Pattern ID": pattern_id_test,
        "Target": y_test,
        "Prediction probabilities": [p.tolist() for p in y_proba_test],
        "Prediction": y_pred_test
    })
    
    train_out_df.to_csv(seed_out_dir / "train_predictions.csv", index=False)
    test_out_df.to_csv(seed_out_dir / "test_predictions.csv", index=False)
    
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)
    
    def save_formatted_cm(cm, file_path, current_seed):
        with open(file_path, "w") as f:
            f.write(f"Seed {current_seed}\n")
            f.write("=====================\n")
            f.write(np.array2string(cm, separator=', '))
            f.write("\n")
            
    save_formatted_cm(cm_train, seed_out_dir / "train_confusion_matrix.txt", seed)
    save_formatted_cm(cm_test, seed_out_dir / "test_confusion_matrix.txt", seed)
    
    joblib.dump(ebano, ebano_models_path / f"{seed}.joblib")

summary_df = pd.DataFrame(ebano_results)
amae_cols = [col for col in summary_df.columns if "amae_test" in col]

mean_row = {"resample_id": "MEAN"}

for col in amae_cols:
    mean_row[col] = summary_df[col].mean()

mean_weights = np.mean(summary_df["optimal_weights"].tolist(), axis=0)
mean_row["optimal_weights"] = mean_weights.tolist()

summary_df = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)

summary_path = Path(f"results/EBANO/summary_{dataset_name}.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nEBANO summary and models saved successfully in results/EBANO/")
