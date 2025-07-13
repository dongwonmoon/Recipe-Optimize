import numpy as np
import pandas as pd
import joblib
import optuna
import time  # 시간 측정을 위해 time 모듈 추가

# --- 1. 모델 및 데이터 로드 ---
print(">>> [1/5] 모델과 데이터를 로드합니다...")
try:
    model = joblib.load("et_regressor_model.joblib")
    df = pd.read_csv("./data/cleaned.csv")
    print(">>> 모델과 데이터 로드 완료.")
except FileNotFoundError as e:
    print(f"오류: {e}")
    print(
        "스크립트를 실행하기 전에 'et_regressor_model.joblib' 파일과 './data/cleaned.csv' 파일이 있는지 확인해주세요."
    )
    exit()

X = df.drop("quality", axis=1)

# --- 2. 최적화 변수 및 탐색 공간 정의 ---
not_to_use = ["density", "residual sugar", "alcohol"]
used_cols = [col for col in X.columns if col not in not_to_use]
X_used_cols_values = X[used_cols].values
param_ranges = {col: (X[col].min(), X[col].max()) for col in used_cols}


# --- 3. Objective 함수 (수정 없음) ---
def objective(trial):
    """하나의 trial(시도)에 대해 평가 점수를 반환하는 함수"""
    individual = [
        trial.suggest_float(col, low, high) for col, (low, high) in param_ranges.items()
    ]
    distances = np.linalg.norm(X_used_cols_values - individual, axis=1)
    closest_index = np.argmin(distances)
    feasibility_score = distances[closest_index]
    closest_real_row = X.iloc[closest_index]
    reconstructed_series = pd.Series(individual, index=used_cols)
    for col in not_to_use:
        reconstructed_series[col] = closest_real_row[col]
    recipe_df = pd.DataFrame([reconstructed_series.reindex(X.columns)])
    predicted_quality = model.predict(recipe_df)[0]
    return predicted_quality, feasibility_score


# --- 4. 최적화 실행 함수 ---
def run_optimization(n_trials=200):
    study = optuna.create_study(directions=["maximize", "minimize"])

    # 최적화 시작을 알리는 print문 추가
    print(f"\n>>> [3/5] Optuna 최적화를 시작합니다. (총 시도 횟수: {n_trials})")
    print(">>> 진행 막대가 표시됩니다. 시간이 다소 소요될 수 있습니다...")

    start_time = time.time()  # 시작 시간 기록
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    end_time = time.time()  # 종료 시간 기록

    print(f">>> 최적화 완료! (소요 시간: {end_time - start_time:.2f}초)")
    return study


# --- 5. 메인 실행 블록 ---
if __name__ == "__main__":
    print("\n>>> [2/5] 최적화 프로세스를 시작합니다.")
    study = run_optimization(n_trials=500)  # 시도 횟수를 조절할 수 있습니다.

    print("\n>>> [4/5] 최적의 레시피 결과를 처리하고 있습니다...")
    best_trials = study.best_trials

    if not best_trials:
        print("최적의 조합을 찾지 못했습니다.")
    else:
        recipes_list = []
        for trial in best_trials:
            optimized_params = trial.params
            individual = [optimized_params[col] for col in used_cols]
            distances = np.linalg.norm(X_used_cols_values - individual, axis=1)
            closest_index = np.argmin(distances)
            closest_real_row = X.iloc[closest_index]
            reconstructed_series = pd.Series(individual, index=used_cols)
            for col in not_to_use:
                reconstructed_series[col] = closest_real_row[col]
            full_recipe = reconstructed_series.reindex(X.columns)
            recipes_list.append(full_recipe)

        best_recipes = pd.DataFrame(recipes_list)
        best_recipes_fitness = pd.DataFrame(
            [t.values for t in best_trials], columns=["quality", "distance"]
        )

        print(">>> [5/5] 결과를 CSV 파일로 저장합니다...")
        best_recipes.to_csv("best_recipes.csv", index=False)
        best_recipes_fitness.to_csv("best_recipes_fitness.csv", index=False)

        print("\n" + "=" * 50)
        print("🎉 모든 프로세스가 성공적으로 완료되었습니다! 🎉")
        print(f"파레토 프론티어에서 {len(best_trials)}개의 최적 레시피를 찾았습니다.")
        print(
            "결과는 'best_recipes.csv'와 'best_recipes_fitness.csv' 파일을 확인해주세요."
        )
        print("=" * 50)
