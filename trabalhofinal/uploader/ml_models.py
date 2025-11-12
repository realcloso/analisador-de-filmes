import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def _get_model(model_name, hp_params):
    """
    Limpa os hiperparâmetros (de string para número) e retorna uma instância do modelo.
    """
    cleaned_hps = {}
    for k, v in hp_params.items():
        if not v:
            continue
        try:
            cleaned_hps[k] = int(v)
        except ValueError:
            try:
                cleaned_hps[k] = float(v)
            except ValueError:
                cleaned_hps[k] = v

    if model_name == "KNN":
        return KNeighborsClassifier(**cleaned_hps)
    if model_name == "DecisionTree":
        return DecisionTreeClassifier(**cleaned_hps)
    if model_name == "RandomForest":
        return RandomForestClassifier(**cleaned_hps)
    if model_name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, **cleaned_hps)
    if model_name == "SVM":
        base_hps = {"probability": True}
        base_hps.update(cleaned_hps)
        return SVC(**base_hps)

    raise ValueError(f"Modelo desconhecido: {model_name}")


def _get_pipeline(X, Y_raw, model_name, hp_params):
    """
    Cria o pipeline de pré-processamento e o modelo final.
    """

    numeric_features = X.select_dtypes(include=["number"]).columns
    categorical_features = X.select_dtypes(exclude=["number"]).columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    try:
        model = _get_model(model_name, hp_params)
    except Exception as e:
        print(f"Erro ao instanciar modelo com HPs {hp_params}: {e}. Usando defaults.")
        model = _get_model(model_name, {})

    main_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", model)]
    )

    le = LabelEncoder().fit(Y_raw)
    return main_pipeline, le


def run_ml_task(
    df: pd.DataFrame, model_name: str, hp_params: dict, new_data_dict: dict, action: str
):
    """
    Função principal que orquestra o pipeline de ML.
    """
    try:
        X = df.iloc[:, :-1]
        Y_raw = df.iloc[:, -1]
    except Exception as e:
        return {
            "output": f"Erro ao separar X e Y. A base precisa ter ao menos 2 colunas. Erro: {e}",
            "metrics": "N/A",
        }

    try:
        pipeline, le = _get_pipeline(X, Y_raw, model_name, hp_params)
        Y = le.transform(Y_raw)
    except Exception as e:
        return {"output": f"Erro ao construir pipeline: {e}", "metrics": "N/A"}

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    try:
        pipeline.fit(X_train, Y_train)
    except Exception as e:
        return {"output": f"Erro ao treinar modelo: {e}", "metrics": "N/A"}

    Y_pred = pipeline.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    metrics = f"acc={acc:.2f} (baseado em split 80/20 da base original)"

    if action == "retrain":
        return {
            "output": f"Modelo {model_name} re-treinado com HPs: {hp_params}",
            "metrics": metrics,
        }

    if action == "predict":
        try:
            cleaned_data_dict = {
                k.replace("X_", ""): v
                for k, v in new_data_dict.items()
                if k.startswith("X_")
            }

            new_data_df = pd.DataFrame([cleaned_data_dict])
            for col in new_data_df.columns:
                if col in X.columns:
                    try:
                        new_data_df[col] = new_data_df[col].astype(X[col].dtype)
                    except Exception:
                        new_data_df[col] = pd.to_numeric(
                            new_data_df[col], errors="ignore"
                        )

            pred_encoded = pipeline.predict(new_data_df)
            pred_proba = pipeline.predict_proba(new_data_df)

            prediction_label = le.inverse_transform(pred_encoded)[0]

            score_index = pred_encoded[0]
            score = pred_proba[0][score_index]

            return {
                "output": f"Predição: classe='{prediction_label}' / score={score:.2f}",
                "metrics": metrics,
            }

        except Exception as e:
            return {"output": f"Erro na predição: {e}", "metrics": metrics}

    return {"output": "Ação desconhecida.", "metrics": "N/A"}
