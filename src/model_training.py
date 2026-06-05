def save_all_models(
    self,
    models,
    feature_names
):

    import joblib
    import os

    os.makedirs(
        "models",
        exist_ok=True
    )

    for name, model in models.items():

        filename = (

            name.lower()

            .replace(" ", "_")

            .replace("-", "_")

            + ".pkl"

        )

        joblib.dump(

            model,

            f"models/{filename}"

        )

    joblib.dump(

        self.best_model,

        "models/best_model.pkl"

    )

    joblib.dump(

        self.scaler,

        "models/scaler.pkl"

    )

    joblib.dump(

        feature_names,

        "models/feature_names.pkl"

    )
        
