if st.button(
    "Train Models & Generate Pickle Files"
):

    trainer = ModelTrainer(
        X,
        y
    )
