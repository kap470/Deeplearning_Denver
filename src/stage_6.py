
# Hyperparameter configurations
param_1 = [lstm_model_1, 50]
param_2 = [lstm_model_1, 100]
param_3 = [lstm_model_2, 50]
param_4 = [lstm_model_2, 100]

parameter_grid = [
    param_1,
    param_2,
    param_3,
    param_4
]

# General settings
feature_cols = []
no_feat = 15
training_loss = []
lookback = 1

# Tuning
for well in wells:

    print(f'\nProcessing {well}')

    # Reset best score for EACH well
    best_score = float('inf')
    best_params = None

    outer_sets, inner_sets = dataset_create(well)

    # Inner-set hyperparameter tuning
    for i, params in enumerate(parameter_grid):

        print(f'\nTesting parameter configuration {i+1}')

        best_score, best_params = train_one_split(
            dataset=inner_sets,
            parameter_list=params,
            split_name=f'Param {i+1}',
            set='inner_sets',
            best_score=best_score,
            best_params=best_params,
            well=well
        )

    print(f'\nBest parameters for {well}:')
    print(best_params)
    print(f'Best inner RMSE: {best_score:.4f}')

    # Train selected configuration on outer training set
    best_score_val, best_params_val = train_one_split(
        dataset=outer_sets,
        parameter_list=best_params,
        split_name='Validation',
        set='outer_sets',
        best_score=float('inf'),
        best_params=None,
        well=well
    )
