def cast_data(df, flavor="plain"):
    pass


def fill_hyperparams(hyperparameters):
    pass


def make_model(hyperparameters):
    # model has to keep track of the training hyperparameters, not just topology hyperparameters
    pass


def train_model(model, train_x, train_y, until=None):
    # return training info
    pass


def forecast_model(model, test_x):
    pass


def train_optimal_models(hyperparameters_ranges, train_x, train_y, trials=None):
    pass
