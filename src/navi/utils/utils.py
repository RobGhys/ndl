from itertools import product
import pandas as pd
from pathlib import Path
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR


class Utils:
    def __init__(self, snapshot_dir: str):
        self.all_grid_params = None
        self.snapshot_dir = snapshot_dir

    def get_all_grid_params(self, model_type):
        # Common grid search parameters
        param_grid: dict = {
            'learning_rate': [1e-3],
            'batch_size': [8],
            'hidden_size': [64],
            'epochs': [50],
            'optimizer': ['AdamW'],
            'scheduler': ['Plateau']
        }
        if model_type == 'gru':
            param_grid.update({
                "context_size": [5],
                "n_layers": [1]
            })
        if model_type == 'fc':
            param_grid.update({
                "context_size": [0],
            })

        self.all_grid_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

        return self.all_grid_params

    def log_params(self):
        df = pd.DataFrame(data=self.all_grid_params)
        print(df)
        filepath = Path(self.snapshot_dir) / 'grid.csv'
        df.to_csv(filepath)
        return df

    @staticmethod
    def get_scheduler(scheduler_name, optimizer, const_lr_factor=None, const_lr_total_iters=None):
        if scheduler_name == 'StepLR':
            return StepLR(optimizer, step_size=10, gamma=0.1)
        if scheduler_name == 'Constant':
            return ConstantLR(optimizer, factor=const_lr_factor, total_iters=const_lr_total_iters)
        if scheduler_name == 'Plateau':
            return ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-5)


if __name__ == '__main__':
    from itertools import product
    import pandas as pd

    snapshot_dir = '/home/rob/Documents/Github/navi_lstm/output/snapshot'
    utils = Utils(snapshot_dir)
    params = utils.get_all_grid_params()
    df = utils.log_params()