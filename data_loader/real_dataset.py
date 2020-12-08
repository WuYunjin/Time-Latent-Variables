import logging
import numpy as np
import pandas as pd

class RealDataset(object):
    """

    """
    _logger = logging.getLogger(__name__)

    def __init__(self):

        self._setup()
        self._logger.debug('Finished setting up dataset class')

    def _setup(self):

        stock = pd.read_csv('data_loader/stock.csv',index_col=0)
        stock.dropna(inplace=True)

        # Normalize
        stock_norm = (stock - stock.min() )/(stock.max()-stock.min())
        self.stock = stock_norm.values


if __name__ == "__main__":
    data = RealDataset()


        
        
