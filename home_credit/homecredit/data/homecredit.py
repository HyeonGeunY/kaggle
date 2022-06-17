import argparse
import toml
from homecredit.util import timer

from homecredit.data.base_data_module import BaseDataModule, check_data_downloaded, load_and_print_info


DATA_DIRNAME = BaseDataModule.data_dirname() # input/
METADATA_FILENAME = DATA_DIRNAME / "raw" / "homecredit_raw.toml"
ZIPFILE_PATH = DATA_DIRNAME / "home-credit-default-risk.zip"

DEBUG = False


if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

class HomeCredit():
    def __init__(self, args: argparse.Namespace = None):
        
        self.args = vars(args) if args is not None else {}
        self.debug = self.args.get("debug", DEBUG)
        
        
    def prepare_data(self):
        if not os.path.exists(ZIPFILE_PATH):
            metadata = toml.load(METADATA_FILENAME)
            check_data_downloaded(metadata, BaseDataModule.data_dirname())
            return
        
        # dataload
        num_rows = 30000 if self.debug else None
        
        with timer("application_train and application_test"):
            df = get_train_test(DATA_DIRNAME, num_rows= num_rows)
            print("Application dataframe shape: ", df.shape)
            
        
    def setup(self, stage: str = None):
        
        
        