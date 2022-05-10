import IPython.display as ipd

class AudioP():
    def __init__(self):
        self.root_dir = ROOTDIR
        # edit metadata
        self.meta = pd.read_csv(fr"{self.root_dir}/train_metadata.csv")
        
    def play_random_sample_by_feature(self, key: str = None, value: str = None, random_state: int = 33):
        audio = train_df[train_df[key] == value].sample(1, random_state = 33)['full_path'].values[0]
        print(f"{key}: {value}")
        return ipd.display(ipd.Audio(audio)) # ipd.Audio(audio)
    
