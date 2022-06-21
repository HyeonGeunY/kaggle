if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from PIL import Image
import torch
import torch.nn as nn

from birdclef.data import BirdClef2022_v3
from birdclef.lit_models import BaseLitModel
from birdclef.models import ResNetBird
from birdclef.data.augmentation_v3 import _audio_to_mel_label
from birdclef.util import show_heatmap


CONFIG_AND_WEIGHTS_DIRNAME = (
    Path(__file__).resolve().parent / "artifacts" / "v3_birdclef"
)

DATA_DIRNAME = Path(__file__).resolve().parents[1] / "input"
SCORED_BIRD_FILENAME = DATA_DIRNAME / "birdclef-2022" / "scored_birds.json"
SUBMISSION_DIR = Path(__file__).resolve().parents[1] / "submission"


class BirdClef2022:
    """Birdclef model trained by v3 data"""

    def __init__(self):
        data = BirdClef2022_v3()
        self.mapping = np.array(data.mapping)  # 기존 mapping에 "\n"가 추가되어 있는 mapping
        self.min_sec_proc = data.min_sec_proc
        self.out_sigmoid = nn.Sigmoid()
        self.sr = data.sr
        self.mel_converter = data.mel_converter
        
        with open(SCORED_BIRD_FILENAME) as f:
            self.scored_birds = json.load(f)
        
    
        with open(CONFIG_AND_WEIGHTS_DIRNAME / "config.json", "r") as file:
            config = json.load(file)
            
        args = argparse.Namespace(**config)

        # 모델 형태를 만들고 가중치를 로드해 업데이트 하는 방식으로 모델을 불러온다.
        model = ResNetBird(data_config=data.config(), args=args)
        
        self.lit_model = BaseLitModel.load_from_checkpoint(
            checkpoint_path=CONFIG_AND_WEIGHTS_DIRNAME / "model.pt", args=args, model=model
        )

        # 평가모드로 전환
        self.lit_model.eval()
        
    @torch.no_grad()
    def predict(self, test_audio_dir):
        
        file_list = [f.split('.')[0] for f in sorted(os.listdir(test_audio_dir))]
        
        pred = {'row_id': [], 'target': []}
        thres = self.lit_model.threshold
        pred_birds_list = []
        
        for filename in file_list:
            path = os.path.join(str(test_audio_dir), filename + ".ogg")
            
            chunks = [[] for i in range(12)]
            
            mel_list = []
            mel_list = _audio_to_mel_label(path, self.min_sec_proc, is_test=True, mel_list=mel_list, sample_rate=self.sr, mel_converter=self.mel_converter)
            mel_list = torch.stack(mel_list)
            
            outputs = self.lit_model(mel_list)
            outputs_test = self.out_sigmoid(outputs)
            for idx, i in enumerate(range(len(chunks))):
                chunk_end_time = (i + 1) * 5
                
                output_birds = outputs_test[idx] > thres
                pred_birds = ", ".join(
                    [self.mapping[j] for j in range(len(output_birds)) if output_birds[j] == True]
                    )
                pred_birds_list.append(pred_birds)
                for bird in self.scored_birds:
                    try:
                        score = outputs_test[idx][np.where(self.mapping==bird)]
                    except IndexError:
                        score = 0
                    
                    row_id = filename + '_' + bird + '_' + str(chunk_end_time)
                    
                    pred['row_id'].append(row_id)
                    pred['target'].append(True if score > thres else False)
                    
                    
        results = pd.DataFrame(pred, columns=['row_id', 'target'])
        print(results['target'])
        
        if not os.path.exists(SUBMISSION_DIR):
            os.makedirs(SUBMISSION_DIR)
            
        results.to_csv(SUBMISSION_DIR / "submission.csv", index=False)
        show_heatmap(results, save_dir=SUBMISSION_DIR)
        
        return pred_birds_list


def main():
    # """
    # Example runs:
    # '''
    # python birdclef/birdclef_model.py C:\Users\ftmlab\Documents\hyoon\project_new\kaggle\birdclef2022\input\birdclef-2022\test_soundscapes
    # '''
    # """
    
    parser = argparse.ArgumentParser(description="Recognize test soundscape in an sound file.")
    # filename을 parser의 위치 인자로 사용
    parser.add_argument("test_audio_dir", type=str)
    args = parser.parse_args()

    birdclef = BirdClef2022()
    pred_str = birdclef.predict(Path(args.test_audio_dir))
    print(pred_str)


if __name__ == "__main__":
    main()