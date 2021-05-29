import os
import sys
import time
import pytube as pt
import pandas as pd
import numpy as np
import time

sys.path.insert(1, '')
from segmentator import Segmentator
from chunck import Chunck


class Parser:
    def __init__(self, seg, report_path, 
                 headers={
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:85.0) Gecko/20100101 Firefox/85.0'
      }):
        self.headers = headers
        self.seg = seg
        self.report_path = report_path
        self.columns = ['word', 'length', 'velocity_x', 'velocity_y', 'author']
        
    def parse(self, url, out_path):
        youtube = pt.YouTube(url)
        temp_video_path = 'temp.mp4'
        video = youtube.streams.first().download(filename='temp')
        partial_df = self.seg.get_chuncks(temp_video_path, out_path)
        df = pd.DataFrame(columns=[self.columns[-1]],
                          data=[[youtube.author]])
        
        report = pd.concat([partial_df, df], axis=1)
        
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
        report.to_csv(self.report_path, index=False)
        return report
        
    def from_csv(self, csv_path, out_path):
        urls_df = pd.read_csv(csv_path)
        urls_temp = urls_df.to_numpy()
        urls = []
        for url in urls_temp:
            urls.append(url[0])
        urls = np.array(urls, dtype=str)
        report = pd.DataFrame(columns=self.columns)
        
        for url in urls:
             report = pd.concat([report, self.parse(url, out_path)])
             time.sleep(300)
                
        report.to_csv(self.report_path, index=False)
        
        return report
    