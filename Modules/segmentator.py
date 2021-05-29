import sys
import os
import subprocess
import json
import math

import vosk
import librosa
import numpy as np
import pandas as pd
import moviepy.editor as mp

sys.path.insert(1, '../Modules/')
from chunck import Chunck

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


class Segmentator:
    def __init__(self, audio_model_path, video_model_path,
                conf=1):
        self.audio_model_path = audio_model_path
        self.video_model_path = video_model_path
        self.conf = conf
        
    def _to_dict(self, df):
        result = {}
        for index, row in df.iterrows():
            if row['conf'] >= self.conf:
                result[row['word']] = []

        for index, row in df.iterrows():
            if row['conf'] >= self.conf:
                result[row['word']].append((row['start'], row['end']))
        return result
        
    def _extract_words(self, res):
        jres = json.loads(res)
        if not 'result' in jres:
            return []
        words = jres['result']
        return words
    
    def _transcribe_words(self, recognizer, bytes):
        result = []

        chunk_size = 4000
        for chunk_no in range(math.ceil(len(bytes) / chunk_size)):
            start = chunk_no * chunk_size
            end = min(len(bytes), (chunk_no + 1) * chunk_size)
            data = bytes[start : end]

            if recognizer.AcceptWaveform(data):
                words = self._extract_words(recognizer.Result())
                result += words
        result += self._extract_words(recognizer.FinalResult())

        return result
    
    def get_info(self, input_path, out_path=None):
        vosk.SetLogLevel(-1)
        
        temp_audio_path = 'temp.wav'
        clip = mp.VideoFileClip(input_path)
        clip.audio.write_audiofile(temp_audio_path)

        sample_rate = 16000
        audio, sr = librosa.load(temp_audio_path, sr=sample_rate)

        int16 = np.int16(audio * 32768).tobytes()

        model = vosk.Model(self.audio_model_path)
        recognizer = vosk.KaldiRecognizer(model, sample_rate)

        res = self._transcribe_words(recognizer, int16)
        df = pd.DataFrame.from_records(res)
        df = df.sort_values('start')
        
        if os.path.isfile(temp_audio_path):
            os.remove(temp_audio_path)
        
        if out_path is not None:
            df.to_csv(out_path, index=False)
            
        return self._to_dict(df)
    
    def get_chuncks(self, input_path, out_path='', size=(50, 50),
                    word_dict=None):
        # Create a dict for segmented words timestamps
        if word_dict is None:
            word_dict = self.get_info(input_path)
        
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        columns = ['word', 'length', 'velocity_x', 'velocity_y']
        full_df = pd.DataFrame(columns=columns)
            
        for word in word_dict:
            w_num = 0
            l_path = f'{out_path}/{word[0]}'
            w_path = f'{l_path}/{word}'
            
            if not os.path.exists(l_path):
                os.mkdir(l_path)
                
            if not os.path.exists(w_path):
                os.mkdir(w_path)
              
            filenames = os.listdir(w_path)
            for filename in filenames:
                string_num = filename.partition('.')[0]
                if string_num.isdigit():
                    w_num = max(w_num, int(filename.partition('.')[0]))
                
            for current_word in word_dict[word]:
                temp_video_path = f'{w_path}/temp.mp4'

                ffmpeg_extract_subclip(input_path, current_word[0], 
                                       current_word[1],
                                       targetname=temp_video_path)
                
                ch = Chunck(temp_video_path, self.video_model_path, size)
                if not ch.prepare():
                    continue
                    
                try:
                    velocity = ch.to_file(f'{w_path}/{w_num}.avi')
                except:
                    continue
                
                df = pd.DataFrame(columns=columns,
                                 data=[[word, current_word[1] - current_word[0], velocity[0], velocity[1]]])

                full_df = pd.concat([full_df, df])
                df.to_csv(f'{w_path}/{w_num}.csv', index=False)
                
                w_num += 1
                
                if os.path.isfile(temp_video_path):
                    os.remove(temp_video_path)

        return full_df
