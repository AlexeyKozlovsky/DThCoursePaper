import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Analyzer:
    def __init__(self, csv_path, props):
        self.report = pd.read_csv(csv_path)
        self.props = props
        self._prepare()
        
    def _prepare(self):
        data = np.zeros((self.report.shape[0]))
        for i, row in self.report.iterrows():
            data[i] = np.sqrt(row['velocity_x'] ** 2 + row['velocity_y'] ** 2)
            velocity_df = pd.DataFrame(columns=['velocity'], data=data)

        self.report = pd.concat([self.report, velocity_df], axis=1)
        
        self.authors = self.report['author'].unique()
        length = self.authors.shape[0]
        self.views = np.zeros((length), dtype=object)

        props = self.props
        self.means, self.stds = {}, {}
        for prop in props:
            self.means[prop] = np.zeros((length))
            self.stds[prop] = np.zeros((length))
        
        for prop in props:
            for i, author in enumerate(self.authors):
                self.views[i] = self.report.loc[self.report.author == author]
                self.means[prop][i] = np.mean(self.views[i][prop])
                self.stds[prop][i] = np.std(self.views[i][prop])             
        
    def _get_view(self, view_num):
        if type(view_num) is list:
            view = self.views[view_num[0]]
            for _, v_num in enumerate(view_num, start=1):
                view = pd.concat([view, self.views[v_num]])
        else:
            view = self.views[view_num]
            
        return view
    
    def get_props(self, view_num, prop):
        mean, std = 0, 0
        if type(view_num) is list:
            mean = np.mean([self.means[prop][v_n] for v_n in view_num])
            std = np.std([self.stds[prop][v_n] for v_n in view_num])
        else:
            mean = self.means[prop][view_num]
            std = self.stds[prop][view_num]
            
        return mean, std
    
    def plot_scatter(self, view_num):
        view = self._get_view(view_num)
        plt.scatter(view.length, view.velocity)
        
        plt.xlabel('length')
        plt.ylabel('velocity')
        plt.show()
    
    def plot_hist(self, view_num, prop, num_bins=10):
        view = self._get_view(view_num) 
        plt.hist(view[prop], num_bins, density=1, alpha=0.7)
        plt.xlabel(prop)
        plt.show()
    
    def compare_to_normal(self, view_num, prop, num_bins=10):
        view = self._get_view(view_num)
        _, bins, _ = plt.hist(view[prop], num_bins, density=True, 
                              alpha=0.7)
        
        mean, std = self.get_props(view_num, prop)
        print(mean, std)
        y = ((1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((bins - mean) / std  ) ** 2))

        plt.plot(bins, y, '--', color='black')
        plt.xlabel(prop)
        plt.show()
        
    def dist_plot(self, view_num, prop, num_bins=10):
        view = self._get_view(view_num)
        sns.displot(view[prop], kde=True, bins=num_bins)
        plt.xlabel(prop)
        plt.show()
