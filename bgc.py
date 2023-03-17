import h5py
import numpy as np
import pandas as pd
import numpyro as no
import plotly.express as px
import plotly.graph_objects as go

__all__ = ["BGC", "Posterior", "cline_func", "cline_plot"]

class BGC():
    def __init__(self, files):
        self.n_chains = len(files)
        self.chains = [h5py.File(path) for path in files]
        self.keys = self.chains[0].keys()

    def posterior(self, key, burnin=0, prob=0.95):
        return Posterior(self, key, burnin, prob)

class Posterior():
    def __init__(self, bgc, key, burnin, prob):
        ## files: list of file paths to hdf5 output from BGC 
        ## key: key value for parameter
        ## burning: number of samples to exclude for burnin
        ## prob: hpd interval 
        self.param = key
        self.n_chains = bgc.n_chains 
        self.lower = round((1.0 - prob) / 2, 3) * 100 
        self.upper = 100 - self.lower
        if len(bgc.chains[0][key].shape) == 1:
            self.param_dims = 1
        elif len(bgc.chains[0][key].shape) == 2 or len(bgc.chains[0][key].shape) == 3:
            self.param_dims = 2 
        else:
            quit("Invalid input array shape")

        chains = []
        for file in bgc.chains: 
            if self.param_dims == 1:
                chains.append(np.array(file[key]))
            elif self.param_dims == 2:
                chains.append(np.array(file[key]).reshape(file[key].shape[0], -1))
        self.chains = np.array(chains)

        data = self.chains[..., burnin:]
        if self.param_dims == 1:
            summary = no.diagnostics.summary(data, prob=prob)["Param:0"]
            summary.update({"parameter": "{}".format(key)})
            summary.move_to_end("parameter", last=False)
            self.summary = pd.DataFrame(summary, index=[0])
        elif self.param_dims == 2:
            summaries = []
            for i in range(data.shape[1]):
                summary = no.diagnostics.summary(data[:,i,:], prob=prob)["Param:0"]
                summary.update({"parameter": "{}_{}".format(key, i)})
                summary.move_to_end("parameter", last=False)
                summaries.append(summary)
            self.summary = pd.DataFrame(summaries)

    def to_df(self):
        data = self.chains
        df = pd.DataFrame(data.transpose())
        df.index.names = ["Sample"]
        df.columns = list(range(1, self.n_chains + 1)) 
        return df

    def to_long_df(self, indices=None):
        data = self.chains 
        if indices is not None:
            data = data[:,indices,:] 
        else:
            indices = range(data.shape[1])
        dfs = [] 
        for i in range(data.shape[1]):
            df = pd.DataFrame(data[:,i,:].transpose())
            df.columns = list(range(1, self.n_chains + 1)) 
            df.index.names = ["Sample"]
            df["param"] = indices[i]
            dfs.append(df)
        df = pd.concat(dfs)   
        return df

    def histogram(self, path, indices=None):
        if self.param_dims == 1:
            if indices is not None:
                print("Warning: indices arg not used for {} parameter".format(self.param))
            df = self.to_df()
            fig = px.histogram(df, x=[*range(1, self.n_chains + 1)], title=self.param, 
                               barmode="overlay", histnorm="probability density")
        elif self.param_dims == 2:
            df = self.to_long_df(indices)
            fig = px.histogram(df, x=[*range(1, self.n_chains + 1)], title=self.param,
                    barmode="overlay", #histnorm='probability density',
                    facet_col="param", facet_col_wrap=4, facet_row_spacing=0.01, 
                    facet_col_spacing=0.01)
                     
        fig.update_layout(legend_title_text="Chain", title=dict(x=0.5))
        # fig.write_html(path)
        fig.show() 

    def trace(self, path, indices=None):
        if self.param_dims == 1:
            if indices is not None:
                print("Warning: indices arg not used for {} parameter".format(self.param))
            df = self.to_df()
            fig = px.line(df, x=df.index, y=[*range(1, self.n_chains + 1)], 
                          title=self.param)
        elif self.param_dims == 2:
            df = self.to_long_df(indices)
            # figs = []
            # for name, group in df.groupby(by="param"):
            #     fig = px.line(
            #             group, 
            #             x=group.index, 
            #             y=[*range(1, self.n_chains + 1)], 
            #             height=200,
            #             width=200,
            #             title="{} {}".format(param, name))
            #     figs.append(fig)
            # with open(path, "a") as fh:
            #     for i in figs:
            #         fh.write(i.to_html(full_html=False, include_plotlyjs='cdn'))
                
            fig = px.line(df, x=df.index, y=[*range(1, self.n_chains + 1)], 
                    title=self.param, facet_col="param", facet_col_wrap=4, 
                    facet_row_spacing=0.01, facet_col_spacing=0.01)
        fig.update_layout(legend_title_text="Chain", title=dict(x=0.5))
        fig.update_traces(line=dict(width=1))
        # fig.write_html(path)
        fig.show()

    def covered(self, x):
        ## Returns list of booleans
        ## True if x is covered by credible interval
        ## False if x is not covered by credible interval
        lower = "{}%".format(self.lower)
        upper = "{}%".format(self.upper)
        df = self.summary
        return (x > df[lower]) & (x < df[upper])

def cline_func(a, b, h):
    thetas = h + 2 * h *(1 - h) * (a + b * (2 * h - 1))
    thetas[thetas > 1] = 1
    thetas[thetas < 0] = 0
    return thetas

def cline_plot(alpha, beta, alpha_outliers=None, beta_outliers=None):
    assert len(alpha) == len(beta)
    if alpha_outliers is None:
        alpha_outliers = pd.Series(False, index=range(len(alpha)))
    else:
        assert len(alpha) == len(alpha_outliers)
    if beta_outliers is None:
        beta_outliers = pd.Series(False, index=range(len(beta)))
    else:
        assert len(alpha) == len(beta_outliers)
    x = np.linspace(0, 1, 100) 
    fig = go.Figure()
    def add_trace( a, b, color):
        for i in range(len(a)): 
            phi = cline_func(a[i], b[i], x)  
            fig.add_trace(go.Scatter(x=x, y=phi, mode="lines", line=dict(color=color)))
    # Non outlier
    add_trace( 
        alpha[~alpha_outliers & ~beta_outliers].reset_index(drop=True),
        beta[~alpha_outliers & ~beta_outliers].reset_index(drop=True),
        "lightslategray")
    # alpha outlier
    add_trace(
        alpha[alpha_outliers & ~beta_outliers].reset_index(drop=True),
        beta[alpha_outliers & ~beta_outliers].reset_index(drop=True),
        "orange")
    # beta outlier
    add_trace(
        alpha[~alpha_outliers & beta_outliers].reset_index(drop=True),
        beta[~alpha_outliers & beta_outliers].reset_index(drop=True),
        "red")
    # alpha and beta outlier
    add_trace(
        alpha[alpha_outliers & beta_outliers].reset_index(drop=True),
        beta[alpha_outliers & beta_outliers].reset_index(drop=True),
        "green")
    fig.update_layout(
        showlegend=False,
        yaxis_range=[0,1],
        xaxis_range=[0,1],
        xaxis_title="Hybrid Index",
        yaxis_title="Prob. Ancestry",
        width=1000,
        height=1000,
    )
    fig.show()