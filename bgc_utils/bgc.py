import h5py
import numpy as np
import pandas as pd
import numpyro as no
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pcolors
from scipy.stats import norm
from collections.abc import Iterable

__all__ = ["BGC", "Posterior", "cline_func", "cline_plot", "outliers"]

class BGC():
    def __init__(self, files, burnin, interval):
        ## files: list of file paths to hdf5 output from BGC 
        ## burnin: number of samples to exclude for burnin
        ## interval: highest posterior density interval 
        self.n_chains = len(files)
        self.chains = [h5py.File(path) for path in files]
        self.keys = self.chains[0].keys()
        self.burnin = burnin
        self.lower_quantile = round((1.0 - interval) / 2, 3)
        self.upper_quantile = 1 - self.lower_quantile
        self.interval = interval

    def posterior(self, key):
        return Posterior(self, key)

class Posterior():
    def __init__(self, bgc, key):
        ## bgc: BGC object
        ## key: key value for parameter
        self.bgc = bgc
        self.param = key
        self.n_chains = bgc.n_chains 
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
        self.samples = np.array(chains)[..., bgc.burnin:]  
        if self.param_dims == 1:
            summary = no.diagnostics.summary(self.samples, prob=bgc.interval)["Param:0"]
            summary.update({"parameter": "{}".format(key)})
            summary.move_to_end("parameter", last=False)
            self.summary = pd.DataFrame(summary, index=[0])
        elif self.param_dims == 2:
            summaries = []
            for i in range(self.samples.shape[1]):
                summary = no.diagnostics.summary(self.samples[:,i,:], prob=bgc.interval)["Param:0"]
                summary.update({"parameter": "{}_{}".format(key, i)})
                summary.move_to_end("parameter", last=False)
                summaries.append(summary)
            self.summary = pd.DataFrame(summaries)

    def to_df(self):
        df = pd.DataFrame(self.samples.transpose())
        df.index.names = ["Sample"]
        df.columns = list(range(1, self.n_chains + 1)) 
        return df

    def to_long_df(self, indices=None):
        if indices is not None:
            data = self.samples[:,indices,:] 
        else:
            data = self.samples 
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

    def histogram(self, indices=None):
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
        return fig

    def trace(self, indices=None):
        if self.param_dims == 1:
            if indices is not None:
                print("Warning: indices arg not used for {} parameter".format(self.param))
            df = self.to_df()
            fig = px.line(df, x=df.index, y=[*range(1, self.n_chains + 1)], 
                          title=self.param)
        elif self.param_dims == 2:
            df = self.to_long_df(indices)
            fig = px.line(df, x=df.index, y=[*range(1, self.n_chains + 1)], 
                    title=self.param, facet_col="param", facet_col_wrap=4, 
                    facet_row_spacing=0.01, facet_col_spacing=0.01)
        fig.update_layout(legend_title_text="Chain", title=dict(x=0.5))
        fig.update_traces(line=dict(width=1))
        return fig

    def covered(self, x):
        ## Returns list of booleans
        ## True if x is covered by credible interval
        ## False if x is not covered by credible interval
        lower = "{}%".format(self.bgc.lower_quantile * 100)
        upper = "{}%".format(self.bgc.upper_quantile * 100)
        df = self.summary
        return (x > df[lower]) & (x < df[upper])


def cline_func(a, b, h):
    thetas = h + 2 * h *(1 - h) * (a + b * (2 * h - 1))
    thetas[thetas > 1] = 1
    thetas[thetas < 0] = 0
    return thetas

def outliers(param, tau, interval, central_tendancy="median"):
#     ## param: posterior object for alpha or beta parameter
#     ## tau: posterior object for alpha tau or alpha beta estimates
#     ## interval: interval of normal probability density function 
#     ## central_tendancy: "median" or "mean"
    lower_quantile = round((1.0 - interval) / 2, 3)
    upper_quantile = 1 - lower_quantile
    tau_central = tau.summary.iloc[0][central_tendancy]
    lower = norm.ppf(lower_quantile, scale=1/tau_central)
    upper = norm.ppf(upper_quantile, scale=1/tau_central)
    param_central = param.summary[central_tendancy]
    return (param_central < lower) | (param_central > upper)

def cline_plot(alpha, beta, outliers=None, outlier_label=None, outlier_colors=None,
        label="Neutral", color="lightslategray"):
    ## outliers: a pandas boolean series or python list containing pandas boolean series
    assert len(alpha) == len(beta)
    if not isinstance(outliers, list):
        outliers = [outliers]
    x = np.linspace(0, 1, 100) 
    fig = go.Figure()
    if outliers is None:
        alpha0 = alpha
        beta0 = beta
    else:
        outliers0 = np.logical_or.reduce(outliers)
        alpha0 = alpha[~outliers0].reset_index(drop=True)
        beta0 = beta[~outliers0].reset_index(drop=True)
    def add_trace(a, b, name, color):
        for i in range(len(a)): 
            showlegend = True if i == 0 else False
            fig.add_trace(go.Scatter(x=x, y=cline_func(a[i], b[i], x), 
                    name=name, legendgroup=name, showlegend=showlegend,
                    line=dict(color=color), mode="lines"))

    add_trace(alpha0, beta0, label, color)
    for i in range(len(outliers)):
        alpha1 = alpha[outliers[i]].reset_index(drop=True)
        beta1 = beta[outliers[i]].reset_index(drop=True) 
        if outlier_label is None:
            lab = f"Outlier {i+1}"
        else: 
            lab[i]
        if outlier_colors is None:
            col = pcolors.DEFAULT_PLOTLY_COLORS[i+1]
        else:
            col = outlier_colors[i]
        add_trace(alpha1, beta1, lab, col)
    fig.update_layout(
        plot_bgcolor="white",
        width=500,
        height=500,
        margin=dict(l=0,r=0,b=0,t=0),
        yaxis_range=[0,1],
        xaxis_range=[0,1],
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01))
    fig.update_xaxes(
        title_text="Hybrid Index",
        title_font=dict(size=20),
        showline=True,
        ticks="outside",
        linecolor="black",
        linewidth=1)
    fig.update_yaxes(
        title_text="Prob. Ancestry",
        title_font=dict(size=20),
        showline=True,
        ticks="outside", 
        linecolor="black",
        linewidth=1)

    return fig