import mgarch
from sklearn.covariance import LedoitWolf

def cov(assets_returns_df, full_upto_assets_returns_df=None):
    return assets_returns_df.cov()

def cov_shrinkage(assets_returns_df, full_upto_assets_returns_df=None):
    cov_matrix = assets_returns_df.cov()
    cov_est = LedoitWolf().fit(cov_matrix).covariance_
    return cov_est

def mgarch_forecast(assets_returns_df, full_upto_assets_returns_df, max_lookback=None):
    lookback_assets_returns_df = (full_upto_assets_returns_df if max_lookback is None 
                                  else full_upto_assets_returns_df.loc[-max_lookback:])
    model = mgarch.mgarch().fit(lookback_assets_returns_df)
    cov_est = model.predict(len(assets_returns_df.index))
    return cov_est

def mgarch_shrinkage(assets_returns_df, full_upto_assets_returns_df, max_lookback=None):
    cov_est = mgarch_forecast(assets_returns_df, full_upto_assets_returns_df, max_lookback=max_lookback)
    cov_est = LedoitWolf().fit(cov_est).covariance_
    return cov_est
    

    
    
