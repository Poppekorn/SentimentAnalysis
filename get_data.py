"""
Christopher Bryant | ITCS-5154
"""
import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
import config

# Pull daily OHLCV from Alpaca
def get_prices(ticker, stock_client):
    print(f"  {ticker} prices   ")
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=pd.Timestamp(config.START),
        end=pd.Timestamp(config.END),
    )
    bars = stock_client.get_stock_bars(request)
    df = bars.df.reset_index()
    df = df.rename(columns={"symbol": "ticker", "timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]

# Pull news  from Alpaca
def get_news(ticker, news_client):
    print(f"  {ticker} news...")
    try:
        request = NewsRequest(
            symbols=ticker,
            start=pd.Timestamp(config.START),
            end=pd.Timestamp(config.END),
            limit=50,
        )
        response = news_client.get_news(request)

        articles = []
        for key, value in response:
            if key == "data" and "news" in value:
                for a in value["news"]:
                    articles.append({
                        "ticker": ticker,
                        "date": pd.to_datetime(a.created_at).tz_localize(None),
                        "headline": a.headline,
                    })
        return pd.DataFrame(articles)
    except Exception as e:
        print(f"    Error: {e}")
        return pd.DataFrame()

# Pull VIX from Yahoo Finance (Alpaca doesn't have it)
def get_vix():
    print("Pulling VIX  ")
    df = yf.download(config.VIX, start=config.START, end=config.END, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    return df[["date", "close"]].rename(columns={"close": "vix"})


if __name__ == "__main__":
    if not config.ALPACA_KEY or not config.ALPACA_SECRET:
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env first!")
        exit()

    stock_client = StockHistoricalDataClient(config.ALPACA_KEY, config.ALPACA_SECRET)
    news_client = NewsClient(config.ALPACA_KEY, config.ALPACA_SECRET)

    vix = get_vix()
    prices = pd.concat([get_prices(t, stock_client) for t in config.TICKERS])
    news = pd.concat([get_news(t, news_client) for t in config.TICKERS], ignore_index=True)

    vix.to_parquet(config.DATA_RAW / "vix.parquet", index=False)
    prices.to_parquet(config.DATA_RAW / "ohlcv.parquet", index=False)
    news.to_parquet(config.DATA_RAW / "news.parquet", index=False)

    print(f"\nDone! {len(prices)} price rows, {len(news)} headlines")