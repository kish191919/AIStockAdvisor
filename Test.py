import os
import logging
import json
import sqlite3
import requests
import pandas as pd
import yfinance as yf
import pyotp
import robin_stocks as r
import fear_and_greed
import time
from youtube_transcript_api import YouTubeTranscriptApi
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from deep_translator import GoogleTranslator
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from transformers import GPT2TokenizerFast

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

import json
from datetime import date, datetime
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 데이터프레임 출력 설정 변경
pd.set_option('display.max_columns', None)  # 모든 컬럼 표시
pd.set_option('display.width', 1000)        # 출력 폭 확대

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stock_advisor.log')
    ]
)
logger = logging.getLogger(__name__)


# Configuration class
class Config:
    ROBINHOOD_USERNAME = os.getenv("username")
    ROBINHOOD_PASSWORD = os.getenv("password")
    ROBINHOOD_TOTP_CODE = os.getenv("totpcode")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    ALPHA_VANTAGE_API_KEY = os.getenv("Alpha_Vantage_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Trading decision model
class TradingDecision(BaseModel):
    decision: str
    percentage: int
    reason: str
    expected_next_day_price: float


class TechnicalDataProcessor:
    def __init__(self, variance_ratio_threshold=0.95):
        self.variance_ratio_threshold = variance_ratio_threshold
        self.scalers = {}
        self.pcas = {}
        self.feature_names = {}

    def _prepare_dataframe(self, df):
        """
        Prepare DataFrame by handling missing values and selecting numeric columns
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Drop columns where all values are NaN
        numeric_df = numeric_df.dropna(axis=1, how='all')

        # Forward fill and backward fill to handle NaN values
        numeric_df = numeric_df.ffill()
        numeric_df = numeric_df.bfill()

        # If there are still any NaN values, replace them with 0
        numeric_df = numeric_df.fillna(0)

        # Drop any columns that still have NaN values
        numeric_df = numeric_df.dropna(axis=1)

        # Ensure we have at least one column
        if numeric_df.empty or len(numeric_df.columns) == 0:
            raise ValueError("No valid numeric columns remaining after NaN handling")

        return numeric_df

    def get_feature_contributions(self, prefix):
        """각 PC에 대한 원본 특성들의 기여도 계산"""
        pca = self.pcas[prefix]
        feature_names = self.feature_names[prefix]

        contributions = {}
        for i, pc in enumerate(pca.components_):
            # 각 특성의 절대값 기여도를 계산
            abs_contributions = np.abs(pc)
            # 기여도를 정규화
            normalized_contributions = abs_contributions / np.sum(abs_contributions)
            # 특성별 기여도를 딕셔너리로 저장
            feature_contributions = dict(zip(feature_names, normalized_contributions))
            # 기여도가 큰 순서로 정렬
            sorted_contributions = dict(sorted(
                feature_contributions.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            # 상위 5개 영향력 있는 특성만 선택
            top_contributions = dict(list(sorted_contributions.items())[:5])
            contributions[f'PC{i + 1}'] = top_contributions

        return contributions

    def get_pc_interpretation(self, prefix):
        """각 PC의 의미를 해석"""
        contributions = self.get_feature_contributions(prefix)
        interpretations = {}

        for pc, features in contributions.items():
            # 상위 3개 특성을 문자열로 결합
            top_features = list(features.items())[:3]
            interpretation = " + ".join([
                f"{feature}({weight:.3f})"
                for feature, weight in top_features
            ])
            interpretations[pc] = interpretation

        return interpretations

    def fit_transform(self, df, prefix):
        """PCA 적용 및 변환"""
        # 데이터 준비
        numeric_df = self._prepare_dataframe(df)
        original_index = numeric_df.index

        # 특성 이름 저장
        self.feature_names[prefix] = numeric_df.columns.tolist()

        # 데이터 스케일링
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        self.scalers[prefix] = scaler

        # PCA 적용
        pca = PCA()
        transformed_data = pca.fit_transform(scaled_data)
        self.pcas[prefix] = pca

        # 필요한 컴포넌트 수 결정
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance_ratio >= self.variance_ratio_threshold) + 1

        # 결과 데이터프레임 생성
        columns = [f'PC{i + 1}' for i in range(n_components)]
        pca_df = pd.DataFrame(
            transformed_data[:, :n_components],
            columns=columns,
            index=original_index
        )

        # 특성 기여도 및 해석 계산
        feature_contributions = self.get_feature_contributions(prefix)
        pc_interpretations = self.get_pc_interpretation(prefix)

        # 메타데이터 생성
        variance_explained = {
            f'{prefix}_variance_explained': {
                'total_variance_preserved': float(cumulative_variance_ratio[n_components - 1]),
                'n_components': int(n_components),
                'component_variance_ratios': pca.explained_variance_ratio_[:n_components].tolist(),
                'feature_contributions': feature_contributions,
                'pc_interpretations': pc_interpretations
            }
        }

        return pca_df, variance_explained


class CustomJSONEncoder(json.JSONEncoder):
    """날짜와 NumPy 타입을 처리할 수 있는 커스텀 JSON 인코더"""

    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super().default(obj)



# AI Stock Advisor System class
class AIStockAdvisor:
    def __init__(self, stock: str, lang='en'):
        self.stock = stock
        self.lang = lang
        self.logger = logging.getLogger(f"{stock}_analyzer")
        self.login = self._get_login()
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.db_connection = self._setup_database('ai_stock_analysis_records.db')
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _get_login(self):
        try:
            totp = pyotp.TOTP(Config.ROBINHOOD_TOTP_CODE).now()
            self.logger.info(f"Current OTP: {totp}")
            login = r.robinhood.login(
                Config.ROBINHOOD_USERNAME,
                Config.ROBINHOOD_PASSWORD,
                mfa_code=totp
            )
            if login:
                self.logger.info("Successfully logged in to Robinhood")
                return login
            else:
                raise Exception("Login failed")
        except Exception as e:
            self.logger.error(f"Login error: {str(e)}")
            raise

    def get_current_price(self):
        self.logger.info(f"Fetching current price for {self.stock}")
        try:
            # Yahoo Finance에서 시도
            ticker = yf.Ticker(self.stock)
            current_price = round(ticker.info['regularMarketPrice'], 2)
            self.logger.info(f"Current price from Yahoo Finance for {self.stock}: ${current_price:.2f}")
            return current_price
        except Exception as e:
            self.logger.warning(f"Error fetching from Yahoo Finance: {str(e)}. Trying Robinhood...")
            try:
                # Robinhood에서 시도
                quote = r.robinhood.stocks.get_latest_price(self.stock)
                current_price = round(float(quote[0]), 2)
                self.logger.info(f"Current price from Robinhood for {self.stock}: ${current_price:.2f}")
                return current_price
            except Exception as e:
                self.logger.error(f"Error fetching from both sources: {str(e)}")
                return None

    def get_chart_data(self):
        self.logger.info(f"Fetching chart data for {self.stock}")
        try:
            # Yahoo Finance에서 데이터 가져오기
            ticker = yf.Ticker(self.stock)

            # 일일 데이터
            daily_data = ticker.history(period="1d", interval="5m")
            daily_historicals = [{
                'begins_at': date.strftime('%Y-%m-%d %H:%M'),
                'open_price': str(row['Open']),
                'close_price': str(row['Close']),
                'high_price': str(row['High']),
                'low_price': str(row['Low']),
                'volume': str(row['Volume'])
            } for date, row in daily_data.iterrows()]

            # 월간 데이터
            monthly_data = ticker.history(period="1mo", interval="1d")
            monthly_historicals = [{
                'begins_at': date.strftime('%Y-%m-%d'),
                'open_price': str(row['Open']),
                'close_price': str(row['Close']),
                'high_price': str(row['High']),
                'low_price': str(row['Low']),
                'volume': str(row['Volume'])
            } for date, row in monthly_data.iterrows()]

            if not monthly_historicals or not daily_historicals:
                raise Exception("No data from Yahoo Finance")

        except Exception as e:
            self.logger.warning(f"Failed to get data from Yahoo Finance : {e}. Trying Robinhood...")

            # Robinhood에서 데이터 가져오기 시도
            monthly_historicals = r.robinhood.stocks.get_stock_historicals(
                self.stock, interval="day", span="month", bounds="regular"
            )

            daily_historicals = r.robinhood.stocks.get_stock_historicals(
                self.stock, interval="5minute", span="day", bounds="regular"
            )

        print("daily_historicals: ",daily_historicals)
        print("monthly_historicals: ",monthly_historicals)

        return self._add_indicators(monthly_historicals, daily_historicals)

    def _add_indicators(self, monthly_data, daily_data):
        """일일 및 월간 기술적 지표 계산 - 최적화 버전"""
        try:
            logger.info("Adding technical indicators")

            def create_dataframe(data, timeframe='daily'):
                """데이터프레임 생성 및 포맷팅"""
                if not data:
                    return pd.DataFrame()

                df = pd.DataFrame(data)
                column_mapping = {
                    'begins_at': 'Date',
                    'open_price': 'Open',
                    'close_price': 'Close',
                    'high_price': 'High',
                    'low_price': 'Low',
                    'volume': 'Volume'
                }
                df = df.rename(columns=column_mapping)

                # 날짜 형식 변환
                if timeframe == 'daily':
                    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M')
                else:  # monthly
                    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

                # OHLC 가격 데이터는 소수점 2자리로 고정
                price_columns = ['Open', 'Close', 'High', 'Low']
                for col in price_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Volume은 정수로 변환
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').astype(int)

                df.set_index('Date', inplace=True)
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]


            # 볼린저 밴드 함수 추가
            def add_bollinger_bands(df, window=20, num_std=2):
                df['BB_Middle'] = df['Close'].rolling(window=window).mean().round(4)
                bb_std = df['Close'].rolling(window=window).std()
                df['BB_Upper'] = (df['BB_Middle'] + (bb_std * num_std)).round(4)
                df['BB_Lower'] = (df['BB_Middle'] - (bb_std * num_std)).round(4)
                df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100).round(4)
                return df

            # ADX 함수 추가
            def add_adx(df, period=14):
                # True Range 계산
                df['TR'] = pd.DataFrame([
                    df['High'] - df['Low'],
                    abs(df['High'] - df['Close'].shift(1)),
                    abs(df['Low'] - df['Close'].shift(1))
                ]).max()

                # +DM, -DM 계산
                df['Plus_DM'] = np.where(
                    (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                    np.maximum(df['High'] - df['High'].shift(1), 0),
                    0
                )
                df['Minus_DM'] = np.where(
                    (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                    np.maximum(df['Low'].shift(1) - df['Low'], 0),
                    0
                )

                # ATR과 DI 계산
                atr = df['TR'].ewm(span=period, adjust=False).mean()
                plus_di = (df['Plus_DM'].ewm(span=period, adjust=False).mean() / atr * 100)
                minus_di = (df['Minus_DM'].ewm(span=period, adjust=False).mean() / atr * 100)

                # ADX 계산
                df['ADX'] = (abs(plus_di - minus_di) / (plus_di + minus_di) * 100) \
                    .ewm(span=period, adjust=False).mean().round(4)

                # 중간 계산 컬럼 제거
                df.drop(['TR', 'Plus_DM', 'Minus_DM'], axis=1, inplace=True)
                return df

            daily_df = create_dataframe(daily_data, timeframe='daily')
            monthly_df = create_dataframe(monthly_data, timeframe='monthly')

            # 1. 일일 데이터 핵심 지표 계산
            if not daily_df.empty:
                logger.info("Processing daily indicators")
                # 주요 이동평균선 (소수점 4자리)
                daily_df['MA5'] = daily_df['Close'].rolling(window=5).mean().round(4)
                daily_df['MA20'] = daily_df['Close'].rolling(window=20).mean().round(4)

                # MACD (소수점 4자리)
                ema12 = daily_df['Close'].ewm(span=12, adjust=False).mean()
                ema26 = daily_df['Close'].ewm(span=26, adjust=False).mean()
                daily_df['MACD'] = (ema12 - ema26).round(4)
                daily_df['Signal_Line'] = daily_df['MACD'].ewm(span=9, adjust=False).mean().round(4)

                # RSI (소수점 4자리)
                delta = daily_df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                daily_df['RSI'] = (100 - (100 / (1 + gain / loss))).round(4)

                # 거래량 동향 (소수점 4자리)
                volume_ma20 = daily_df['Volume'].rolling(window=20).mean()
                daily_df['Volume_Trend'] = (daily_df['Volume'] / volume_ma20).round(4)

                # 볼린저 밴드 추가
                daily_df = add_bollinger_bands(daily_df)

                # ADX 추가
                daily_df = add_adx(daily_df)

                # OHLC 데이터 소수점 2자리로 설정
                daily_df[['Open', 'Close', 'High', 'Low']] = daily_df[['Open', 'Close', 'High', 'Low']].round(2)

            # 2. 월간 데이터 핵심 지표 계산
            if not monthly_df.empty:
                logger.info("Processing monthly indicators")
                # 중장기 이동평균 (소수점 4자리)
                monthly_df['MA20'] = monthly_df['Close'].rolling(window=20).mean().round(4)

                # 월간 RSI (소수점 4자리)
                delta = monthly_df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                monthly_df['Monthly_RSI'] = (100 - (100 / (1 + gain / loss))).round(4)

                # 거래량 동향 (소수점 4자리)
                volume_ma20 = monthly_df['Volume'].rolling(window=20).mean()
                monthly_df['Volume_Trend'] = (monthly_df['Volume'] / volume_ma20).round(4)

                # 수익률 (소수점 4자리)
                monthly_df['Monthly_Return'] = monthly_df['Close'].pct_change(20).round(4)

                # 월간 데이터에도 볼린저 밴드와 ADX 추가
                monthly_df = add_bollinger_bands(monthly_df)
                monthly_df = add_adx(monthly_df)

                # OHLC 데이터 소수점 2자리로 설정
                monthly_df[['Open', 'Close', 'High', 'Low']] = monthly_df[['Open', 'Close', 'High', 'Low']].round(2)

            logger.info("Completed adding optimized indicators")
            return monthly_df, daily_df

        except Exception as e:
            logger.error(f"Error in _add_indicators: {str(e)}", exc_info=True)
            return pd.DataFrame(), pd.DataFrame()

    def get_vix_index(self):
        self.logger.info("Fetching VIX INDEX data")
        try:
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1d")
            current_vix = round(vix_data['Close'].iloc[-1], 2)
            self.logger.info(f"Current VIX INDEX: {current_vix}")
            return current_vix
        except Exception as e:
            self.logger.error(f"Error fetching VIX INDEX: {str(e)}")
            return None

    def get_fear_and_greed_index(self):
        self.logger.info("Fetching Fear and Greed Index")
        fgi = fear_and_greed.get()
        return {
            "value": fgi.value,
            "description": fgi.description,
            "last_update": fgi.last_update.isoformat()
        }

    def get_news(self):
        yahoo_finance_news = self._get_news_from_yahoo()
        alpha_vantage_news = self._get_news_from_alpha_vantage()

        return {
            "yahoo_finance_news": yahoo_finance_news,
            "alpha_vantage_news": alpha_vantage_news,
        }

    def _get_news_from_yahoo(self):
        self.logger.info(f"Fetching news from Yahoo Finance for {self.stock}")
        try:
            # Yahoo Finance Ticker 객체 생성
            ticker = yf.Ticker(self.stock)

            # 뉴스 데이터 가져오기
            news_data = ticker.news
            news_items = []

            # 최대 5개의 뉴스 항목 처리
            for item in news_data[:5]:
                # Unix timestamp를 datetime으로 변환
                news_date = datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S')

                news_items.append({
                    "title": item['title'],
                    "date": news_date,
                    "url": item.get('link', ''),
                    "source": 'Yahoo Finance'
                })

            self.logger.info(f"Retrieved {len(news_items)} news items from Yahoo Finance")
            return news_items
        except Exception as e:
            self.logger.error(f"Error fetching news from Yahoo Finance: {str(e)}")
            return []

    def _get_news_from_alpha_vantage(self):
        self.logger.info("Fetching news from Alpha Vantage")
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.stock}&apikey={Config.ALPHA_VANTAGE_API_KEY}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            news_items = []
            if "feed" in data:
                for item in data["feed"][:5]:
                    time_published = item.get("time_published", "No date")
                    if time_published != "No date":
                        try:
                            dt = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                            time_published = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            pass

                    news_items.append({
                        "title": item.get("title", "No title"),
                        "date": time_published,
                        "url": item.get("url", ""),
                        "source": 'Alpha Vantage'
                    })
            self.logger.info(f"Retrieved {len(news_items)} news items from Alpha Vantage")
            return news_items  # 딕셔너리 리스트 직접 반환
        except Exception as e:
            self.logger.error(f"Error during Alpha Vantage API request: {e}")
            return []

    def _translate_to_language(self, text, lang='en'):
        # 언어 코드를 소문자로 변환
        lang = lang.lower()
        self.logger.info(f"Translating text to {lang}")
        try:
            translator = GoogleTranslator(source='auto', target=lang)
            translated = translator.translate(text)
            self.logger.info("Translation successful")
            return translated
        except Exception as e:
            self.logger.error(f"Error during translation: {e}")
            return text

    def _setup_database(self, db_name: str):
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        if db_name == 'ai_stock_analysis_records.db':
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_stock_analysis_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Stock TEXT,
                Time DATETIME,
                Decision TEXT,
                Percentage INTEGER,
                Reason TEXT,
                CurrentPrice REAL,
                ExpectedNextDayPrice REAL,
                ExpectedPriceDifference REAL
            )
            ''')

        conn.commit()
        self.logger.info(f"Database {db_name} setup completed")
        return conn

    def _record_trading_decision(self, decision: Dict[str, Any]):
        time_ = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_price = decision['CurrentPrice']
        expected_next_day_price = decision['ExpectedNextDayPrice']
        expected_price_difference = round(expected_next_day_price - current_price, 2)

        cursor = self.db_connection.cursor()
        cursor.execute('''
        INSERT INTO ai_stock_analysis_records 
        (Stock, Time, Decision, Percentage, Reason, CurrentPrice, ExpectedNextDayPrice, ExpectedPriceDifference)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.stock,
            time_,
            decision['Decision'],
            decision['Percentage'],
            decision['Reason'],
            current_price,
            expected_next_day_price,
            expected_price_difference
        ))
        self.db_connection.commit()


    def ai_stock_analysis(self):
        monthly_df, daily_df = self.get_chart_data()

        print("daily_df1111:", daily_df)
        print("monthly_df11111:", monthly_df)

        # # 데이터 전처리 및 분석 준비
        # daily_df, monthly_df = self.prepare_data_for_analysis(daily_df, monthly_df)
        #
        # print("daily_df22222:", daily_df)
        # print("monthly_df2222:", monthly_df)



        news = self.get_news()
        f = open("strategy.txt", "r")
        strategy = f.read()
        f.close()

        fgi = self.get_fear_and_greed_index()
        current_price = self.get_current_price()
        vix_index = self.get_vix_index()

        if current_price is None:
            self.logger.error("Failed to get current price. Aborting analysis.")
            return None, None, None, None, None, None

        # 뉴스 데이터의 날짜 처리
        for source in news.values():
            for item in source:
                if isinstance(item.get('date'), (date, datetime)):
                    item['date'] = item['date'].isoformat()

        # Fear & Greed Index의 날짜 처리
        if isinstance(fgi.get('last_update'), (date, datetime)):
            fgi['last_update'] = fgi['last_update'].isoformat()


def main():

    advisor = AIStockAdvisor(stock="TSLA")
    advisor.ai_stock_analysis()


if __name__ == "__main__":
    main()