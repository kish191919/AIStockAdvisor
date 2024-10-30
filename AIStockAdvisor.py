import os
import logging
import sqlite3
import yfinance as yf
import pyotp
import robin_stocks as r
import fear_and_greed
import pandas as pd
import json
import requests
import numpy as np
import re


from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from openai import OpenAI
from deep_translator import GoogleTranslator
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from transformers import GPT2TokenizerFast
from datetime import date, datetime
from youtube_transcript_api import YouTubeTranscriptApi
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# 데이터프레임 출력 설정 변경
pd.set_option('display.max_columns', None)  # 모든 컬럼 표시
pd.set_option('display.width', 1000)        # 출력 폭 확대
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

        # 경고를 피하기 위해 ffill()과 bfill() 직접 사용
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

    # Fit PCA and transform data
    def fit_transform(self, df, prefix):

        # Prepare data
        numeric_df = self._prepare_dataframe(df)
        original_index = numeric_df.index

        # Store feature names
        self.feature_names[prefix] = numeric_df.columns.tolist()

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        self.scalers[prefix] = scaler

        # Apply PCA
        pca = PCA()
        transformed_data = pca.fit_transform(scaled_data)
        self.pcas[prefix] = pca

        # Determine number of components needed
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance_ratio >= self.variance_ratio_threshold) + 1

        # Create DataFrame with reduced components
        columns = [f'PC{i + 1}' for i in range(n_components)]
        pca_df = pd.DataFrame(
            transformed_data[:, :n_components],
            columns=columns,
            index=original_index
        )

        # Add variance explained information
        variance_explained = {
            f'{prefix}_variance_explained': {
                'total_variance_preserved': float(cumulative_variance_ratio[n_components - 1]),
                'n_components': int(n_components),
                'component_variance_ratios': pca.explained_variance_ratio_[:n_components].tolist()
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
            data = ticker.history(period='1d')
            if not data.empty:
                current_price = round(data['Close'].iloc[-1], 2)
                self.logger.info(f"Current price from Yahoo Finance for {self.stock}: ${current_price:.2f}")
                return current_price
            raise Exception("No data from Yahoo Finance")

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

        print("daily_historicals: ", daily_historicals)
        print("monthly_historicals: ", monthly_historicals)

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

    def optimize_input_data(self, data):
        # 1. 특수문자 및 불필요한 공백 제거
        def clean_text(text):
            # 연속된 공백을 하나로 변경
            text = re.sub(r'\s+', ' ', text)
            # 줄바꿈 제거
            text = text.replace('\n', ' ')
            # 양쪽 공백 제거
            text = text.strip()
            return text

        # 2. JSON 데이터 최적화
        def optimize_json(json_data):
            if isinstance(json_data, dict):
                return {k: optimize_json(v) for k, v in json_data.items() if v is not None}
            elif isinstance(json_data, list):
                return [optimize_json(item) for item in json_data]
            elif isinstance(json_data, str):
                return clean_text(json_data)
            elif isinstance(json_data, float):
                return round(json_data, 2)  # 소수점 2자리로 제한
            else:
                return json_data

        # 3. 문자열을 JSON으로 파싱 후 최적화
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return clean_text(data)

        # 4. 최적화 적용
        optimized_data = optimize_json(data)

        # 5. 최적화된 데이터를 compact JSON 문자열로 변환
        return json.dumps(optimized_data, separators=(',', ':'))

    def ai_stock_analysis(self):
        monthly_df, daily_df = self.get_chart_data()
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


        # 데이터 준비 및 JSON 직렬화
        input_data = json.dumps({
            "stock": self.stock,
            "daily_df": json.dumps(daily_df.to_dict(), cls=CustomJSONEncoder),
            "monthly_df": json.dumps(monthly_df.to_dict(), cls=CustomJSONEncoder),
            "fear_and_greed_index": fgi,
            "vix_index": vix_index,
            "news": news,
            "current_price": current_price
        }, cls=CustomJSONEncoder)

        print("input_data: ", input_data)
        optimized_input = self.optimize_input_data(input_data)

        # 토큰 수 계산 및 로깅
        system_prompt = f"""You are an expert in stock investing. Analyze the following elements: market data, recent news headlines, the Fear and Greed Index, and the VIX index.

Based on your analysis, including {strategy}, provide:

A decision (BUY, SELL, or HOLD).
An intensity level as a percentage (1 to 100) reflecting your conviction in the decision.
A reason for your decision based on reliable analysis.
A prediction for the next day's closing price.
Explain your analysis clearly and simply, suitable for someone new to stocks. If you mention terms like MA5, MA20, MACD, Signal Line, RSI, Volume Trend, Monthly RSI, or Monthly Return, please provide simple explanations for them."""

        system_tokens = self.count_tokens(system_prompt)
        input_tokens = self.count_tokens(input_data)
        optimized_input_tokens = self.count_tokens(optimized_input)
        strategy_tokens = self.count_tokens(strategy)

        self.logger.info(f"""Token counts:
                    System prompt: {system_tokens}
                    Input token : {input_tokens}
                    optimized_input_tokens: {optimized_input_tokens}
                    strategy : {strategy_tokens}
                    Total: {system_tokens + optimized_input_tokens + strategy_tokens }
                """)

        self.logger.info("Sending request to OpenAI")
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": optimized_input
                        }
                    ]
                }
            ],
            max_tokens=4095,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "trading_decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "decision": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                            "percentage": {"type": "integer"},
                            "reason": {"type": "string"},
                            "expected_next_day_price": {"type": "number"},
                        },
                        "required": ["decision", "percentage", "reason", "expected_next_day_price"],
                        "additionalProperties": False
                    }
                }
            }
        )
        result = TradingDecision.model_validate_json(response.choices[0].message.content)
        self.logger.info("Received response from OpenAI")

        reason_translated = self._translate_to_language(result.reason, self.lang)

        # Record the trading decision and current state
        self._record_trading_decision({
            'Decision': result.decision,
            'Percentage': result.percentage,
            'Reason': result.reason,
            'CurrentPrice': round(current_price, 2),
            'ExpectedNextDayPrice': round(result.expected_next_day_price, 2),
            'VIX_INDEX': vix_index
        })
        print("result:", result)
        print("input:",input_data )
        return result, reason_translated, news, fgi, current_price, vix_index, monthly_df, daily_df


# FastAPI 앱 생성
app = FastAPI()

# Request 모델
class StockAnalysisRequest(BaseModel):
    symbol: str
    language: str = "en"

class DataFrameModel(BaseModel):
    index: List[str]
    columns: List[str]
    data: List[List[Any]]

# Response 모델
class StockAnalysisResponse(BaseModel):
    decision: str
    percentage: int
    reason: str
    current_price: float
    expected_next_day_price: float
    vix_index: Optional[float]
    fear_greed_index: Dict[str, Any]
    news: Dict[str, List[Dict[str, str]]]  # NewsItem 대신 Dict 사용
    monthly_df: Optional[DataFrameModel] = None
    daily_df: Optional[DataFrameModel] = None

    class Config:
        from_attributes = True

def serialize_dataframe(df: pd.DataFrame) -> Optional[DataFrameModel]:
    if df is None:
        return None

    return DataFrameModel(
        index=[str(idx) for idx in df.index],
        columns=df.columns.tolist(),
        data=df.values.tolist()
    )

@app.get("/")
async def root():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# API 엔드포인트 핸들러 수정
@app.post("/api/analyze", response_model=StockAnalysisResponse)
async def analyze_stock(request: StockAnalysisRequest):
    try:
        logger.info(f"Analyzing stock {request.symbol}")
        advisor = AIStockAdvisor(request.symbol, request.language)

        # 분석 실행
        result, reason_translated, news, fgi, current_price, vix_index, monthly_df, daily_df = advisor.ai_stock_analysis()

        print(monthly_df)
        print(daily_df)

        # DataFrame 직렬화
        monthly_df_serialized = serialize_dataframe(monthly_df)
        daily_df_serialized = serialize_dataframe(daily_df)

        # 응답 생성
        response = StockAnalysisResponse(
            decision=result.decision,
            percentage=result.percentage,
            reason=reason_translated,
            current_price=current_price,
            expected_next_day_price=result.expected_next_day_price,
            vix_index=vix_index,
            fear_greed_index=fgi,
            news=news,  # 이미 딕셔너리 형태로 되어 있음
            monthly_df=monthly_df_serialized,
            daily_df=daily_df_serialized
        )

        return response

    except Exception as e:
        logger.error(f"Error analyzing stock {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 에러 핸들러
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "AIStockAdvisor:app",  # 파일명:app_변수명
        host="0.0.0.0",
        port=8000,
        reload=False
    )