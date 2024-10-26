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
from fastapi.encoders import jsonable_encoder


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

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context


class CustomHttpAdapter(HTTPAdapter):
    def __init__(self, *args, **kwargs):
        self.ssl_context = create_urllib3_context(
            ssl_version=None,
            cert_reqs=None,
            options=None
        )
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = self.ssl_context
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs['ssl_context'] = self.ssl_context
        return super().proxy_manager_for(*args, **kwargs)


def create_session():
    session = requests.Session()
    adapter = CustomHttpAdapter()
    session.mount('https://', adapter)
    return session


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


# AI Stock Advisor System class
class AIStockAdvisor:
    def __init__(self, stock: str, lang='en'):
        self.stock = stock
        self.lang = lang
        self.logger = logging.getLogger(f"{stock}_analyzer")
        self.login = self._get_login()
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.db_connection = self._setup_database('ai_stock_analysis_records.db')
        self.performance_db_connection = self._setup_database('ai_stock_performance.db')
        self._migrate_and_update_performance_data()
        self.session = create_session()

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
            quote = r.robinhood.stocks.get_latest_price(self.stock)
            current_price = round(float(quote[0]), 2)
            self.logger.info(f"Current price for {self.stock}: ${current_price:.2f}")
            return current_price
        except Exception as e:
            self.logger.error(f"Error fetching current price: {str(e)}")
            return None

    def get_chart_data(self):
        self.logger.info(f"Fetching chart data for {self.stock}")
        monthly_historicals = r.robinhood.stocks.get_stock_historicals(
            self.stock, interval="day", span="3month", bounds="regular"
        )
        daily_historicals = r.robinhood.stocks.get_stock_historicals(
            self.stock, interval="5minute", span="day", bounds="regular"
        )
        monthly_df = self._process_df(monthly_historicals)
        daily_df = self._process_df(daily_historicals)
        return self._add_indicators(monthly_df, daily_df)

    def _process_df(self, historicals):
        """
        히스토리컬 데이터를 DataFrame으로 변환
        """
        try:
            if not historicals:
                logger.error("No historical data received")
                return pd.DataFrame()

            df = pd.DataFrame(historicals)
            logger.info(f"Original columns: {df.columns.tolist()}")

            # 컬럼 매핑
            column_mapping = {
                'begins_at': 'Date',
                'open_price': 'Open',
                'close_price': 'Close',
                'high_price': 'High',
                'low_price': 'Low',
                'volume': 'Volume',
                'open': 'Open',
                'close': 'Close',
                'high': 'High',
                'low': 'Low'
            }

            # 사용 가능한 컬럼 찾기
            available_columns = [col for col in column_mapping.keys() if col in df.columns]

            if not available_columns:
                logger.error(f"No matching columns found. Available columns: {df.columns.tolist()}")
                raise ValueError("No matching columns found in historical data")

            # 컬럼 선택 및 이름 변경
            df = df[available_columns]
            df.columns = [column_mapping[col] for col in df.columns]

            # 날짜 처리
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

            # 데이터 타입 변환
            for col in ['Open', 'Close', 'High', 'Low']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            if 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)

            logger.info(f"Processed columns: {df.columns.tolist()}")
            return df

        except Exception as e:
            logger.error(f"Error in _process_df: {str(e)}", exc_info=True)
            raise

    def _add_indicators(self, monthly_df, daily_df):
        """
        모든 기술적 지표 추가
        """
        try:
            logger.info("Adding technical indicators")
            logger.info(f"Monthly DataFrame columns before indicators: {monthly_df.columns.tolist()}")
            logger.info(f"Daily DataFrame columns before indicators: {daily_df.columns.tolist()}")

            for df in [monthly_df, daily_df]:
                if not df.empty:
                    df = self._calculate_bollinger_bands(df)
                    df = self._calculate_rsi(df)
                    df = self._calculate_macd(df)

            if not monthly_df.empty:
                monthly_df = self._calculate_moving_averages(monthly_df)

            logger.info(f"Monthly DataFrame columns after indicators: {monthly_df.columns.tolist()}")
            logger.info(f"Daily DataFrame columns after indicators: {daily_df.columns.tolist()}")

            return monthly_df, daily_df
        except Exception as e:
            logger.error(f"Error in _add_indicators: {str(e)}", exc_info=True)
            return monthly_df, daily_df

    def _calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """
        볼린저 밴드 계산
        """
        try:
            # 종가 컬럼명 확인
            close_column = 'Close'

            if close_column not in df.columns:
                logger.error(f"Close price column not found. Available columns: {df.columns.tolist()}")
                return df

            df['SMA'] = df[close_column].rolling(window=window).mean()
            df['STD'] = df[close_column].rolling(window=window).std()
            df['Upper_Band'] = df['SMA'] + (df['STD'] * num_std)
            df['Lower_Band'] = df['SMA'] - (df['STD'] * num_std)
            return df
        except Exception as e:
            logger.error(f"Error in _calculate_bollinger_bands: {str(e)}")
            return df

    def _calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        RSI 계산
        """
        try:
            close_column = 'Close'

            if close_column not in df.columns:
                logger.error(f"Close price column not found. Available columns: {df.columns.tolist()}")
                return df

            delta = df[close_column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            return df
        except Exception as e:
            logger.error(f"Error in _calculate_rsi: {str(e)}")
            return df

    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        MACD 계산
        """
        try:
            close_column = 'Close'

            if close_column not in df.columns:
                logger.error(f"Close price column not found. Available columns: {df.columns.tolist()}")
                return df

            df['EMA_fast'] = df[close_column].ewm(span=fast, adjust=False).mean()
            df['EMA_slow'] = df[close_column].ewm(span=slow, adjust=False).mean()
            df['MACD'] = df['EMA_fast'] - df['EMA_slow']
            df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
            return df
        except Exception as e:
            logger.error(f"Error in _calculate_macd: {str(e)}")
            return df

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        이동평균선 계산
        """
        try:
            close_column = 'Close'

            if close_column not in df.columns:
                logger.error(f"Close price column not found. Available columns: {df.columns.tolist()}")
                return df

            windows = [10, 20, 60, 120]
            for window in windows:
                df[f'MA_{window}'] = df[close_column].rolling(window=window).mean()
            return df
        except Exception as e:
            logger.error(f"Error in _calculate_moving_averages: {str(e)}")
            return df

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
        google_news = self._get_news_from_google()
        alpha_vantage_news = self._get_news_from_alpha_vantage()
        robinhood_news = self._get_news_from_robinhood()

        return {
            "google_news": google_news,
            "alpha_vantage_news": alpha_vantage_news,
            "robinhood_news": robinhood_news
        }

    def _get_news_from_google(self):
        self.logger.info("Fetching news from Google")
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "api_key": Config.SERPAPI_API_KEY,
            "engine": "google_news",
            "q": self.stock,
            "num": 5
        }
        headers = {"Accept": "application/json"}
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            news_items = []
            for result in data.get('organic_results', [])[:5]:
                news_items.append({
                    "title": result['title'],
                    "date": result['date'],
                    "url": result.get('link', ''),
                    "source": 'Google News'
                })
            self.logger.info(f"Retrieved {len(news_items)} news items from Google")
            return news_items  # 딕셔너리 리스트 직접 반환
        except Exception as e:
            self.logger.error(f"Error during Google News API request: {e}")
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

    def _get_news_from_robinhood(self):
        self.logger.info("Fetching news from Robinhood")
        try:
            news_data = r.robinhood.stocks.get_news(self.stock)
            news_items = []
            for item in news_data[:5]:
                news_items.append({
                    "title": item['title'],
                    "date": item['published_at'],
                    "url": item.get('url', ''),
                    "source": 'Robinhood'
                })
            self.logger.info(f"Retrieved {len(news_items)} news items from Robinhood")
            return news_items  # 딕셔너리 리스트 직접 반환
        except Exception as e:
            self.logger.error(f"Error fetching news from Robinhood: {str(e)}")
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
        elif db_name == 'ai_stock_performance.db':
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock TEXT,
                date DATE,
                avg_current_price REAL,
                next_date DATE,
                avg_expected_next_day_price REAL,
                actual_next_day_price REAL,
                price_difference REAL,
                error_percentage REAL DEFAULT 0,
                count INTEGER DEFAULT 1
            )
            ''')

        conn.commit()
        self.logger.info(f"Database {db_name} setup completed")
        return conn

    def _migrate_and_update_performance_data(self):
        cursor_analysis = self.db_connection.cursor()
        cursor_performance = self.performance_db_connection.cursor()

        cursor_analysis.execute("""
        SELECT Stock, DATE(Time) as Date, CurrentPrice, ExpectedNextDayPrice
        FROM ai_stock_analysis_records
        """)
        records = cursor_analysis.fetchall()

        df = pd.DataFrame(records, columns=['Stock', 'Date', 'CurrentPrice', 'ExpectedNextDayPrice'])
        grouped = df.groupby(['Stock', 'Date'])

        aggregated = grouped.agg({
            'CurrentPrice': 'mean',
            'ExpectedNextDayPrice': 'mean',
            'Stock': 'count'
        }).rename(columns={'Stock': 'Count'})

        for (stock, date), row in aggregated.iterrows():
            next_date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

            cursor_performance.execute("""
            SELECT COUNT(*) FROM stock_performance
            WHERE stock = ? AND date = ?
            """, (stock, date))

            if cursor_performance.fetchone()[0] == 0:
                cursor_performance.execute("""
                INSERT INTO stock_performance
                (stock, date, next_date, avg_current_price, avg_expected_next_day_price, count)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (stock, date, next_date, row['CurrentPrice'], row['ExpectedNextDayPrice'], row['Count']))
                self.logger.info(f"Inserted new performance data for {stock} on {date}")

        self.performance_db_connection.commit()
        self._fetch_actual_stock_prices()
        self.logger.info("Performance data migration and update completed")

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

    def _fetch_actual_stock_prices(self):
        cursor = self.performance_db_connection.cursor()

        cursor.execute("""
        SELECT DISTINCT stock, date, next_date, avg_expected_next_day_price
        FROM stock_performance
        WHERE actual_next_day_price IS NULL
        """)
        stocks_to_update = cursor.fetchall()

        for stock, date, next_date, avg_expected_next_day_price in stocks_to_update:
            next_date = datetime.strptime(next_date, '%Y-%m-%d').date()

            if next_date > datetime.now().date():
                self.logger.info(f"Skipping future date for {stock}: {next_date}")
                continue

            try:
                ticker = yf.Ticker(stock)
                actual_price = None

                days_to_check = 5

                for i in range(days_to_check):
                    check_date = next_date + timedelta(days=i)
                    hist = ticker.history(start=check_date, end=check_date + timedelta(days=1))

                    if not hist.empty:
                        actual_price = round(hist['Close'].iloc[0], 2)
                        break

                if actual_price is not None:
                    price_difference = round(actual_price - avg_expected_next_day_price, 2)
                    error_percentage = abs(
                        round((price_difference / actual_price) * 100, 2)) if actual_price != 0 else 0

                    cursor.execute("""
                    UPDATE stock_performance
                    SET actual_next_day_price = ?,
                        price_difference = ?,
                        error_percentage = ?,
                        next_date = ?
                    WHERE stock = ? AND date = ?
                    """, (
                        actual_price, price_difference, error_percentage, check_date.strftime('%Y-%m-%d'), stock, date))
                    self.logger.info(
                        f"Updated actual price for {stock}. Original next_date: {next_date}, "
                        f"Updated next_date: {check_date}, Price: {actual_price:.2f}, "
                        f"Difference: {price_difference:.2f}, Error percentage: {error_percentage:.2f}%")
                else:
                    self.logger.warning(
                        f"No data available for {stock} within {days_to_check} days after {next_date}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {stock} starting from {next_date}: {str(e)}")

        self.performance_db_connection.commit()
        self.logger.info("Actual stock prices fetched and updated")

    def ai_stock_analysis(self):
        monthly_df, daily_df = self.get_chart_data()
        news = self.get_news()
        f = open("strategy.txt", "r")
        youtube_transcript = f.read()
        f.close()

        fgi = self.get_fear_and_greed_index()
        current_price = self.get_current_price()
        vix_index = self.get_vix_index()

        if current_price is None:
            self.logger.error("Failed to get current price. Aborting analysis.")
            return None, None, None, None, None, None

        self.logger.info("Sending request to OpenAI")
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert in Stock investing. Analyze the provided data including technical indicators, market data, recent news headlines, the Fear and Greed Index, YouTube video transcript, VIX INDEX, and the chart image. Tell me whether to buy, sell, or hold at the moment. Consider the following in your analysis:
                                - Technical indicators and market data
                                - Recent news headlines and their potential impact on Stock price
                                - The Fear and Greed Index and its implications
                                - VIX INDEX and its implications for market volatility
                                - Overall market sentiment
                                - Insights from the YouTube video transcript
                                - Current stock price: ${current_price}
                                - Current VIX INDEX: {vix_index}

                                Particularly important is to always refer to the trading method of 'Larry Williams', a legendary stock investor, to assess the current situation and make trading decisions. Larry Williams's trading method is as follows:

                                {youtube_transcript}

                                Based on this trading method, analyze the current market situation and make a judgment by synthesizing it with the provided data.

                                Additionally, predict the next day's closing price for the stock based on your analysis.

                                Respond with:
                                1. A decision (BUY, SELL, or HOLD)
                                2. If the decision is 'BUY' or 'SELL', provide an intensity expressed as a percentage ratio (1 to 100).
                                   If the decision is 'HOLD', set the percentage to 0.
                                3. A reason for your decision
                                4. A prediction for the next day's closing price

                                Ensure that the percentage is an integer between 1 and 100 for buy/sell decisions, and exactly 0 for hold decisions.
                                Your percentage should reflect the strength of your conviction in the decision based on the analyzed data.
                                The next day's closing price prediction should be a float value."""},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "stock": self.stock,
                                "monthly_data": monthly_df.to_json(),
                                "daily_data": daily_df.to_json(),
                                "fear_and_greed_index": fgi,
                                "vix_index": vix_index,
                                "news": news
                            })
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

class NewsItem(BaseModel):
    title: str
    date: str
    url: str
    source: str

    class Config:
        from_attributes = True


class NewsData(BaseModel):
    google_news: List[NewsItem]
    alpha_vantage_news: List[NewsItem]
    robinhood_news: List[NewsItem]


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

# 클라이언트 측에서 DataFrame을 재구성하기 위한 함수
def reconstruct_dataframe(df_model: DataFrameModel) -> pd.DataFrame:
    if df_model is None:
        return None

    df = pd.DataFrame(
        data=df_model.data,
        index=df_model.index,
        columns=df_model.columns
    )
    return df

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
